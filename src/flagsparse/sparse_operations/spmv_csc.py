# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native CSC SpMV kernels and public helpers."""

from ._common import *

import triton
import triton.language as tl

# Resolved once at import: the runtime cannot change under a live process, and
# prepare_spmv_csc sits on the per-call path.
_SPMV_CSC_CUDA = _backend_name() == "cuda"

SUPPORTED_SPMV_CSC_VALUE_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_CSC_OP_NON = 0
SPMV_CSC_OP_TRANS = 1
SPMV_CSC_OP_CONJ_TRANS = 2
SPMV_CSC_OP_NAMES = {
    SPMV_CSC_OP_NON: "non",
    SPMV_CSC_OP_TRANS: "trans",
    SPMV_CSC_OP_CONJ_TRANS: "conj",
}
_SPMV_CSC_OP_NAME_TO_CODE = {name: code for code, name in SPMV_CSC_OP_NAMES.items()}


def _spmv_csc_dtype_error_message():
    return "CSC SpMV supports float16, float32, float64, complex64, and complex128"


def _normalize_spmv_csc_op(op=None, transpose=False):
    if op is None:
        return SPMV_CSC_OP_TRANS if bool(transpose) else SPMV_CSC_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_CSC_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_CSC_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_CSC_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_csc_op_to_name(op):
    return SPMV_CSC_OP_NAMES[_normalize_spmv_csc_op(op)]


def _spmv_csc_op_transposes(op):
    return _normalize_spmv_csc_op(op) in (
        SPMV_CSC_OP_TRANS,
        SPMV_CSC_OP_CONJ_TRANS,
    )


def _normalize_spmv_csc_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedCscSpmv:
    """Prepared CSC metadata for repeated SpMV calls."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "block_nnz",
        "max_segments",
        "col_lengths",
        "max_col_nnz",
        "col_ids",
        "csr_delegate",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
        "launch_backend",
        "device_warp_size",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        block_nnz,
        max_segments,
        max_col_nnz,
        col_lengths=None,
        op=None,
        transpose=False,
        col_ids=None,
        csr_delegate=None,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
        launch_backend=None,
        device_warp_size=None,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.nnz = int(data.numel())
        self.block_nnz = int(block_nnz)
        self.max_segments = int(max_segments)
        if col_lengths is None:
            col_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
        self.col_lengths = col_lengths
        self.col_ids = col_ids
        self.csr_delegate = csr_delegate
        self.max_col_nnz = int(max_col_nnz)
        self.op = _normalize_spmv_csc_op(op, transpose=transpose)
        self.transpose = _spmv_csc_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason
        info = _get_device_backend_info(data.device)
        self.launch_backend = launch_backend or info["backend"]
        self.device_warp_size = int(
            device_warp_size
            if device_warp_size is not None
            else info["device_warp_size"]
        )


@triton.jit
def _spmv_csc_non_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    alpha,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
):
    """y += alpha * A @ x, accumulated with atomics.

    Only alpha lives here: this route scatters into y, so ``beta * y`` cannot be
    folded into the store and the C API applies it with a separate prologue
    (``_dense_scale_kernel``) before launching. This module's callers pass
    alpha = 1 into a zeroed y, which is the code this kernel always generated.
    """
    col = tl.program_id(0)
    seg = tl.program_id(1)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + col)
    tl.atomic_add(y_ptr + rows, alpha * vals * x_val, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_non_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    alpha_re,
    alpha_im,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
):
    """Complex counterpart; see _spmv_csc_non_real_kernel for why beta is absent."""
    col = tl.program_id(0)
    seg = tl.program_id(1)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    x_re = tl.load(x_ri_ptr + col * 2)
    x_im = tl.load(x_ri_ptr + col * 2 + 1)
    prod_re = a_re * x_re - a_im * x_im
    prod_im = a_re * x_im + a_im * x_re
    out_re = alpha_re * prod_re - alpha_im * prod_im
    out_im = alpha_re * prod_im + alpha_im * prod_re
    tl.atomic_add(y_ri_ptr + rows * 2, out_re, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + rows * 2 + 1, out_im, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_trans_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    alpha,
    beta,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """y = alpha * op(A) * x + beta * y, one program per column.

    Unlike the op="non" route this one is deterministic and writes every column
    exactly once, so both scalars fold into the store; no prologue is needed and
    an empty column still gets beta * y. This module's callers pass 1 and 0.
    """
    col = tl.program_id(0)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    # Trip count comes from *this* column, not from the global longest one.
    # ``MAX_SEGMENTS`` is ceil(max_col_nnz / BLOCK_NNZ), a constexpr, so
    # ``for seg in range(MAX_SEGMENTS)`` made every column execute the same fixed number
    # of BLOCK_NNZ-wide masked loads no matter how short it is.  The waste is
    # ``MAX_SEGMENTS * BLOCK_NNZ / mean_col_nnz``: ~335x on amazon0601 (mean 8.4,
    # longest 2751) and ~93x on TSOPF_FS_b300_c1.  It is the same rectangle the
    # op="non" path had in its grid, just moved inside the kernel.
    # MAX_SEGMENTS is kept as a parameter so the launch signature and any explicit
    # max_segments override stay valid; it now only bounds the loop.
    n_seg = min(tl.cdiv(end - start, BLOCK_NNZ), MAX_SEGMENTS)
    for seg in tl.range(0, n_seg):
        offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        rows = tl.load(indices_ptr + offs, mask=mask, other=0)
        vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + rows, mask=mask, other=0.0)
        acc = acc + tl.sum(tl.where(mask, vals * x_vals, 0.0))
    out = alpha * acc
    if HAS_BETA:
        out = out + beta * tl.load(y_ptr + col)
    tl.store(y_ptr + col, out)


@triton.jit
def _spmv_csc_trans_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    alpha_re,
    alpha_im,
    beta_re,
    beta_im,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    CONJ: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Complex counterpart; see _spmv_csc_trans_real_kernel."""
    col = tl.program_id(0)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    acc_re = tl.load(data_ri_ptr + start * 2, mask=start < end, other=0.0) * 0
    acc_im = tl.load(data_ri_ptr + start * 2 + 1, mask=start < end, other=0.0) * 0
    # Same dynamic trip count as the real kernel; see there for the measurement.
    n_seg = min(tl.cdiv(end - start, BLOCK_NNZ), MAX_SEGMENTS)
    for seg in tl.range(0, n_seg):
        offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        rows = tl.load(indices_ptr + offs, mask=mask, other=0)
        a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
        a_im_raw = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
        if CONJ:
            a_im = -a_im_raw
        else:
            a_im = a_im_raw
        x_re = tl.load(x_ri_ptr + rows * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + rows * 2 + 1, mask=mask, other=0.0)
        prod_re = a_re * x_re - a_im * x_im
        prod_im = a_re * x_im + a_im * x_re
        acc_re = acc_re + tl.sum(tl.where(mask, prod_re, 0.0))
        acc_im = acc_im + tl.sum(tl.where(mask, prod_im, 0.0))
    out_re = alpha_re * acc_re - alpha_im * acc_im
    out_im = alpha_re * acc_im + alpha_im * acc_re
    if HAS_BETA:
        prev_re = tl.load(y_ri_ptr + col * 2)
        prev_im = tl.load(y_ri_ptr + col * 2 + 1)
        out_re = out_re + beta_re * prev_re - beta_im * prev_im
        out_im = out_im + beta_re * prev_im + beta_im * prev_re
    tl.store(y_ri_ptr + col * 2, out_re)
    tl.store(y_ri_ptr + col * 2 + 1, out_im)


def _prepare_spmv_csc_matrix(data, indices, indptr, shape):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be nonnegative")
    if indptr.numel() != n_cols + 1:
        raise ValueError(
            f"indptr length must be n_cols+1={n_cols + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if not all(_is_accel_tensor(t) for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMV_CSC_VALUE_DTYPES:
        raise TypeError(_spmv_csc_dtype_error_message())
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    if indptr.numel() > 0:
        if int(indptr[0].item()) != 0:
            raise ValueError("indptr must start at zero")
        if int(indptr[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if indptr.numel() > 1 and torch.any(indptr[1:] < indptr[:-1]).item():
            raise ValueError("indptr must be non-decreasing")
    if data.numel() > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_rows:
            raise IndexError("indices out of range for n_rows")
    col_lengths = indptr[1:] - indptr[:-1]
    max_col_nnz = int(col_lengths.max().item()) if n_cols > 0 else 0
    return data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz


def prepare_spmv_csc(
    data,
    indices,
    indptr,
    shape,
    block_nnz=256,
    max_segments=None,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
    alg=None, config=None, _validated=None, _allow_delegate=True,
):
    if alg is None and _validated is None and data.dtype == torch.float16:
        alg = "spmv_csc_base"
    if alg is not None:
        return _prepare_csc_registered(data, indices, indptr, shape, op, transpose, alg, config, index_fallback_policy, "spmv",
                                       {"block_nnz": block_nnz, "max_segments": max_segments})
    if config is not None:
        raise ValueError("config requires an explicit algorithm")
    index_fallback_policy = _normalize_spmv_csc_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_csc_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_CSC_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz = (
        _prepare_spmv_csc_matrix(data, indices, indptr, shape) if _validated is None else _validated
    )
    block_nnz_use = int(block_nnz)
    if block_nnz_use <= 0:
        raise ValueError("block_nnz must be positive")
    launch = _spmv_rocm_launch_overrides(
        fmt="csc",
        dtype=data.dtype,
        max_row_nnz=max_col_nnz,
        nnz=data.numel(),
        block_nnz=block_nnz_use,
        device=data.device,
    )
    launch_backend = None
    device_warp_size = None
    if launch is not None:
        block_nnz_use = int(launch["block_nnz"])
        launch_backend = launch["backend"]
        device_warp_size = launch["device_warp_size"]
    if max_segments is None:
        max_segments_use = max((max_col_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max(
                (max_col_nnz + block_nnz_use - 1) // block_nnz_use,
                1,
            )
    else:
        max_segments_use = max(1, int(max_segments))
    # Per-nonzero owning column, for the nnz-parallel op="non" kernel.  Built once here
    # (a structural conversion, so outside any timed window) rather than binary-searched
    # per launch; see _spmv_csc_non_real_nnzpar_kernel for why.
    col_ids = None
    if int(data.numel()) > 0 and not _spmv_csc_op_transposes(op_code):
        try:
            # Deferred import: sddmm_csr only depends on _common, so there is no cycle,
            # but keeping it local avoids adding a load-time edge between operators.
            from .sddmm_csr import _build_row_ids

            col_ids = _build_row_ids(indptr, int(data.numel()))
        except Exception:
            col_ids = None
    # op="trans"/"conj": CSC(A) and CSR(A.T) are the same three arrays, so the
    # transposed product is a plain CSR SpMV -- and spmv_csr has the tuned CSR-Vector
    # bucket machinery this module never grew.  The bespoke CSC trans kernel runs one
    # program per column with a serial segment loop, which on a mean-8-nnz matrix is
    # mostly masked-off loads; delegating measured 4.39x geomean over 240 cases
    # (fp64 5.73x, complex128 8.22x) with 16 mild regressions, worst 0.75x.
    # prepare_spmv_csr already uses this same transpose-in-prepare technique for its own
    # trans/conj, so the timing convention matches the cuSPARSE baseline, which
    # materialises A.conj().T once outside the timed window.
    # CUDA only: this swaps which kernel actually runs, and the CSR-Vector bucket tiers
    # it lands on were tuned per backend separately.  Elsewhere the original CSC
    # transpose kernel is kept.
    csr_delegate = None
    if (
        _allow_delegate and _SPMV_CSC_CUDA
        and _spmv_csc_op_transposes(op_code)
        and int(data.numel()) > 0
    ):
        try:
            from .spmv_csr import prepare_spmv_csr

            delegate_values = data
            if op_code == SPMV_CSC_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
                delegate_values = data.conj()
                if hasattr(delegate_values, "resolve_conj"):
                    delegate_values = delegate_values.resolve_conj()
                delegate_values = delegate_values.contiguous()
            csr_delegate = prepare_spmv_csr(
                delegate_values,
                indices,
                indptr,
                (n_cols, n_rows),
                op="non",
                index_fallback_policy=index_fallback_policy,
            )
        except Exception:
            csr_delegate = None
    return PreparedCscSpmv(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_col_nnz=max_col_nnz,
        col_lengths=col_lengths,
        col_ids=col_ids,
        csr_delegate=csr_delegate,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
        launch_backend=launch_backend,
        device_warp_size=device_warp_size,
    )


def _validate_spmv_csc_x(x, prepared, op_code):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    expected = prepared.n_rows if _spmv_csc_op_transposes(op_code) else prepared.n_cols
    if x.numel() != expected:
        raise ValueError(f"x length must be {expected}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


@triton.jit
def _spmv_csc_non_real_nnzpar_kernel(
    data_ptr,
    indices_ptr,
    col_ids_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    """One program per BLOCK nonzeros, independent of n_cols and max_col_nnz.

    The segmented kernel above launches ``grid = (n_cols, max_segments)`` with
    ``max_segments = ceil(max_col_nnz / BLOCK_NNZ)``, so the grid is a *rectangle sized
    by the longest column* while the useful work is only ``nnz / BLOCK_NNZ``.  The waste
    factor is roughly ``max_col_nnz / mean_col_nnz``: 327x on amazon0601 (403k columns,
    8.4 nonzeros each, longest 2751) and 93x on TSOPF_FS_b300_c1.  Tuning BLOCK_NNZ
    cannot fix that -- raising it to shrink the rectangle costs more than it saves, and
    a full sweep put the per-matrix oracle at only 1.09-1.38x over the current default.

    Flattening the grid over nonzeros removes the rectangle entirely.  Measured over the
    30-matrix corpus, fp32: **0.338 -> 1.403 of cupy's CSC SpMV**, with 21/30 matrices
    below 0.8x falling to 3/30, and up to 67.7x on a single matrix (wiki-Talk 14.52ms ->
    0.21ms).  BLOCK=256 is within 1% of the per-matrix oracle, so no adaptive rule is
    needed here.

    ``col_ids`` is built once in prepare by the same binary search the SDDMM path uses.
    Inlining that search into this kernel was measured a net loss there (0.710x kernel
    geomean), so it stays a precomputed array.
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < nnz
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    cols = tl.load(col_ids_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    xs = tl.load(x_ptr + cols, mask=mask, other=0.0)
    tl.atomic_add(y_ptr + rows, vals * xs, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_non_complex_nnzpar_kernel(
    data_ri_ptr,
    indices_ptr,
    col_ids_ptr,
    x_ri_ptr,
    y_ri_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    """Complex counterpart of :func:`_spmv_csc_non_real_nnzpar_kernel`."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < nnz
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    cols = tl.load(col_ids_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    x_re = tl.load(x_ri_ptr + cols * 2, mask=mask, other=0.0)
    x_im = tl.load(x_ri_ptr + cols * 2 + 1, mask=mask, other=0.0)
    tl.atomic_add(y_ri_ptr + rows * 2, a_re * x_re - a_im * x_im, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + rows * 2 + 1, a_re * x_im + a_im * x_re, mask=mask, sem="relaxed")


SPMV_CSC_NNZPAR_BLOCK = 256


def _triton_spmv_csc_kernel(prepared, x, op_code):
    dtype = prepared.data.dtype
    trans = _spmv_csc_op_transposes(op_code)
    out_len = prepared.n_cols if trans else prepared.n_rows
    if trans and prepared.nnz != 0:
        csr_delegate = getattr(prepared, "csr_delegate", None)
        if csr_delegate is not None:
            from .spmv_csr import flagsparse_spmv_csr

            # Let spmv_csr allocate: it fills the whole vector, so zeroing first is a
            # wasted memset -- that alone moved the operator 0.765 -> 0.940.
            # use_opt=False deliberately: the bucketed path measured 0.39x on fp32,
            # while the default matches the oracle over both (4.39x vs 4.41x).
            return flagsparse_spmv_csr(prepared=csr_delegate, x=x, use_opt=False)
    y = torch.zeros(out_len, dtype=dtype, device=prepared.data.device)
    if prepared.nnz == 0:
        return y
    if not trans:
        col_ids = getattr(prepared, "col_ids", None)
        if col_ids is not None:
            # nnz-parallel path: grid scales with nnz instead of n_cols x max_segments.
            nnz = int(prepared.nnz)
            grid_nnz = (triton.cdiv(nnz, SPMV_CSC_NNZPAR_BLOCK),)
            if _is_complex_dtype(dtype):
                data_ri = torch.view_as_real(prepared.data).reshape(-1)
                x_ri = torch.view_as_real(x).reshape(-1)
                y_ri = torch.zeros(out_len * 2, dtype=data_ri.dtype, device=y.device)
                _spmv_csc_non_complex_nnzpar_kernel[grid_nnz](
                    data_ri,
                    prepared.kernel_indices,
                    col_ids,
                    x_ri,
                    y_ri,
                    nnz,
                    BLOCK=SPMV_CSC_NNZPAR_BLOCK,
                )
                y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
                return y
            _spmv_csc_non_real_nnzpar_kernel[grid_nnz](
                prepared.data,
                prepared.kernel_indices,
                col_ids,
                x,
                y,
                nnz,
                BLOCK=SPMV_CSC_NNZPAR_BLOCK,
            )
            return y
        grid = (prepared.n_cols, prepared.max_segments)
        if _is_complex_dtype(dtype):
            data_ri = torch.view_as_real(prepared.data).reshape(-1)
            x_ri = torch.view_as_real(x).reshape(-1)
            y_ri = torch.zeros(out_len * 2, dtype=data_ri.dtype, device=y.device)
            _spmv_csc_non_complex_kernel[grid](
                data_ri,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x_ri,
                y_ri,
            # y = op(A) @ x here; alpha/beta exist for the C API's
            # cuSPARSE-compatible signature and fold away at 1 / 0.
                1,
                0,
                prepared.n_cols,
                BLOCK_NNZ=prepared.block_nnz,
            )
            y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
            return y
        _spmv_csc_non_real_kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x,
            y,
            1,                       # alpha; beta is a prologue on this route
            prepared.n_cols,
            BLOCK_NNZ=prepared.block_nnz,
        )
        return y
    grid = (prepared.n_cols,)
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.empty(out_len * 2, dtype=data_ri.dtype, device=y.device)
        _spmv_csc_trans_complex_kernel[grid](
            data_ri,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x_ri,
            y_ri,
            1,
            0,
            0,
            0,
            prepared.n_cols,
            BLOCK_NNZ=prepared.block_nnz,
            MAX_SEGMENTS=prepared.max_segments,
            CONJ=(op_code == SPMV_CSC_OP_CONJ_TRANS),
            HAS_BETA=False,
        )
        y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
        return y
    _spmv_csc_trans_real_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        x,
        y,
        1,
        0,
        prepared.n_cols,
        BLOCK_NNZ=prepared.block_nnz,
        MAX_SEGMENTS=prepared.max_segments,
        HAS_BETA=False,
    )
    return y


def _spmv_csc_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmv_csc_int32_fallback_blocker(prepared):
    if prepared.nnz > _INDEX_LIMIT_INT32:
        return f"nnz {prepared.nnz} cannot fit int32"
    if prepared.kernel_indices.numel() > 0:
        max_row = int(prepared.kernel_indices.max().item())
        if max_row > _INDEX_LIMIT_INT32:
            return f"row index {max_row} cannot fit int32"
    if prepared.kernel_indptr.numel() > 0:
        max_ptr = int(prepared.kernel_indptr[-1].item())
        if max_ptr > _INDEX_LIMIT_INT32:
            return f"indptr offset {max_ptr} cannot fit int32"
    return None


def _spmv_csc_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_csc_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(f"int32 fallback is unsafe: {blocker}") from reason
    return PreparedCscSpmv(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices.to(torch.int32).contiguous(),
        kernel_indptr=prepared.kernel_indptr.to(torch.int32).contiguous(),
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        max_col_nnz=prepared.max_col_nnz,
        col_lengths=prepared.col_lengths,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
        launch_backend=prepared.launch_backend,
        device_warp_size=prepared.device_warp_size,
    )


def _run_spmv_csc_prepared_with_fallback(prepared, x, op_code):
    try:
        return _triton_spmv_csc_kernel(prepared, x, op_code)
    except RuntimeError as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmv_csc_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmv_csc_prepared_with_int32_indices(prepared, exc)
        return _triton_spmv_csc_kernel(fallback_prepared, x, op_code)


def flagsparse_spmv_csc(
    data=None,
    indices=None,
    indptr=None,
    x=None,
    shape=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    return_meta=False,
    prepared=None,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
    alg=None, config=None, timing=False,
):
    """CSC SpMV using native Triton CSC kernels."""
    if alg is None and prepared is None and data is not None and data.dtype == torch.float16:
        alg = "spmv_csc_base"
    if alg is not None or isinstance(prepared, PreparedCscRoute):
        if prepared is None:
            prepared = _prepare_csc_registered(data, indices, indptr, shape, op, transpose, alg, config, index_fallback_policy, "spmv",
                                       {"block_nnz": block_nnz, "max_segments": max_segments})
        if transpose is not None and bool(transpose) != _spmv_csc_op_transposes(prepared.op):
            raise ValueError("transpose conflicts with prepared op")
        return flagsparse_spmv_csc_run(prepared, x, alg=alg, config=config, op=op, out=out,
                                       return_time=return_time, return_meta=return_meta, timing=timing)
    if config is not None:
        raise ValueError("config requires an explicit algorithm")
    op_explicit = op is not None
    op_code = _normalize_spmv_csc_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_csc_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmv_csc(
            data,
            indices,
            indptr,
            shape,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=op_code,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if op_explicit and op_code != prepared.op:
            raise ValueError(
                f"op={_spmv_csc_op_to_name(op_code)} does not match prepared.op={_spmv_csc_op_to_name(prepared.op)}"
            )
        if (
            not op_explicit
            and transpose is not None
            and bool(transpose) != prepared.transpose
        ):
            raise ValueError(
                f"transpose={bool(transpose)} does not match prepared.transpose={prepared.transpose}"
            )
        if not op_explicit:
            op_code = prepared.op
    x = _validate_spmv_csc_x(x, prepared, op_code)
    do_timing = bool(return_time or return_meta)
    if do_timing:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    y = _run_spmv_csc_prepared_with_fallback(prepared, x, op_code)
    if do_timing:
        _ACCEL.synchronize()
        compute_ms = (time.perf_counter() - t0) * 1000.0
        op_total_ms = compute_ms
    else:
        compute_ms = None
        op_total_ms = None
    if out is not None:
        if not _is_accel_tensor(out):
            raise ValueError("out must be a CUDA tensor")
        if out.device != y.device:
            raise ValueError("out must be on the same CUDA device as the result")
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_meta:
        meta = {
            "op": _spmv_csc_op_to_name(op_code),
            "execution_path": "csr_delegate" if prepared.csr_delegate is not None else "native_csc",
            "symbolic_ms": 0.0 if do_timing else None,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "block_nnz": prepared.block_nnz,
            "max_segments": prepared.max_segments,
            "launch_backend": prepared.launch_backend,
            "device_warp_size": prepared.device_warp_size,
            "index_fallback_applied": prepared.index_fallback_applied,
            "index_fallback_reason": prepared.index_fallback_reason,
        }
        if return_time:
            return y, op_total_ms, meta
        return y, meta
    if return_time:
        return y, op_total_ms
    return y


@triton.jit
def _csc_product(A, B, pos, dense_pos, mask, COMPLEX: tl.constexpr, FP64: tl.constexpr, CONJ: tl.constexpr):
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    if COMPLEX:
        ar = tl.load(A + 2 * pos, mask, 0).to(dtype)
        ai = tl.load(A + 2 * pos + 1, mask, 0).to(dtype)
        br = tl.load(B + 2 * dense_pos, mask, 0).to(dtype)
        bi = tl.load(B + 2 * dense_pos + 1, mask, 0).to(dtype)
        if CONJ:
            ai = -ai
        return ar * br - ai * bi, ar * bi + ai * br
    return tl.load(A + pos, mask, 0).to(dtype) * tl.load(B + dense_pos, mask, 0).to(dtype), tl.full(pos.shape, 0, dtype)


@triton.jit
def _csc_columns_reduce_kernel(A, I, P, B, Y, COLS, N, BS0, BS1, YS0, YS1,
                               R: tl.constexpr, V: tl.constexpr, SUBGROUP: tl.constexpr,
                               ACCS: tl.constexpr, COMPLEX: tl.constexpr, FP64: tl.constexpr,
                               CONJ: tl.constexpr):
    cols = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    lane = tl.arange(0, V).to(tl.int64)
    start = tl.load(P + cols, cols < COLS, 0).to(tl.int64)
    end = tl.load(P + cols + 1, cols < COLS, 0).to(tl.int64)
    longest = tl.max(end - start, 0)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    re = tl.full((R, V), 0, dtype)
    im = tl.full((R, V), 0, dtype)
    re1 = tl.full((R, V), 0, dtype)
    im1 = tl.full((R, V), 0, dtype)
    if SUBGROUP:
        for offset in range(tl.cdiv(longest, V)):
            pos = start[:, None] + offset * V + lane[None, :]
            valid = (cols[:, None] < COLS) & (pos < end[:, None])
            rows = tl.load(I + pos, valid, 0).to(tl.int64)
            vr, vi = _csc_product(A, B, pos, rows * BS0, valid, COMPLEX, FP64, CONJ)
            re += vr
            im += vi
        rr = tl.sum(re, 1)
        ii = tl.sum(im, 1)
        dest = cols * YS0
        if COMPLEX:
            tl.store(Y + 2 * dest, rr, cols < COLS)
            tl.store(Y + 2 * dest + 1, ii, cols < COLS)
        else:
            tl.store(Y + dest, rr, cols < COLS)
    else:
        ns = tl.program_id(1).to(tl.int64) * V + lane
        for offset in range(tl.cdiv(longest, ACCS)):
            for a in tl.static_range(ACCS):
                pos = start[:, None] + offset * ACCS + a + tl.zeros((1, V), tl.int64)
                valid = (cols[:, None] < COLS) & (pos < end[:, None]) & (ns[None, :] < N)
                rows = tl.load(I + pos, valid, 0).to(tl.int64)
                vr, vi = _csc_product(A, B, pos, rows * BS0 + ns[None, :] * BS1, valid, COMPLEX, FP64, CONJ)
                if a == 0:
                    re += vr
                    im += vi
                else:
                    re1 += vr
                    im1 += vi
        dest = cols[:, None] * YS0 + ns[None, :] * YS1
        valid = (cols[:, None] < COLS) & (ns[None, :] < N)
        if COMPLEX:
            tl.store(Y + 2 * dest, re + re1, valid)
            tl.store(Y + 2 * dest + 1, im + im1, valid)
        else:
            tl.store(Y + dest, re + re1, valid)


@triton.jit
def _csc_columns_atomic_kernel(A, I, P, B, Y, COLS, N, BS0, BS1, YS0, YS1,
                               R: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
                               COMPLEX: tl.constexpr, FP64: tl.constexpr):
    cols = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    ks = tl.arange(0, K).to(tl.int64)
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
    start = tl.load(P + cols, cols < COLS, 0).to(tl.int64)
    end = tl.load(P + cols + 1, cols < COLS, 0).to(tl.int64)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    dense_pos = cols[:, None] * BS0 + ns[None, :] * BS1
    dense_mask = (cols[:, None] < COLS) & (ns[None, :] < N)
    if COMPLEX:
        br = tl.load(B + 2 * dense_pos, dense_mask, 0).to(dtype)
        bi = tl.load(B + 2 * dense_pos + 1, dense_mask, 0).to(dtype)
    else:
        br = tl.load(B + dense_pos, dense_mask, 0).to(dtype)
    for offset in range(tl.cdiv(tl.max(end - start, 0), K)):
        pos = start[:, None] + offset * K + ks[None, :]
        sparse_mask = (cols[:, None] < COLS) & (pos < end[:, None])
        rows = tl.load(I + pos, sparse_mask, 0).to(tl.int64)
        if COMPLEX:
            ar = tl.load(A + 2 * pos, sparse_mask, 0).to(dtype)
            ai = tl.load(A + 2 * pos + 1, sparse_mask, 0).to(dtype)
            re = ar[:, :, None] * br[:, None, :] - ai[:, :, None] * bi[:, None, :]
            im = ar[:, :, None] * bi[:, None, :] + ai[:, :, None] * br[:, None, :]
        else:
            ar = tl.load(A + pos, sparse_mask, 0).to(dtype)
            re = ar[:, :, None] * br[:, None, :]
        dest = rows[:, :, None] * YS0 + ns[None, None, :] * YS1
        valid = sparse_mask[:, :, None] & (ns[None, None, :] < N)
        if COMPLEX:
            tl.atomic_add(Y + 2 * dest, re, valid, sem="relaxed")
            tl.atomic_add(Y + 2 * dest + 1, im, valid, sem="relaxed")
        else:
            tl.atomic_add(Y + dest, re, valid, sem="relaxed")


def _launch_csc_columns(data, indices, indptr, dense, shape, op, alg, cfg):
    """Shared native CSC launch; all strides are in logical (possibly complex) elements."""
    vector = dense.ndim == 1
    n = 1 if vector else dense.shape[1]
    rows = shape[0] if op == "non" else shape[1]
    output_shape = (rows,) if vector else (rows, n)
    result = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if op == "non":
        result.zero_()
    if not rows or not n or not shape[1]:
        return result
    complex_values = data.is_complex()
    a = torch.view_as_real(data) if complex_values else data
    b = torch.view_as_real(dense) if complex_values else dense
    y = torch.view_as_real(result) if complex_values else result
    common = dict(R=cfg["columns_per_program"], COMPLEX=complex_values,
                  FP64=data.dtype in (torch.float64, torch.complex128),
                  num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    args = (a, indices, indptr, b, y, shape[1], n, dense.stride(0),
            0 if vector else dense.stride(1), result.stride(0), 0 if vector else result.stride(1))
    width = cfg.get("block_n", 1)
    grid = (triton.cdiv(shape[1], cfg["columns_per_program"]), triton.cdiv(n, width))
    if op == "non":
        _csc_columns_atomic_kernel[grid](*args, K=cfg["block_nnz"], BN=width, **common)
    else:
        _csc_columns_reduce_kernel[grid](*args, V=cfg.get("lanes_per_column", width),
                                       SUBGROUP=vector, ACCS=cfg.get("panel_accumulators", 1),
                                       CONJ=op == "conj", **common)
    return result


class PreparedCscRoute:
    """Validated original CSC inputs; registered runs never retain execution plans."""
    def __init__(self, data, indices, indptr, shape, op, alg, config, policy, kind):
        self.data, self.kernel_indices, self.kernel_indptr = data, indices, indptr
        self.shape = tuple(map(int, shape))
        self.n_rows, self.n_cols = self.shape
        self.nnz = data.numel()
        self.op, self.alg, self.config = op, alg, dict(config or {})
        self.index_fallback_policy, self.kind = policy, kind
        self.int32_safe = self.nnz <= _INDEX_LIMIT_INT32 and (not indices.numel() or int(indices.max().item()) <= _INDEX_LIMIT_INT32)


def _csc_spec(alg, kind):
    from ._spmm_csr_config import CSC_ALGORITHMS, BACKENDS
    base = "spmv_csc_base" if kind == "spmv" else "spmm_csc_base"
    names = ("csc_col_subgroup", "csc_col_tile_atomic") if kind == "spmv" else ("csc_col_panel", "csc_col_tile_panel_atomic")
    if alg not in (base, *names):
        raise ValueError(f"unknown {kind} CSC algorithm {alg!r}")
    return dict(name=alg, ops=CSC_ALGORITHMS.get(alg, ("non", "trans", "conj")),
                value_dtypes=SUPPORTED_SPMV_CSC_VALUE_DTYPES, backends=BACKENDS,
                layouts=("row", "col", "strided"), implementation_version=1,
                compute_dtype="native_component", validation="unverified")


def _prepare_csc_registered(data, indices, indptr, shape, op, transpose, alg, config, policy, kind, legacy_options=None):
    code = _normalize_spmv_csc_op(op, transpose=bool(transpose))
    if transpose is not None and op is not None and bool(transpose) != _spmv_csc_op_transposes(code):
        raise ValueError("transpose conflicts with op")
    base = "spmv_csc_base" if kind == "spmv" else "spmm_csc_base"
    selected = base if alg in (None, "auto", "base", "csc_base") else alg
    spec = _csc_spec(selected, kind)
    name = _spmv_csc_op_to_name(code)
    if name not in spec["ops"]:
        raise ValueError(f"{selected} does not support op={name}")
    _prepare_spmv_csc_matrix(data, indices, indptr, shape)
    options = dict(legacy_options or {})
    if selected != base and any(value is not None and not (kind == "spmv" and key == "block_nnz" and value == 256) for key, value in options.items()):
        raise ValueError("legacy block parameters conflict with new algorithm; use config")
    route = PreparedCscRoute(data, indices, indptr, shape, name, selected, config,
                             _normalize_spmv_csc_index_fallback_policy(policy), kind)
    route.legacy_options = options if selected == base else {}
    return route


def get_spmv_csc_algorithm_spec(alg):
    return _csc_spec(alg, "spmv")


def list_spmv_csc_algorithms(op=None, dtype=None, backend=None):
    names = ("spmv_csc_base", "csc_col_subgroup", "csc_col_tile_atomic")
    return tuple(name for name in names if
                 (op is None or _spmv_csc_op_to_name(op) in _csc_spec(name, "spmv")["ops"]) and
                 (dtype is None or dtype in SUPPORTED_SPMV_CSC_VALUE_DTYPES) and
                 (backend is None or backend in _csc_spec(name, "spmv")["backends"]))


def _run_csc_registered(prepared, dense, *, alg=None, config=None, op=None, out=None,
                        return_time=False, return_meta=False, timing=False):
    from ._spmm_csr_runtime import backend_caps, Phases
    from ._spmm_csr_config import resolve_csc_config, CSC_ALGORITHMS
    from ._spmv_csr_config import is_index_compatibility_error
    if op is not None and _spmv_csc_op_to_name(op) != prepared.op:
        raise ValueError("op conflicts with prepared CSC")
    selected = prepared.alg if alg in (None, "auto") else alg
    spec = _csc_spec(selected, prepared.kind)
    if prepared.op not in spec["ops"]:
        raise ValueError(f"{selected} does not support {prepared.op}")
    cfg_input = config if config is not None else (prepared.config if selected == prepared.alg else {})
    data, indices, ptr = prepared.data, prepared.kernel_indices, prepared.kernel_indptr
    vector = prepared.kind == "spmv"
    expected = prepared.n_cols if prepared.op == "non" else prepared.n_rows
    if not torch.is_tensor(dense) or dense.ndim != (1 if vector else 2):
        raise ValueError("dense operand rank does not match operation")
    if dense.shape[0] != expected or dense.dtype != data.dtype or dense.device != data.device:
        raise ValueError("dense operand shape/dtype/device mismatch")
    out_rows = prepared.n_rows if prepared.op == "non" else prepared.n_cols
    out_shape = (out_rows,) if vector else (out_rows, dense.shape[1])
    if out is not None:
        if out.shape != out_shape or out.dtype != data.dtype or out.device != data.device:
            raise ValueError("out shape/dtype/device mismatch")
        if any(torch._C._overlaps(out, value) for value in (data, indices, ptr, dense)):
            raise ValueError("out overlaps an input")
    cfg, info = {}, dict(backend=_backend_name())
    if selected in CSC_ALGORITHMS:
        cfg, info = resolve_csc_config(selected, str(data.dtype).removeprefix("torch."),
                                       1 if vector else dense.shape[1], backend_caps(data.device), cfg_input, op=prepared.op)
    elif cfg_input:
        raise ValueError("legacy CSC route does not accept config")
    # Range checks precede measurement. Conversion is performed inside each run.
    safe32 = prepared.int32_safe
    def execute(phases):
        fallback_reason = None
        with phases.measure("process_gpu_ms"):
            a = data.resolve_conj().contiguous()
            i, p = indices.contiguous(), ptr.contiguous()
            b = dense.resolve_conj()
            if data.dtype == torch.float16:
                a, b = a.float(), b.float()
            execution = None
            if selected not in CSC_ALGORITHMS:
                if prepared.n_cols > _INDEX_LIMIT_INT32 or prepared.nnz > _INDEX_LIMIT_INT32:
                    raise ValueError("legacy CSC owner-map exceeds supported int32 address range")
                lengths = p[1:] - p[:-1]
                maximum = int(lengths.max().item()) if prepared.n_cols else 0
                validated = (a, i, p, prepared.n_rows, prepared.n_cols, lengths, maximum)
                if vector:
                    execution = prepare_spmv_csc(a, i, p, prepared.shape, op=prepared.op,
                                                 _validated=validated, _allow_delegate=False, **(prepared.legacy_options if selected == prepared.alg else {}))
                else:
                    from .spmm_csc import prepare_spmm_csc_route
                    execution = prepare_spmm_csc_route(a, i, p, prepared.shape, op=prepared.op,
                                                       _validated=validated, **(prepared.legacy_options if selected == prepared.alg else {}))
        def compute(ii, pp):
            if selected in CSC_ALGORITHMS:
                return _launch_csc_columns(a, ii, pp, b, prepared.shape, prepared.op, selected, cfg)
            execution.kernel_indices, execution.kernel_indptr = ii, pp
            if vector:
                return _triton_spmv_csc_kernel(execution, b.contiguous(), _normalize_spmv_csc_op(prepared.op))
            from .spmm_csc import _triton_spmm_csc_base_kernel
            return _triton_spmm_csc_base_kernel(execution, b)
        with phases.measure("compute_ms"):
            try:
                result = compute(i, p)
            except (RuntimeError, TypeError) as exc:
                if prepared.index_fallback_policy != "auto" or not is_index_compatibility_error(exc) or not (i.dtype == torch.int64 or p.dtype == torch.int64):
                    raise
                if not safe32:
                    raise RuntimeError("int32 fallback is unsafe") from exc
                i, p = i.to(torch.int32), p.to(torch.int32)
                fallback_reason = str(exc)
                result = compute(i, p)
            result = result.to(data.dtype)
            if out is not None:
                out.copy_(result)
                result = out
        meta = dict(info, alg=selected, alg_requested=alg or prepared.alg, alg_resolved=selected,
                    config=cfg, op=prepared.op, implementation_version=1, timing_contract_version=2,
                    compute_dtype="float64" if data.dtype in (torch.float64, torch.complex128) else "float32",
                    input_indices_dtype=str(indices.dtype), input_indptr_dtype=str(ptr.dtype),
                    execution_indices_dtype=str(i.dtype), execution_indptr_dtype=str(p.dtype),
                    index_fallback_applied=fallback_reason is not None, index_fallback_reason=fallback_reason,
                    transpose_strategy="native_column_reduce" if prepared.op != "non" else "native_scatter",
                    native_format="csc", execution_path="native_csc", process_cpu_ms=0.0,
                    component_dtype="float64" if data.dtype in (torch.float64, torch.complex128) else "float32",
                    dense_stride=tuple(dense.stride()), output_stride=tuple(result.stride()))
        return result, meta
    measured = return_time or return_meta or timing
    if measured:
        start, end = _ACCEL.Event(enable_timing=True), _ACCEL.Event(enable_timing=True)
        start.record()
    result, meta = execute(Phases(False))
    if measured:
        end.record()
        end.synchronize()
        meta["gpu_ms"] = start.elapsed_time(end)
        meta["operator_ms"] = meta["op_total_ms"] = meta["gpu_ms"]
    if timing:
        phases = Phases(True)
        execute(phases)
        meta.update(phases.results())
    if return_time and return_meta:
        return result, meta["operator_ms"], meta
    if return_time:
        return result, meta["operator_ms"]
    return (result, meta) if return_meta else result


def flagsparse_spmv_csc_run(prepared, x, *, alg=None, config=None, op=None, out=None,
                            return_time=False, return_meta=False, timing=False):
    if isinstance(prepared, PreparedCscRoute) and prepared.kind != "spmv":
        raise TypeError("prepared must describe CSC SpMV")
    if not isinstance(prepared, PreparedCscRoute):
        prepared = _prepare_csc_registered(prepared.data, prepared.kernel_indices, prepared.kernel_indptr,
                                           prepared.shape, prepared.op, None, alg or "auto", config,
                                           prepared.index_fallback_policy, "spmv",
                                           {"block_nnz": prepared.block_nnz, "max_segments": prepared.max_segments}
                                           if alg in (None, "auto", "spmv_csc_base") else None)
    return _run_csc_registered(prepared, x, alg=alg, config=config, op=op, out=out,
                               return_time=return_time, return_meta=return_meta, timing=timing)
