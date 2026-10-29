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
from . import mixed_spmx as _mixed_spmx
from ._rocm_sparse_cache import TensorPlanCache

import triton
import triton.language as tl

# Resolved once at import: the runtime cannot change under a live process, and
# prepare_spmv_csc sits on the per-call path.
_SPMV_CSC_CUDA = _backend_name() == "cuda"

SUPPORTED_SPMV_CSC_VALUE_DTYPES = (
    torch.float16,
    torch.bfloat16,
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
    return "CSC SpMV supports float32, float64, complex64, and complex128"


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
        "original_inputs",
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
        "non_gather_plan",
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
        self.non_gather_plan = None
        if (_normalize_spmv_csc_op(op, transpose=transpose) == SPMV_CSC_OP_NON
                and ((_SPMV_CSC_CUDA and data.dtype == torch.complex64)
                     or (_is_rocm_runtime() and not _is_ascend_runtime()
                         and data.dtype in (torch.complex64, torch.float32, torch.float16)))):
            from . import _spmv_csr_transpose
            # CSC(A) is CSR(A.T); transposing this topology gives row owners of A.
            self.non_gather_plan = _spmv_csr_transpose.prepare(
                kernel_indices, kernel_indptr, (self.n_cols, self.n_rows))
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


def _prepare_spmv_csc_matrix(data, indices, indptr, shape, *, value_dtypes=None):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
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
    if data.dtype not in (SUPPORTED_SPMV_CSC_VALUE_DTYPES if value_dtypes is None else value_dtypes):
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
    transpose=False,
    op=None,
    index_fallback_policy="auto",
    *, alg=None, config=None,
):
    from .mixed_spmx import remember_registered_inputs
    original_inputs = (data, indices, indptr)
    if alg not in (None, "auto", "base", "spmv_csc_base"):
        return remember_registered_inputs(_prepare_csc_registered(data, indices, indptr, shape, op,
            True if transpose else None, alg, config, index_fallback_policy, "spmv",
            {"block_nnz": block_nnz, "max_segments": max_segments}), original_inputs)
    if config is not None:
        raise ValueError("config requires an explicit CSC extension algorithm")
    index_fallback_policy = _normalize_spmv_csc_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_csc_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_CSC_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz = (
        _prepare_spmv_csc_matrix(data, indices, indptr, shape)
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
            if _is_ascend_runtime():
                # A CSC indptr runs over columns, so its "row ids" are column ids.
                col_ids = _csr_row_ids_torch(indptr, n_cols)
            else:
                # Deferred import: sddmm_csr only depends on _common, so there is no cycle,
                # but keeping it local avoids adding a load-time edge between operators.
                from .sddmm_csr import _build_row_ids

                col_ids = _build_row_ids(indptr.to(torch.int32), int(data.numel()))
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
        _SPMV_CSC_CUDA
        and data.dtype not in (torch.float16, torch.bfloat16)
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
    return remember_registered_inputs(PreparedCscSpmv(
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
    ), original_inputs)


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


def _ascend_spmv_csc_torch(prepared, x, op_code):
    """CSC SpMV fallback using torch_npu indexing primitives.

    CANN currently cannot lower the native CSC Triton kernels.  Keep this route
    local to Ascend; all other backends continue to use the kernels above.
    """
    n_rows, n_cols = prepared.n_rows, prepared.n_cols
    nnz = int(prepared.nnz)
    if nnz == 0:
        out_len = n_cols if _spmv_csc_op_transposes(op_code) else n_rows
        return torch.zeros(out_len, dtype=prepared.data.dtype, device=prepared.data.device)
    col_ids = getattr(prepared, "col_ids", None)
    if col_ids is None:
        col_ids = _csr_row_ids_torch(prepared.kernel_indptr, n_cols)
    rows = prepared.kernel_indices.to(torch.int64)
    cols = col_ids.to(torch.int64)
    values = prepared.data
    if op_code == SPMV_CSC_OP_CONJ_TRANS:
        values = values.conj()
    if _spmv_csc_op_transposes(op_code):
        result = torch.zeros(n_cols, dtype=values.dtype, device=values.device)
        _index_add_values(result, 0, cols, values * _gather_values(x, rows))
    else:
        result = torch.zeros(n_rows, dtype=values.dtype, device=values.device)
        _index_add_values(result, 0, rows, values * _gather_values(x, cols))
    return result


def _triton_spmv_csc_kernel(prepared, x, op_code):
    dtype = prepared.data.dtype
    trans = _spmv_csc_op_transposes(op_code)
    out_len = prepared.n_cols if trans else prepared.n_rows
    if not trans and prepared.non_gather_plan is not None:
        from . import _spmv_csr_transpose
        return _spmv_csr_transpose.gather(prepared.data, x, prepared.non_gather_plan, out_len)
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
    if _is_ascend_runtime():
        return _ascend_spmv_csc_torch(prepared, x, op_code)
    try:
        return _triton_spmv_csc_kernel(prepared, x, op_code)
    except RuntimeError as exc:
        from ._spmv_csr_config import is_index_compatibility_error

        if (
            prepared.index_fallback_policy != "auto"
            or not _spmv_csc_uses_int64_indices(prepared)
            or not is_index_compatibility_error(exc)
        ):
            raise
        fallback_prepared = _spmv_csc_prepared_with_int32_indices(prepared, exc)
        return _triton_spmv_csc_kernel(fallback_prepared, x, op_code)


_SPMV_CSC_VIA_MIXED_DTYPES = (torch.float16, torch.bfloat16, torch.int8)
_ROCM_CSC_GATHER_CACHE = TensorPlanCache()


def _spmv_csc_via_mixed(data, indices, indptr, x, shape, op_code, out, out_dtype,
                        return_time, return_meta):
    """CSC(A) arrays read two ways, both accumulating in float32 / int32.

    op='non': entry p of column c is (row=indices[p], col=c) -- a COO product.
    op='trans'/'conj': CSC(A) is CSR(A.T), so it is a plain CSR product
    (conj is the identity on these real dtypes).
    """
    if _is_complex_dtype(data.dtype):
        raise TypeError("mixed CSC SpMV takes real values")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_cols + 1 or data.numel() != indices.numel():
        raise ValueError("invalid CSC dimensions")
    timed = bool(return_time or return_meta)
    if (_is_rocm_runtime() and not _is_ascend_runtime()
            and data.dtype == torch.float16 and not _spmv_csc_op_transposes(op_code)):
        # CSC(A) is CSR(A.T).  A prepared gather plan computes A @ x by
        # grouping entries by output row, avoiding the per-nnz FP32 atomic
        # path used by the generic mixed COO fallback.
        from . import _spmv_csr_transpose
        _mixed_spmx._check_1d((data, "data"), (x, "x"))
        if x.numel() != n_cols:
            raise ValueError("x shape does not match CSC operation")
        if x.device != data.device:
            raise ValueError("CSC inputs must be on the same device")
        result_dtype, _ = _mixed_spmx._resolve_types(data.dtype, x.dtype, out, out_dtype, False)
        if out is not None and (out.shape != (n_rows,) or out.device != data.device):
            raise ValueError("out shape/device must match the CSC SpMV result")
        t0 = _mixed_spmx._start(timed)
        sources, options = (indices, indptr), (n_rows, n_cols, data.device)
        plan = _ROCM_CSC_GATHER_CACHE.get(sources, options)
        if plan is None:
            _mixed_spmx._check_1d((indices, "indices"), (indptr, "indptr"))
            if indices.device != data.device or indptr.device != data.device:
                raise ValueError("CSC inputs must be on the same device")
            plan = _spmv_csr_transpose.prepare(indices, indptr, (n_cols, n_rows))
            _ROCM_CSC_GATHER_CACHE.put(sources, options, plan)
        target = (None if result_dtype == data.dtype
                  else torch.empty(n_rows, dtype=result_dtype, device=data.device))
        y = _spmv_csr_transpose.gather(data.contiguous(), x, plan, n_rows, out=target)
        y = _mixed_spmx._finish(y, out, t0, timed)
        if not timed:
            return y
        y, elapsed = y
        if return_meta:
            meta = {"route": "rocm_csc_gather", "compute_ms": elapsed, "op_total_ms": elapsed}
            return (y, elapsed, meta) if return_time else (y, meta)
        return y, elapsed
    if _spmv_csc_op_transposes(op_code):
        y = _mixed_spmx.spmv_csr_mixed(
            data, indices, indptr, x, (n_cols, n_rows),
            out=out, out_dtype=out_dtype, return_time=timed,
        )
    elif (_SPMV_CSC_CUDA or _is_maca_runtime()) and data.dtype == torch.float16:
        y = _mixed_spmx.spmv_csc_half(
            data, indices, indptr, x, shape, out=out, out_dtype=out_dtype, return_time=timed)
    else:
        lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
        col_ids = torch.repeat_interleave(
            torch.arange(n_cols, device=indptr.device, dtype=torch.int64),
            lengths,
            output_size=int(data.numel()),
        )
        y = _mixed_spmx.spmv_coo_mixed(
            data, indices, col_ids, x, (n_rows, n_cols),
            out=out, out_dtype=out_dtype, return_time=timed,
        )
    if not timed:
        return y
    y, elapsed = y
    if return_meta:
        meta = {"route": "mixed", "compute_ms": elapsed, "op_total_ms": elapsed}
        return (y, elapsed, meta) if return_time else (y, meta)
    return y, elapsed


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
    *,
    out_dtype=None,
    alg=None,
    config=None,
    timing=False,
):
    """CSC SpMV using native Triton CSC kernels.

    float16 / bfloat16 / int8 values (and widened ``out_dtype``) avoid the native
    kernels, whose op='non' path atomically adds in the value dtype -- fp16 atomics
    are not available on every backend. See ``_spmv_csc_via_mixed``.
    """
    from .mixed_spmx import validate_registered_output_alias
    validate_registered_output_alias(prepared, x, out, (data, indices, indptr))
    input_dtype = prepared.data.dtype if prepared is not None else getattr(data, "dtype", None)
    if input_dtype in (torch.float16, torch.bfloat16) and (
        prepared is not None and not isinstance(prepared, PreparedCscRoute)
        or alg in ("base", "spmv_csc_base")
    ):
        if op is not None and transpose is not None and bool(transpose) != _spmv_csc_op_transposes(_normalize_spmv_csc_op(op)):
            raise ValueError("transpose conflicts with op")
        requested_op = op if op is not None else ("trans" if transpose else "non") if transpose is not None else None
        if prepared is None:
            prepared = prepare_spmv_csc(data, indices, indptr, shape,
                block_nnz=block_nnz, max_segments=max_segments, op=requested_op,
                index_fallback_policy=index_fallback_policy)
        elif shape is not None and tuple(shape) != prepared.shape:
            raise ValueError("shape conflicts with prepared CSC")
        return flagsparse_spmv_csc_run(prepared, x, alg="spmv_csc_base", config=config,
            op=requested_op, out=out, out_dtype=out_dtype, return_time=return_time,
            return_meta=return_meta, timing=timing)
    if alg not in (None, "auto", "base", "spmv_csc_base") or isinstance(prepared, PreparedCscRoute):
        if prepared is None:
            prepared = _prepare_csc_registered(data, indices, indptr, shape, op, transpose,
                alg, config, index_fallback_policy, "spmv",
                {"block_nnz": block_nnz, "max_segments": max_segments})
        elif shape is not None and tuple(shape) != prepared.shape:
            raise ValueError("shape conflicts with prepared CSC")
        if op is not None and transpose is not None and bool(transpose) != _spmv_csc_op_transposes(_normalize_spmv_csc_op(op)):
            raise ValueError("transpose conflicts with op")
        requested_op = op if op is not None else ("trans" if transpose else "non") if transpose is not None else None
        return flagsparse_spmv_csc_run(prepared, x, alg=alg, config=config, op=requested_op,
            out=out, out_dtype=out_dtype, return_time=return_time, return_meta=return_meta, timing=timing)
    if config is not None:
        raise ValueError("config requires an explicit CSC extension algorithm")
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
    if (
        prepared is None
        and torch.is_tensor(data)
        and (
            data.dtype in _SPMV_CSC_VIA_MIXED_DTYPES
            or _mixed_spmx.spmv_needs_mixed(
                data.dtype, x.dtype if torch.is_tensor(x) else None, out, out_dtype
            )
        )
    ):
        return _spmv_csc_via_mixed(
            data, indices, indptr, x, shape, op_code, out, out_dtype, return_time, return_meta
        )
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
    else:
        product = tl.load(A + pos, mask, 0).to(dtype) * tl.load(B + dense_pos, mask, 0).to(dtype)
        return product, tl.full(product.shape, 0, dtype)


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


@triton.jit
def _csc_vector_kernel(A, I, P, B, Y, Owners, Starts, Ends, COUNT, N,
                       BS0, BS1, YS0, YS1, N_OFFSET,
                       BK: tl.constexpr, BN: tl.constexpr,
                       SEGMENTS: tl.constexpr, COMPLEX: tl.constexpr,
                       FP64: tl.constexpr, CONJ: tl.constexpr, DIRECT: tl.constexpr = False):
    item = tl.program_id(0).to(tl.int64)
    col = tl.load(Owners + item) if SEGMENTS else item
    start = tl.load(Starts + item).to(tl.int64) if SEGMENTS else tl.load(P + col).to(tl.int64)
    end = tl.load(Ends + item).to(tl.int64) if SEGMENTS else tl.load(P + col + 1).to(tl.int64)
    k = tl.arange(0, BK)
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    acc_type = tl.float64 if FP64 else tl.float32
    re = tl.zeros((BK, BN), acc_type)
    im = tl.zeros((BK, BN), acc_type)
    for offset in range(tl.cdiv(end - start, BK)):
        pos = start + offset.to(tl.int64) * BK + k
        rows = tl.load(I + pos, pos < end, 0).to(tl.int64)
        valid = (pos[:, None] < end) & (ns[None, :] < N)
        vr, vi = _csc_product(A, B, pos[:, None],
                              rows[:, None] * BS0 + (ns[None, :] + N_OFFSET) * BS1,
                              valid, COMPLEX, FP64, CONJ)
        re += vr
        im += vi
    dst = (col if DIRECT else item) * YS0 + ns * YS1
    if COMPLEX:
        tl.store(Y + 2 * dst, tl.sum(re, 0), ns < N)
        tl.store(Y + 2 * dst + 1, tl.sum(im, 0), ns < N)
    else:
        tl.store(Y + dst, tl.sum(re, 0), ns < N)


@triton.jit
def _csc_split_store_kernel(X, Y, Counts, FIRST, COUNT, N, OFFSET, YS0, YS1,
                            BN: tl.constexpr, COMPLEX: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    active = tl.load(Counts + FIRST + row) > 0
    mask = active & (n < N)
    src = row * N + n
    dst = (FIRST + row) * YS0 + (OFFSET + n) * YS1
    if COMPLEX:
        tl.store(Y + 2 * dst, tl.load(X + 2 * src, mask, 0), mask)
        tl.store(Y + 2 * dst + 1, tl.load(X + 2 * src + 1, mask, 0), mask)
    else:
        tl.store(Y + dst, tl.load(X + src, mask, 0), mask)


def _run_csc_split(data, indices, ptr, dense, shape, op, cfg, phases, stats, output_dtype):
    # The shared reduction kernels consume only partial sums, never a CSR matrix.
    from ._spmm_csr_kernels import reduce_kernel, finish_kernel
    vector = dense.ndim == 1
    n = 1 if vector else dense.shape[1]
    columns = shape[1]
    complex_ = data.is_complex()
    view = lambda value: torch.view_as_real(value) if complex_ else value
    acc = tl.float64 if data.dtype in (torch.float64, torch.complex128) else tl.float32
    element_bytes = data.element_size()
    launch = dict(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    stats.update(segment_count=0, reduction_levels=0, segment_batches=0,
                 workspace_peak_bytes=0, descriptor_peak_bytes_estimate=0)
    with phases.measure("compute_ms"):
        result = torch.zeros((columns, n), device=data.device, dtype=output_dtype)
    if not columns or not n:
        return result[:, 0] if vector else result
    with phases.measure("process_gpu_ms"):
        starts = ptr[:-1].to(torch.int64)
        lengths = ptr[1:].to(torch.int64) - starts
        singles = torch.nonzero((lengths > 0) & (lengths <= cfg["segment_nnz"])).flatten()
        single_start = starts[singles]
        single_end = single_start + lengths[singles]
        counts = torch.where(lengths > cfg["segment_nnz"],
                             torch.div(lengths + cfg["segment_nnz"] - 1,
                                       cfg["segment_nnz"], rounding_mode="floor"), 0)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        stats["segment_count"] = int(offsets[-1].item())
    bn = cfg["block_n"]
    bs1 = 0 if vector else dense.stride(1)
    if singles.numel():
        with phases.measure("compute_ms"):
            _csc_vector_kernel[(singles.numel(), triton.cdiv(n, bn))](
                view(data), indices, ptr, view(dense), view(result), singles,
                single_start, single_end, singles.numel(), n,
                dense.stride(0), bs1, n, 1, 0,
                cfg["block_nnz"], bn, True, complex_, acc == tl.float64, op == "conj",
                DIRECT=True, **launch)
    stats["descriptor_peak_bytes_estimate"] = sum(t.numel() * t.element_size() for t in
        (starts, lengths, singles, single_start, single_end, counts, offsets))
    if not stats["segment_count"]:
        return result[:, 0] if vector else result
    # Four slabs cover an accumulator, two reduction buffers and transition room.
    wave = min(n, bn, cfg["workspace_bytes"] // (4 * element_bytes))
    if wave < 1:
        raise ValueError("CSC workspace cannot hold one panel element")
    capacity = cfg["workspace_bytes"] // (4 * wave * element_bytes)
    base_descriptors = stats["descriptor_peak_bytes_estimate"]
    stats["descriptor_peak_bytes_estimate"] = base_descriptors
    for panel in range(0, n, wave):
        width = min(wave, n - panel)
        for first_col in range(0, columns, capacity):
            last_col = min(columns, first_col + capacity)
            with phases.measure("process_gpu_ms"):
                group_offsets = offsets[first_col:last_col + 1]
                begin, end = int(group_offsets[0].item()), int(group_offsets[-1].item())
                local_rows = torch.arange(last_col - first_col, device=data.device, dtype=torch.int64)
            if begin == end:
                continue
            with phases.measure("compute_ms"):
                accumulated = torch.zeros((last_col - first_col, width), device=data.device, dtype=data.dtype)
            accumulator_bytes = accumulated.numel() * element_bytes
            for first in range(begin, end, capacity):
                last = min(end, first + capacity)
                size = last - first
                with phases.measure("process_gpu_ms"):
                    ids = torch.arange(first, last, device=data.device, dtype=torch.int64)
                    owners = torch.searchsorted(offsets[1:], ids, right=True)
                    seg_start = starts[owners] + (ids - offsets[owners]) * cfg["segment_nnz"]
                    seg_end = torch.minimum(seg_start + cfg["segment_nnz"], starts[owners] + lengths[owners])
                    local_offsets = group_offsets.clamp(first, last) - first
                    local_counts = local_offsets[1:] - local_offsets[:-1]
                with phases.measure("compute_ms"):
                    partial = torch.empty((size, width), device=data.device, dtype=data.dtype)
                    _csc_vector_kernel[(size, triton.cdiv(width, bn))](
                        view(data), indices, ptr, view(dense), view(partial), owners,
                        seg_start, seg_end, size, width, dense.stride(0), bs1, width, 1, panel,
                        cfg["block_nnz"], bn, True, complex_, acc == tl.float64, op == "conj", **launch)
                peak = accumulator_bytes + partial.numel() * element_bytes
                levels = 0
                while True:
                    with phases.measure("process_gpu_ms"):
                        maximum = int(local_counts.max().item())
                    if maximum <= 1:
                        break
                    with phases.measure("process_gpu_ms"):
                        next_counts = torch.div(local_counts + cfg["reduce_block_size"] - 1,
                                                cfg["reduce_block_size"], rounding_mode="floor")
                        next_offsets = torch.cat((counts.new_zeros(1), next_counts.cumsum(0)))
                        next_size = int(next_offsets[-1].item())
                        group_ids = torch.arange(next_size, device=data.device, dtype=torch.int64)
                        group_rows = torch.searchsorted(next_offsets[1:], group_ids, right=True)
                        reduce_start = local_offsets[group_rows] + (group_ids - next_offsets[group_rows]) * cfg["reduce_block_size"]
                        reduce_end = torch.minimum(reduce_start + cfg["reduce_block_size"], local_offsets[group_rows + 1])
                    with phases.measure("compute_ms"):
                        reduced = torch.empty((next_size, width), device=data.device, dtype=data.dtype)
                        peak = max(peak, accumulator_bytes + (partial.numel() + reduced.numel()) * element_bytes)
                        reduce_kernel[(next_size, triton.cdiv(width, bn))](
                            view(partial), view(reduced), reduce_start, reduce_end, width,
                            cfg["reduce_block_size"], bn, complex_, acc, **launch)
                        partial = reduced
                        del reduced
                    local_counts, local_offsets = next_counts, next_offsets
                    levels += 1
                with phases.measure("compute_ms"):
                    finish_kernel[(local_rows.numel(), triton.cdiv(width, bn))](
                        view(partial), view(accumulated), local_rows, local_offsets, local_counts,
                        width, 0, width, 1, bn, complex_, **launch)
                    del partial
                stats["workspace_peak_bytes"] = max(stats["workspace_peak_bytes"], peak)
                stats["reduction_levels"] = max(stats["reduction_levels"], levels)
                stats["segment_batches"] += 1
                stats["descriptor_peak_bytes_estimate"] = max(stats["descriptor_peak_bytes_estimate"],
                    base_descriptors + (size * 12 + local_rows.numel() * 8) * 8)
            with phases.measure("compute_ms"):
                _csc_split_store_kernel[(local_rows.numel(), triton.cdiv(width, bn))](
                    view(accumulated), view(result), counts, first_col, local_rows.numel(),
                    width, panel, n, 1, bn, complex_, **launch)
                del accumulated
    return result[:, 0] if vector else result


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
    if alg in ("csc_col_vector", "csc_col_kparallel"):
        _csc_vector_kernel[(shape[1], triton.cdiv(n, cfg["block_n"]))](
            a, indices, indptr, b, y, indices, indices, indices, shape[1], n,
            dense.stride(0), 0 if vector else dense.stride(1),
            result.stride(0), 0 if vector else result.stride(1), 0,
            cfg["block_nnz"], cfg["block_n"], False, complex_values,
            data.dtype in (torch.float64, torch.complex128), op == "conj",
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        return result
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


SPMV_CSC_NEW_ALGORITHMS = ("csc_col_subgroup", "csc_col_tile_atomic",
                           "csc_col_vector", "csc_col_split_reduce")


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
    from ._spmx_column_config import CSC_ALGORITHMS
    from ._spmm_csr_config import BACKENDS
    base = "spmv_csc_base" if kind == "spmv" else "spmm_csc_base"
    names = (("csc_col_subgroup", "csc_col_tile_atomic", "csc_col_vector", "csc_col_split_reduce")
             if kind == "spmv" else
             ("csc_col_panel", "csc_col_tile_panel_atomic", "csc_col_kparallel", "csc_col_split_reduce"))
    if alg not in (base, *names):
        raise ValueError(f"unknown {kind} CSC algorithm {alg!r}")
    return dict(name=alg, ops=CSC_ALGORITHMS.get(alg, ("non", "trans", "conj")),
                value_dtypes=SUPPORTED_SPMV_CSC_VALUE_DTYPES,
                backends=BACKENDS,
                layouts=("row", "col", "strided"), implementation_version=3,
                half_output_dtypes={dtype: (dtype, torch.float32)
                                    for dtype in (torch.float16, torch.bfloat16)},
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
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if len(shape) != 2 or min(shape) < 0:
        raise ValueError("shape must contain two nonnegative dimensions")
    _prepare_spmv_csc_matrix(data, indices, indptr, shape, value_dtypes=spec["value_dtypes"])
    options = dict(legacy_options or {})
    if selected != base and any(value is not None and not (kind == "spmv" and key == "block_nnz" and value == 256) for key, value in options.items()):
        raise ValueError("legacy block parameters conflict with new algorithm; use config")
    route = PreparedCscRoute(data, indices, indptr, shape, name, selected, config,
                             _normalize_spmv_csc_index_fallback_policy(policy), kind)
    route.legacy_options = options if selected == base else {}
    return route


def get_spmv_csc_algorithm_spec(alg):
    return _csc_spec(alg, "spmv")


def list_spmv_csc_algorithms(op=None, dtype=None, backend=None, *, rhs_dtype=None, out_dtype=None):
    if rhs_dtype is not None or out_dtype is not None:
        if dtype is None:
            raise ValueError("dtype is required for type-combination filtering")
        try:
            _mixed_spmx.resolve_registered_types(
                dtype, dtype if rhs_dtype is None else rhs_dtype, out_dtype=out_dtype)
        except TypeError:
            return ()
    names = ("spmv_csc_base", *SPMV_CSC_NEW_ALGORITHMS)
    return tuple(name for name in names if
                 (op is None or _spmv_csc_op_to_name(op) in _csc_spec(name, "spmv")["ops"]) and
                 (dtype is None or dtype in _csc_spec(name, "spmv")["value_dtypes"]) and
                 (backend is None or backend in _csc_spec(name, "spmv")["backends"]))


def _run_csc_registered(prepared, dense, *, alg=None, config=None, op=None, out=None,
                        out_dtype=None, return_time=False, return_meta=False, timing=False):
    from contextlib import nullcontext
    from ._spmm_csr_runtime import backend_caps, Phases
    from ._spmx_column_config import resolve_csc_config, CSC_ALGORITHMS
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
    output_dtype = _mixed_spmx.resolve_registered_types(
        data.dtype, dense.dtype, out=out, out_dtype=out_dtype
    )["output_dtype"]
    if out is not None:
        if out.shape != out_shape or out.dtype != output_dtype or out.device != data.device:
            raise ValueError("out shape/dtype/device mismatch")
        if any(torch._C._overlaps(out, value) for value in (data, indices, ptr, dense)):
            raise ValueError("out overlaps an input")
        from ._spmm_csr_runtime import output_has_internal_overlap
        if output_has_internal_overlap(out):
            raise ValueError("out must have non-overlapping storage")
    cfg, info = {}, dict(backend=_backend_name())
    if selected in CSC_ALGORITHMS:
        cfg, info = resolve_csc_config(selected, str(data.dtype).removeprefix("torch."),
                                       1 if vector else dense.shape[1], backend_caps(data.device), cfg_input, op=prepared.op, kind=prepared.kind)
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
            if data.dtype in (torch.float16, torch.bfloat16):
                a, b = a.float(), b.float()
        stats = {}
        def compute(ii, pp):
            if selected == "csc_col_split_reduce":
                return _run_csc_split(a, ii, pp, b, prepared.shape, prepared.op, cfg, phases, stats, output_dtype)
            if selected in CSC_ALGORITHMS:
                return _launch_csc_columns(a, ii, pp, b, prepared.shape, prepared.op, selected, cfg)
            raise ValueError("legacy CSC routes require the original prepared interface")
        with (nullcontext() if selected == "csc_col_split_reduce" else phases.measure("compute_ms")):
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
        with phases.measure("compute_ms"):
            result = result.to(output_dtype)
            if out is not None:
                out.copy_(result)
                result = out
        meta = dict(info, alg=selected, alg_requested=alg or prepared.alg, alg_resolved=selected,
                    a_dtype=str(data.dtype), rhs_dtype=str(dense.dtype), output_dtype=str(output_dtype),
                    config=cfg, op=prepared.op, implementation_version=2, timing_contract_version=2,
                    compute_dtype="float64" if data.dtype in (torch.float64, torch.complex128) else "float32",
                    input_indices_dtype=str(indices.dtype), input_indptr_dtype=str(ptr.dtype),
                    execution_indices_dtype=str(i.dtype), execution_indptr_dtype=str(p.dtype),
                    index_fallback_applied=fallback_reason is not None, index_fallback_reason=fallback_reason,
                    transpose_strategy="native_column_reduce" if prepared.op != "non" else "native_scatter",
                    native_format="csc", execution_path="native_csc", process_cpu_ms=0.0,
                    component_dtype="float64" if data.dtype in (torch.float64, torch.complex128) else "float32",
                    dense_stride=tuple(dense.stride()), output_stride=tuple(result.stride()))
        meta.update(stats)
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
                            out_dtype=None, return_time=False, return_meta=False, timing=False):
    from .mixed_spmx import validate_registered_output_alias
    validate_registered_output_alias(prepared, x, out)
    if isinstance(prepared, PreparedCscRoute):
        if prepared.kind != "spmv":
            raise TypeError("prepared must describe CSC SpMV")
        return _run_csc_registered(prepared, x, alg=alg, config=config, op=op, out=out,
                                  out_dtype=out_dtype, return_time=return_time, return_meta=return_meta, timing=timing)
    if alg not in (None, "auto", "spmv_csc_base") or config is not None:
        raise ValueError("prepare with alg before switching a legacy CSC prepared route")
    if prepared.data.dtype in (torch.float16, torch.bfloat16):
        from copy import copy
        from .spmv_coo import _coo_measure_run
        from ._spmm_csr_runtime import output_has_internal_overlap

        code = prepared.op if op is None else _normalize_spmv_csc_op(op)
        if code != prepared.op:
            raise ValueError("op conflicts with prepared CSC")
        output_dtype = _mixed_spmx.resolve_registered_types(
            prepared.data.dtype, x.dtype, out=out, out_dtype=out_dtype
        )["output_dtype"]
        rows = prepared.n_cols if _spmv_csc_op_transposes(code) else prepared.n_rows
        if out is not None:
            if out.shape != (rows,) or out.dtype != output_dtype or out.device != prepared.data.device:
                raise ValueError("out shape/dtype/device mismatch")
            if any(torch._C._overlaps(out, value) for value in
                   (prepared.data, prepared.kernel_indices, prepared.kernel_indptr, x)):
                raise ValueError("out overlaps an input")
            if output_has_internal_overlap(out):
                raise ValueError("out must have non-overlapping storage")
        vector = _validate_spmv_csc_x(x, prepared, code)

        def execute(phases):
            with phases.measure("process_gpu_ms"):
                execution = copy(prepared)
                execution.data = prepared.data.float()
                operand = vector.float()
            with phases.measure("compute_ms"):
                result = _run_spmv_csc_prepared_with_fallback(execution, operand, code).to(output_dtype)
                if out is not None:
                    out.copy_(result)
                    result = out
            return result, dict(alg="spmv_csc_base", alg_resolved="spmv_csc_base",
                a_dtype=str(prepared.data.dtype), rhs_dtype=str(x.dtype),
                compute_dtype="torch.float32", output_dtype=str(output_dtype),
                conversion_mode="materialize", op=_spmv_csc_op_to_name(code),
                execution_path="torch_npu" if _is_ascend_runtime() else "native_csc",
                process_cpu_ms=0.0)

        return _coo_measure_run(execute, return_time, return_meta, timing)
    return flagsparse_spmv_csc(prepared=prepared, x=x, op=op, out=out,
                               out_dtype=out_dtype, return_time=return_time, return_meta=return_meta)
