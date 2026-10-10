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

"""COO SpMV **without CSR / indptr** (fp32/fp64/complex64/complex128).

- ``sort_by_row=True`` (default): lex-sort (row,col), build compact **row-run** offsets
  ``seg_starts`` (length #runs+1, not ``n_rows+1``), one Triton program per run — register
  reduction + single ``tl.store`` per output row (no atomics on ``y``).
- ``sort_by_row=False``: grid over NNZ with ``tl.atomic_add`` (slower / contentions).

Storage: sorted ``data, row, col`` plus optional ``seg_starts`` vector — never ``indptr``.
"""

from ._common import *

import time

import triton
import triton.language as tl

SUPPORTED_SPMV_COO_VALUE_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_COO_OP_NON = 0
SPMV_COO_OP_TRANS = 1
SPMV_COO_OP_CONJ_TRANS = 2
SPMV_COO_OP_NAMES = {
    SPMV_COO_OP_NON: "non",
    SPMV_COO_OP_TRANS: "trans",
    SPMV_COO_OP_CONJ_TRANS: "conj",
}
_SPMV_COO_OP_NAME_TO_CODE = {name: code for code, name in SPMV_COO_OP_NAMES.items()}


def _spmv_coo_dtype_error_message():
    return "COO SpMV supports float16, float32, float64, complex64, and complex128"


def _normalize_spmv_coo_op(op=None, transpose=False):
    if op is None:
        return SPMV_COO_OP_TRANS if bool(transpose) else SPMV_COO_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_COO_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_COO_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_COO_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_coo_op_to_name(op):
    op_code = _normalize_spmv_coo_op(op)
    return SPMV_COO_OP_NAMES[op_code]


def _spmv_coo_op_transposes(op):
    return _normalize_spmv_coo_op(op) in (
        SPMV_COO_OP_TRANS,
        SPMV_COO_OP_CONJ_TRANS,
    )


def _normalize_spmv_coo_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedCoo:
    """Prepared COO structure metadata that can serve non/trans/conj at runtime."""

    __slots__ = (
        "data_non",
        "row_non",
        "col_non",
        "seg_starts_non",
        "data_trans",
        "row_trans",
        "col_trans",
        "seg_starts_trans",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "sort_by_row",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
    )

    def __init__(
        self,
        data_non,
        row_non,
        col_non,
        shape,
        seg_starts_non=None,
        data_trans=None,
        row_trans=None,
        col_trans=None,
        seg_starts_trans=None,
        sort_by_row=True,
        transpose=False,
        op=None,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data_non = data_non
        self.row_non = row_non
        self.col_non = col_non
        self.seg_starts_non = seg_starts_non
        self.data_trans = data_trans if data_trans is not None else data_non
        self.row_trans = row_trans if row_trans is not None else col_non
        self.col_trans = col_trans if col_trans is not None else row_non
        self.seg_starts_trans = seg_starts_trans
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows, self.n_cols = self.shape
        self.nnz = int(data_non.numel())
        self.sort_by_row = bool(sort_by_row)
        self.op = _normalize_spmv_coo_op(op, transpose=transpose)
        self.transpose = _spmv_coo_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


class _PreparedCooLaunch:
    """Concrete launch view for one runtime op on top of structure-only prepare."""

    __slots__ = (
        "data",
        "row",
        "col",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "seg_starts",
        "n_segs",
        "use_seg_kernel",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
    )

    def __init__(
        self,
        data,
        row,
        col,
        shape,
        seg_starts=None,
        op=SPMV_COO_OP_NON,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data = data
        self.row = row
        self.col = col
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows, self.n_cols = self.shape
        self.nnz = int(data.numel())
        self.seg_starts = seg_starts
        if seg_starts is None:
            self.n_segs = 0
            self.use_seg_kernel = False
        else:
            self.n_segs = int(seg_starts.numel()) - 1
            self.use_seg_kernel = self.n_segs > 0
        self.op = _normalize_spmv_coo_op(op)
        self.transpose = _spmv_coo_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


@triton.jit
def _spmv_coo_seg_f32(
    data_ptr,
    col_ptr,
    row_ptr,
    x_ptr,
    y_ptr,
    seg_starts_ptr,
    alpha,
    beta,
    n_segs,
    BLOCK_INNER: tl.constexpr,
    SEG_IS_ROW: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """y[row] = alpha * sum(A[row] * x) + beta * y[row].

    alpha/beta/HAS_BETA exist so the C API can express cuSPARSE's SpMV in one
    launch; this module's callers pass 1 and 0 and the generated code is
    unchanged. SEG_IS_ROW says what a segment IS: False (this module) means one
    entry per RUN of equal row ids, so a row with no nonzeros gets no program at
    all -- fine when y was zeroed first. True means seg_starts is a full
    row-offsets array and the segment index IS the row id, which is what lets
    beta * y reach an empty row. Reading the row from the COO would be wrong
    there: an empty row has start == end and would pick up the NEXT row's id.
    """
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    if SEG_IS_ROW:
        row_id = seg
    else:
        row_id = tl.load(row_ptr + start)
    acc = tl.zeros((), dtype=tl.float32)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        v = tl.load(data_ptr + offs, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        xv = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, v * xv, 0.0))
        pos += BLOCK_INNER
    out = alpha * acc
    if HAS_BETA:
        out = out + beta * tl.load(y_ptr + row_id)
    tl.store(y_ptr + row_id, out)


@triton.jit
def _spmv_coo_seg_f64(
    data_ptr,
    col_ptr,
    row_ptr,
    x_ptr,
    y_ptr,
    seg_starts_ptr,
    alpha,
    beta,
    n_segs,
    BLOCK_INNER: tl.constexpr,
    SEG_IS_ROW: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """y[row] = alpha * sum(A[row] * x) + beta * y[row].

    alpha/beta/HAS_BETA exist so the C API can express cuSPARSE's SpMV in one
    launch; this module's callers pass 1 and 0 and the generated code is
    unchanged. SEG_IS_ROW says what a segment IS: False (this module) means one
    entry per RUN of equal row ids, so a row with no nonzeros gets no program at
    all -- fine when y was zeroed first. True means seg_starts is a full
    row-offsets array and the segment index IS the row id, which is what lets
    beta * y reach an empty row. Reading the row from the COO would be wrong
    there: an empty row has start == end and would pick up the NEXT row's id.
    """
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    if SEG_IS_ROW:
        row_id = seg
    else:
        row_id = tl.load(row_ptr + start)
    acc = tl.zeros((), dtype=tl.float64)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        v = tl.load(data_ptr + offs, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        xv = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, v * xv, 0.0))
        pos += BLOCK_INNER
    out = alpha * acc
    if HAS_BETA:
        out = out + beta * tl.load(y_ptr + row_id)
    tl.store(y_ptr + row_id, out)


@triton.jit
def _spmv_coo_seg_complex(
    data_ri_ptr,
    col_ptr,
    row_ptr,
    x_ri_ptr,
    y_ri_ptr,
    seg_starts_ptr,
    alpha_re,
    alpha_im,
    beta_re,
    beta_im,
    n_segs,
    BLOCK_INNER: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    SEG_IS_ROW: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Complex counterpart; see _spmv_coo_seg_f32 for SEG_IS_ROW and HAS_BETA."""
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    if SEG_IS_ROW:
        row_id = seg
    else:
        row_id = tl.load(row_ptr + start)
    acc_re = tl.zeros((), dtype=ACC_DTYPE)
    acc_im = tl.zeros((), dtype=ACC_DTYPE)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        a_re = tl.load(data_ri_ptr + offs * 2, mask=m, other=0.0)
        a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        x_re = tl.load(x_ri_ptr + c * 2, mask=m, other=0.0)
        x_im = tl.load(x_ri_ptr + c * 2 + 1, mask=m, other=0.0)
        prod_re = tl.where(m, a_re * x_re - a_im * x_im, 0.0)
        prod_im = tl.where(m, a_re * x_im + a_im * x_re, 0.0)
        acc_re += tl.sum(prod_re)
        acc_im += tl.sum(prod_im)
        pos += BLOCK_INNER
    out_re = alpha_re * acc_re - alpha_im * acc_im
    out_im = alpha_re * acc_im + alpha_im * acc_re
    if HAS_BETA:
        prev_re = tl.load(y_ri_ptr + row_id * 2)
        prev_im = tl.load(y_ri_ptr + row_id * 2 + 1)
        out_re = out_re + beta_re * prev_re - beta_im * prev_im
        out_im = out_im + beta_re * prev_im + beta_im * prev_re
    tl.store(y_ri_ptr + row_id * 2, out_re)
    tl.store(y_ri_ptr + row_id * 2 + 1, out_im)


@triton.jit
def _spmv_coo_atomic_f32(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(data_ptr + offs, mask=m, other=0.0)
    xv = tl.load(x_ptr + c, mask=m, other=0.0)
    contrib = tl.where(m, v * xv, 0.0).to(tl.float32)
    tl.atomic_add(y_ptr + r, contrib, mask=m, sem="relaxed")


@triton.jit
def _spmv_coo_atomic_complex(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    x_ri_ptr,
    y_ri_ptr,
    nnz,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=m, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=m, other=0.0)
    x_re = tl.load(x_ri_ptr + c * 2, mask=m, other=0.0)
    x_im = tl.load(x_ri_ptr + c * 2 + 1, mask=m, other=0.0)
    prod_re = tl.where(m, a_re * x_re - a_im * x_im, 0.0).to(ACC_DTYPE)
    prod_im = tl.where(m, a_re * x_im + a_im * x_re, 0.0).to(ACC_DTYPE)
    tl.atomic_add(y_ri_ptr + r * 2, prod_re, mask=m, sem="relaxed")
    tl.atomic_add(y_ri_ptr + r * 2 + 1, prod_im, mask=m, sem="relaxed")


@triton.jit
def _spmv_coo_atomic_f64(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(data_ptr + offs, mask=m, other=0.0)
    xv = tl.load(x_ptr + c, mask=m, other=0.0)
    contrib = tl.where(m, v * xv, 0.0).to(tl.float64)
    tl.atomic_add(y_ptr + r, contrib, mask=m, sem="relaxed")


def _sort_coo_lex_inplace(data, row, col, n_cols):
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    index_dtype = torch.promote_types(row.dtype, col.dtype)
    if data.numel() == 0:
        return (
            data.contiguous(),
            row.to(index_dtype).contiguous(),
            col.to(index_dtype).contiguous(),
        )
    key = row64 * max(1, int(n_cols)) + col64
    order = torch.argsort(key)
    return (
        _gather_values(data, order).contiguous(),
        row[order].to(index_dtype).contiguous(),
        col[order].to(index_dtype).contiguous(),
    )


def _seg_starts_from_sorted_rows(row, nnz, device):
    """Boundaries of constant-row runs in sorted COO."""
    if nnz == 0:
        return None
    index_dtype = row.dtype
    if nnz > _INDEX_LIMIT_INT32:
        index_dtype = torch.int64
    diff = row[1:] != row[:-1]
    breaks = torch.nonzero(diff, as_tuple=False).flatten().to(index_dtype) + 1
    return torch.cat(
        [
            torch.zeros(1, dtype=index_dtype, device=device),
            breaks,
            torch.tensor([nnz], dtype=index_dtype, device=device),
        ]
    )


def _prepare_coo_tensors(data, row, col, shape, sort_by_row):
    if not all(torch.is_tensor(t) for t in (data, row, col)):
        raise TypeError("data, row, col must all be torch.Tensor")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if not all(_is_accel_tensor(t) for t in (data, row, col)):
        raise ValueError("data, row, col must be CUDA tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have the same length")
    if data.dtype not in SUPPORTED_SPMV_COO_VALUE_DTYPES:
        raise TypeError(_spmv_coo_dtype_error_message())
    if row.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    index_dtype = torch.promote_types(row.dtype, col.dtype)
    if sort_by_row:
        data, kr, kc = _sort_coo_lex_inplace(data, row, col, n_cols)
        seg = _seg_starts_from_sorted_rows(kr, data.numel(), data.device)
    else:
        data = data.contiguous()
        kr = row.to(index_dtype).contiguous()
        kc = col.to(index_dtype).contiguous()
        seg = None
    if data.numel() > 0:
        row64 = kr.to(torch.int64)
        col64 = kc.to(torch.int64)
        if int(row64.min().item()) < 0 or int(row64.max().item()) >= n_rows:
            raise IndexError("row indices out of range")
        if int(col64.min().item()) < 0 or int(col64.max().item()) >= n_cols:
            raise IndexError("col indices out of range")
    return data, kr, kc, seg


def prepare_spmv_coo(
    data,
    row,
    col,
    shape,
    sort_by_row=None,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
    *, alg=None, config=None,
):
    """Cache sorted COO + row-run ``seg_starts`` when ``sort_by_row``. No ``indptr``."""
    if alg is None and data.dtype == torch.float16:
        alg = "coo_atomic" if sort_by_row is False else "coo_rowrun"
    if alg is not None:
        selected = "coo_rowrun" if alg == "auto" else alg
        get_spmv_coo_algorithm_spec(selected)
        if sort_by_row is not None and bool(sort_by_row) != ("atomic" not in selected):
            raise ValueError("sort_by_row conflicts with alg")
        op_code = _normalize_spmv_coo_op(op, transpose=transpose)
        if transpose and op_code == 0:
            raise ValueError("transpose conflicts with op")
        if len(shape) != 2 or min(shape) < 0:
            raise ValueError("shape must contain two nonnegative dimensions")
        if row.device != data.device or col.device != data.device:
            raise ValueError("COO inputs must share a device")
        _prepare_coo_tensors(data, row, col, shape, False)
        return PreparedCooRoute(data, row, col, shape, op_code, alg, config,
                                _normalize_spmv_coo_index_fallback_policy(index_fallback_policy))
    if config is not None:
        raise ValueError("config requires an explicit alg")
    sort_by_row = True if sort_by_row is None else bool(sort_by_row)
    index_fallback_policy = _normalize_spmv_coo_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_coo_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_COO_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    shape = (int(shape[0]), int(shape[1]))
    data_non, row_non, col_non, seg_non = _prepare_coo_tensors(
        data, row, col, shape, sort_by_row
    )
    trans_shape = (shape[1], shape[0])
    if sort_by_row:
        data_trans, row_trans, col_trans, seg_trans = _prepare_coo_tensors(
            data,
            col,
            row,
            trans_shape,
            True,
        )
    else:
        data_trans = data_non
        row_trans = col_non
        col_trans = row_non
        seg_trans = None
    return PreparedCoo(
        data_non,
        row_non,
        col_non,
        shape,
        seg_starts_non=seg_non,
        data_trans=data_trans,
        row_trans=row_trans,
        col_trans=col_trans,
        seg_starts_trans=seg_trans,
        sort_by_row=sort_by_row,
        transpose=transpose,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _prepare_spmv_coo_launch_from_raw(
    data,
    row,
    col,
    shape,
    sort_by_row=True,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    """Build one runtime launch view from raw COO inputs for benchmark-only timing."""
    index_fallback_policy = _normalize_spmv_coo_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_coo_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_COO_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    shape = (int(shape[0]), int(shape[1]))
    launch_shape = shape
    launch_data = data
    launch_row = row
    launch_col = col
    if _spmv_coo_op_transposes(op_code):
        launch_shape = (shape[1], shape[0])
        launch_row = col
        launch_col = row
        if op_code == SPMV_COO_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
            launch_data = data.conj()
            if hasattr(launch_data, "resolve_conj"):
                launch_data = launch_data.resolve_conj()
    launch_data, launch_row, launch_col, launch_seg_starts = _prepare_coo_tensors(
        launch_data,
        launch_row,
        launch_col,
        launch_shape,
        sort_by_row,
    )
    return _PreparedCooLaunch(
        data=launch_data,
        row=launch_row,
        col=launch_col,
        shape=launch_shape,
        seg_starts=launch_seg_starts,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _resolve_spmv_coo_launch(prepared, op):
    op_code = _normalize_spmv_coo_op(op)
    if op_code == SPMV_COO_OP_NON:
        return _PreparedCooLaunch(
            data=prepared.data_non,
            row=prepared.row_non,
            col=prepared.col_non,
            shape=prepared.shape,
            seg_starts=prepared.seg_starts_non,
            op=op_code,
            index_fallback_policy=prepared.index_fallback_policy,
            index_fallback_applied=prepared.index_fallback_applied,
            index_fallback_reason=prepared.index_fallback_reason,
        )
    data = prepared.data_trans
    if op_code == SPMV_COO_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
        data = data.conj()
        if hasattr(data, "resolve_conj"):
            data = data.resolve_conj()
    return _PreparedCooLaunch(
        data=data,
        row=prepared.row_trans,
        col=prepared.col_trans,
        shape=(prepared.n_cols, prepared.n_rows),
        seg_starts=prepared.seg_starts_trans,
        op=op_code,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=prepared.index_fallback_applied,
        index_fallback_reason=prepared.index_fallback_reason,
    )


def _validate_x_coo(x, prepared):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    if x.numel() != prepared.n_cols:
        raise ValueError(f"x length must be n_cols={prepared.n_cols}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


def _triton_spmv_coo_kernel(prepared, x, block_size, num_warps, block_inner):
    dtype = prepared.data.dtype
    y = torch.zeros(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    nnz = prepared.nnz
    if nnz == 0:
        return y
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.zeros(prepared.n_rows * 2, dtype=data_ri.dtype, device=y.device)
        acc_dtype = tl.float64 if dtype == torch.complex128 else tl.float32
        if prepared.use_seg_kernel:
            grid = (prepared.n_segs,)
            _spmv_coo_seg_complex[grid](
                data_ri,
                prepared.col,
                prepared.row,
                x_ri,
                y_ri,
                prepared.seg_starts,
                # y = A @ x here; alpha/beta exist for the C API's
                # cuSPARSE-compatible signature and SEG_IS_ROW=False keeps the
                # run-compressed seg_starts this path builds. All three fold away.
                1,
                0,
                0,
                0,
                prepared.n_segs,
                BLOCK_INNER=block_inner,
                ACC_DTYPE=acc_dtype,
                SEG_IS_ROW=False,
                HAS_BETA=False,
                num_warps=1,
            )
        else:
            grid = (triton.cdiv(nnz, block_size),)
            _spmv_coo_atomic_complex[grid](
                data_ri,
                prepared.row,
                prepared.col,
                x_ri,
                y_ri,
                nnz,
                BLOCK=block_size,
                ACC_DTYPE=acc_dtype,
                num_warps=num_warps,
            )
        y.copy_(torch.view_as_complex(y_ri.reshape(prepared.n_rows, 2)))
        return y
    if prepared.use_seg_kernel:
        ker = _spmv_coo_seg_f64 if dtype == torch.float64 else _spmv_coo_seg_f32
        grid = (prepared.n_segs,)
        ker[grid](
            prepared.data,
            prepared.col,
            prepared.row,
            x,
            y,
            prepared.seg_starts,
            # See the complex branch above: identity alpha/beta, run-compressed
            # segments, both constexprs fold away.
            1,
            0,
            prepared.n_segs,
            BLOCK_INNER=block_inner,
            SEG_IS_ROW=False,
            HAS_BETA=False,
            num_warps=1,
        )
        return y
    ker = _spmv_coo_atomic_f64 if dtype == torch.float64 else _spmv_coo_atomic_f32
    grid = (triton.cdiv(nnz, block_size),)
    ker[grid](
        prepared.data,
        prepared.row,
        prepared.col,
        x,
        y,
        nnz,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return y


def _spmv_coo_uses_int64_indices(prepared):
    return (
        prepared.row.dtype == torch.int64
        or prepared.col.dtype == torch.int64
        or (
            prepared.seg_starts is not None and prepared.seg_starts.dtype == torch.int64
        )
    )


def _spmv_coo_int32_fallback_blocker(prepared):
    for name, tensor in (("row", prepared.row), ("col", prepared.col)):
        if tensor.numel() > 0:
            min_index = int(tensor.min().item())
            max_index = int(tensor.max().item())
            if min_index < 0 or max_index > _INDEX_LIMIT_INT32:
                return (
                    f"{name} index range [{min_index}, {max_index}] cannot fit int32 "
                    f"for shape={prepared.shape}"
                )
    if prepared.seg_starts is not None and prepared.seg_starts.numel() > 0:
        max_offset = int(prepared.seg_starts[-1].item())
        if max_offset > _INDEX_LIMIT_INT32:
            return f"COO nnz offset {max_offset} cannot fit int32 for shape={prepared.shape}"
    if prepared.n_rows > _INDEX_LIMIT_INT32 or prepared.n_cols > _INDEX_LIMIT_INT32:
        return f"shape {prepared.shape} cannot fit int32 row/col metadata"
    return None


def _spmv_coo_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_coo_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(
            f"native int64 COO SpMV failed and int32 fallback is unsafe: {blocker}"
        )
    seg_starts = (
        None
        if prepared.seg_starts is None
        else prepared.seg_starts.to(torch.int32).contiguous()
    )
    return _PreparedCooLaunch(
        data=prepared.data,
        row=prepared.row.to(torch.int32).contiguous(),
        col=prepared.col.to(torch.int32).contiguous(),
        shape=prepared.shape,
        seg_starts=seg_starts,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmv_coo_prepared_with_fallback(
    prepared, x, block_size, num_warps, block_inner
):
    try:
        return _triton_spmv_coo_kernel(
            prepared,
            x,
            block_size=block_size,
            num_warps=num_warps,
            block_inner=block_inner,
        )
    except Exception as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _coo_index_compatibility_error(exc)
            or not _spmv_coo_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmv_coo_prepared_with_int32_indices(prepared, exc)
        return _triton_spmv_coo_kernel(
            fallback_prepared,
            x,
            block_size=block_size,
            num_warps=num_warps,
            block_inner=block_inner,
        )


def _resolve_spmv_coo_kernel_launch(prepared, block_size, num_warps):
    launch = _spmv_rocm_launch_overrides(
        fmt="coo",
        dtype=prepared.data.dtype,
        nnz=prepared.nnz,
        block_size=block_size,
        num_warps=num_warps,
        device=prepared.data.device,
    )
    if launch is None:
        return int(block_size), int(num_warps)
    return int(launch["block_size"]), int(launch["num_warps"])


def flagsparse_spmv_coo(
    data=None,
    row=None,
    col=None,
    x=None,
    shape=None,
    out=None,
    return_time=False,
    prepared=None,
    sort_by_row=None,
    block_size=256,
    num_warps=4,
    block_inner=128,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
    *, alg=None, config=None, return_meta=False, timing=False,
):
    """COO SpMV with no CSR indptr. See module docstring.

    ``block_inner``: tile for the row-run kernel (``sort_by_row=True``).
    ``block_size`` / ``num_warps``: grid over NNZ when ``sort_by_row=False`` (atomics).
    """
    if alg is None and prepared is None and data is not None and data.dtype == torch.float16:
        alg = "coo_atomic" if sort_by_row is False else "coo_rowrun"
    if alg is not None or isinstance(prepared, PreparedCooRoute):
        if transpose is not None and op is not None and bool(transpose) != _spmv_coo_op_transposes(op):
            raise ValueError("transpose conflicts with op")
        selected = alg if alg not in (None, "auto") else (prepared.alg if prepared is not None else "coo_rowrun")
        if sort_by_row is not None and bool(sort_by_row) != ("atomic" not in selected):
            raise ValueError("sort_by_row conflicts with alg")
        if prepared is None:
            prepared = prepare_spmv_coo(data, row, col, shape, op=op,
                transpose=bool(transpose), alg=alg, config=config,
                index_fallback_policy=index_fallback_policy)
        return flagsparse_spmv_coo_run(prepared, x, alg=alg, config=config,
            op=op if op is not None else (("trans" if transpose else "non") if transpose is not None else None), out=out,
            return_time=return_time, return_meta=return_meta, timing=timing)
    if config is not None or return_meta or timing:
        raise ValueError("config/metadata/timing require a registered alg")
    transpose_flag = False if transpose is None else bool(transpose)
    op_explicit = op is not None
    op_code = _normalize_spmv_coo_op(op, transpose=transpose_flag)
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_coo_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if prepared is None:
        if any(a is None for a in (data, row, col, x, shape)):
            raise ValueError("data, row, col, x, shape required when prepared is None")
        prepared = prepare_spmv_coo(
            data,
            row,
            col,
            shape,
            sort_by_row=sort_by_row,
            transpose=transpose_flag,
            op=op,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if x is None:
            raise TypeError("x is required when prepared is set")
        if shape is None:
            shape = prepared.shape
        sh = (int(shape[0]), int(shape[1]))
        if sh != prepared.shape:
            raise ValueError(
                f"shape {sh} does not match prepared.shape {prepared.shape}"
            )
        if not op_explicit and transpose is None:
            op_code = prepared.op
        elif not op_explicit and transpose is not None:
            op_code = _normalize_spmv_coo_op(None, transpose=transpose_flag)
    launch = _resolve_spmv_coo_launch(prepared, op_code)
    x = _validate_x_coo(x, launch)
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a power of 2 in [1, 32]")
    if block_inner <= 0 or (block_inner & (block_inner - 1)) != 0:
        raise ValueError("block_inner must be a positive power of 2")
    block_size, num_warps = _resolve_spmv_coo_kernel_launch(
        launch, block_size, num_warps
    )
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    y = _run_spmv_coo_prepared_with_fallback(
        launch,
        x,
        block_size=block_size,
        num_warps=num_warps,
        block_inner=block_inner,
    )
    elapsed_ms = None
    if return_time:
        _ACCEL.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_time:
        return y, elapsed_ms
    return y


# Native COO kernels shared by the SpMV and SpMM registered extensions.
@triton.jit
def _coo_scan_pair(a, head_a, b, head_b):
    return tl.where(head_b, b, a + b), head_a | head_b


@triton.jit
def _coo_product(A, B, k, c, n, bs0, bs1, mask,
                 COMPLEX: tl.constexpr, CONJ: tl.constexpr, FP64: tl.constexpr):
    acc_type = tl.float64 if FP64 else tl.float32
    if COMPLEX:
        ar = tl.load(A + 2 * k, mask, 0).to(acc_type)
        ai = tl.load(A + 2 * k + 1, mask, 0).to(acc_type)
        if CONJ:
            ai = -ai
        br = tl.load(B + 2 * (c * bs0 + n * bs1), mask, 0).to(acc_type)
        bi = tl.load(B + 2 * (c * bs0 + n * bs1) + 1, mask, 0).to(acc_type)
        return ar * br - ai * bi, ar * bi + ai * br
    else:
        a = tl.load(A + k, mask, 0).to(acc_type)
        b = tl.load(B + c * bs0 + n * bs1, mask, 0).to(acc_type)
        return a * b, tl.full(k.shape, 0, acc_type)


@triton.jit
def _coo_segmented_panel_kernel(A, Row, Col, B, C, NNZ, N,
                                bs0, bs1, cs0, cs1,
                                T: tl.constexpr, BN: tl.constexpr,
                                COMPLEX: tl.constexpr, CONJ: tl.constexpr,
                                FP64: tl.constexpr, SEGMENT: tl.constexpr):
    k = tl.program_id(0).to(tl.int64) * T + tl.arange(0, T)
    n = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    r = tl.load(Row + k, k < NNZ, 0).to(tl.int64)
    c = tl.load(Col + k, k < NNZ, 0).to(tl.int64)
    mask = (k[:, None] < NNZ) & (n[None, :] < N)
    real, imag = _coo_product(A, B, k[:, None], c[:, None], n[None, :],
                             bs0, bs1, mask, COMPLEX, CONJ, FP64)
    if SEGMENT:
        prev = tl.load(Row + k - 1, (k < NNZ) & (tl.arange(0, T) > 0), -1)
        head = (tl.arange(0, T) == 0) | (r != prev)
        flags = tl.broadcast_to(head[:, None], (T, BN))
        real, _ = tl.associative_scan((real, flags), 0, _coo_scan_pair)
        if COMPLEX:
            imag, _ = tl.associative_scan((imag, flags), 0, _coo_scan_pair)
        nxt = tl.load(Row + k + 1, (k + 1 < NNZ) & (tl.arange(0, T) < T - 1), -1)
        tail = (tl.arange(0, T) == T - 1) | (k + 1 == NNZ) | (r != nxt)
        mask = mask & tail[:, None]
    dst = r[:, None] * cs0 + n[None, :] * cs1
    if COMPLEX:
        tl.atomic_add(C + 2 * dst, real, mask, sem="relaxed")
        tl.atomic_add(C + 2 * dst + 1, imag, mask, sem="relaxed")
    else:
        tl.atomic_add(C + dst, real, mask, sem="relaxed")


@triton.jit
def _coo_rowrun_group_kernel(A, Row, Col, Starts, B, C, RUNS, N,
                             bs0, bs1, cs0, cs1,
                             R: tl.constexpr, V: tl.constexpr,
                             SUBGROUP: tl.constexpr, COMPLEX: tl.constexpr,
                             CONJ: tl.constexpr, FP64: tl.constexpr):
    run = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    lane = tl.arange(0, V)
    start = tl.load(Starts + run, run < RUNS, 0).to(tl.int64)
    end = tl.load(Starts + run + 1, run < RUNS, 0).to(tl.int64)
    row = tl.load(Row + start, run < RUNS, 0).to(tl.int64)
    acc_type = tl.float64 if FP64 else tl.float32
    ar = tl.zeros((R, V), acc_type)
    ai = tl.zeros((R, V), acc_type)
    br = tl.zeros((R, V), acc_type)
    bi = tl.zeros((R, V), acc_type)
    longest = tl.max(end - start, 0)
    if SUBGROUP:
        for pos in range(0, tl.cdiv(longest, V)):
            k = start[:, None] + pos * V + lane[None, :]
            mask = (run[:, None] < RUNS) & (k < end[:, None])
            c = tl.load(Col + k, mask, 0).to(tl.int64)
            vr, vi = _coo_product(A, B, k, c, 0, bs0, bs1, mask, COMPLEX, CONJ, FP64)
            ar += vr
            ai += vi
        real = tl.sum(ar, 1)
        imag = tl.sum(ai, 1)
        dst = row * cs0
        mask_out = run < RUNS
    else:
        n = tl.program_id(1).to(tl.int64) * V + lane
        for pos in range(0, tl.cdiv(longest, 2)):
            k = start[:, None] + 2 * pos
            mask = (run[:, None] < RUNS) & (k < end[:, None]) & (n[None, :] < N)
            c = tl.load(Col + k, (run[:, None] < RUNS) & (k < end[:, None]), 0).to(tl.int64)
            vr, vi = _coo_product(A, B, k, c, n[None, :], bs0, bs1, mask, COMPLEX, CONJ, FP64)
            ar += vr
            ai += vi
            k = k + 1
            mask = (run[:, None] < RUNS) & (k < end[:, None]) & (n[None, :] < N)
            c = tl.load(Col + k, (run[:, None] < RUNS) & (k < end[:, None]), 0).to(tl.int64)
            vr, vi = _coo_product(A, B, k, c, n[None, :], bs0, bs1, mask, COMPLEX, CONJ, FP64)
            br += vr
            bi += vi
        real = ar + br
        imag = ai + bi
        dst = row[:, None] * cs0 + n[None, :] * cs1
        mask_out = (run[:, None] < RUNS) & (n[None, :] < N)
    if COMPLEX:
        tl.store(C + 2 * dst, real, mask_out)
        tl.store(C + 2 * dst + 1, imag, mask_out)
    else:
        tl.store(C + dst, real, mask_out)


_COO_NEW_ALGORITHMS = (
    "coo_segmented_atomic", "coo_rowrun_subgroup",
    "coo_segmented_panel_atomic", "coo_rowrun_panel",
)
SPMV_COO_ALGORITHMS = ("coo_rowrun", "coo_atomic") + _COO_NEW_ALGORITHMS[:2]


def _resolve_coo_config(alg, dtype, device, config=None, n=1):
    from ._spmm_csr_runtime import backend_caps
    from ._spmm_csr_config import resolve_coo_config
    return resolve_coo_config(alg, str(dtype).removeprefix("torch."), backend_caps(device), config, n)


def _coo_sorted_runs(data, row, col):
    order = torch.argsort(row, stable=True)
    data, row, col = data[order], row[order], col[order]
    starts = _seg_starts_from_sorted_rows(row, row.numel(), row.device)
    return data, row, col, starts


def _launch_coo_extension(data, row, col, starts, B, shape, alg, cfg, conj=False):
    # B is always a 2D view; SpMV uses a singleton output panel.
    C = torch.zeros((shape[0], B.shape[1]), device=data.device, dtype=data.dtype)
    if not data.numel() or not C.numel():
        return C
    complex_ = data.is_complex()
    a = torch.view_as_real(data).reshape(-1) if complex_ else data
    b = torch.view_as_real(B) if complex_ else B
    c = torch.view_as_real(C) if complex_ else C
    args = (a, row, col)
    common = dict(COMPLEX=complex_, CONJ=conj,
                  FP64=data.dtype in (torch.float64, torch.complex128),
                  num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if "atomic" in alg:
        bn = cfg.get("block_n", 1)
        _coo_segmented_panel_kernel[(triton.cdiv(data.numel(), cfg["block_nnz"]),
                                     triton.cdiv(B.shape[1], bn))](
            *args, b, c, data.numel(), B.shape[1], *B.stride(), *C.stride(),
            T=cfg["block_nnz"], BN=bn, SEGMENT=cfg["local_reduce"] == "segment", **common)
    else:
        subgroup = alg == "coo_rowrun_subgroup"
        r = cfg["rows_per_program"] if subgroup else cfg["tile_rows"]
        v = cfg["lanes_per_row"] if subgroup else cfg["tile_n"]
        runs = starts.numel() - 1
        _coo_rowrun_group_kernel[(triton.cdiv(runs, r), 1 if subgroup else triton.cdiv(B.shape[1], v))](
            *args, starts, b, c, runs, B.shape[1], *B.stride(), *C.stride(),
            R=r, V=v, SUBGROUP=subgroup, **common)
    return C


class PreparedCooRoute:
    """Original COO only; execution data is rebuilt for every registered run."""
    def __init__(self, data, row, col, shape, op, alg, config=None, index_fallback_policy="auto"):
        self.data, self.row, self.col = data, row, col
        self.shape = tuple(int(v) for v in shape)
        self.n_rows, self.n_cols = self.shape
        self.op = _spmv_coo_op_to_name(op)
        self.alg = alg
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        self.config = dict(config or {})
        self.index_fallback_policy = index_fallback_policy


def get_spmv_coo_algorithm_spec(alg):
    if alg not in SPMV_COO_ALGORITHMS:
        raise ValueError(f"unknown COO SpMV algorithm {alg!r}")
    return dict(name=alg, supported_ops=("non", "trans", "conj"),
                supported_dtypes=SUPPORTED_SPMV_COO_VALUE_DTYPES,
                implementation_version=1, timing_contract_version=2,
                requires_sort="atomic" not in alg)


def list_spmv_coo_algorithms(op=None, dtype=None, backend=None):
    if op is not None:
        _normalize_spmv_coo_op(op)
    if dtype is not None and dtype not in SUPPORTED_SPMV_COO_VALUE_DTYPES:
        return ()
    # Runtime capabilities are resolved against the actual device at run time.
    if backend is not None and backend not in ("cuda", "rocm", "metax", "mthreads", "ascend", "xpu", "gcu", "mlu"):
        return ()
    return SPMV_COO_ALGORITHMS


def _coo_measure_run(execute, return_time, return_meta, timing):
    from ._spmm_csr_runtime import Phases
    measured = return_time or return_meta or timing
    if measured:
        start, end = _ACCEL.Event(enable_timing=True), _ACCEL.Event(enable_timing=True)
        start.record()
    out, meta = execute(Phases(False))
    if measured:
        end.record()
        end.synchronize()
        meta["gpu_ms"] = start.elapsed_time(end)
        meta["process_cpu_ms"] = 0.0
        meta["operator_ms"] = meta["gpu_ms"]
    if timing:
        phases = Phases(True)
        execute(phases)
        meta.update(phases.results())
    if return_time and return_meta:
        return out, meta["operator_ms"], meta
    if return_time:
        return out, meta["operator_ms"]
    return (out, meta) if return_meta else out


def flagsparse_spmv_coo_run(prepared, x, *, alg=None, config=None, op=None,
                            out=None, return_time=False, return_meta=False, timing=False):
    if not isinstance(prepared, PreparedCooRoute):
        raise TypeError("registered run requires prepare_spmv_coo(..., alg=...)")
    if op is not None and _spmv_coo_op_to_name(op) != prepared.op:
        raise ValueError("op conflicts with prepared COO")
    requested = prepared.alg if alg is None else alg
    selected = "coo_rowrun" if requested == "auto" else requested
    get_spmv_coo_algorithm_spec(selected)
    cfg_input = config if config is not None else (prepared.config if requested == prepared.alg else {})
    data, row, col = prepared.data, prepared.row, prepared.col
    trans = prepared.op != "non"
    shape = prepared.shape[::-1] if trans else prepared.shape
    if not torch.is_tensor(x):
        raise TypeError("x must be a tensor")
    if x.ndim != 1 or x.numel() != shape[1] or x.dtype != data.dtype or x.device != data.device:
        raise ValueError("x shape/dtype/device does not match COO operation")
    if out is not None:
        if out.shape != (shape[0],) or out.dtype != data.dtype or out.device != data.device:
            raise ValueError("out shape/dtype/device does not match COO operation")
        if any(torch._C._overlaps(out, t) for t in (data, row, col, x)):
            raise ValueError("out overlaps an input")
    cfg, info = ({}, {})
    if selected in _COO_NEW_ALGORITHMS:
        cfg, info = _resolve_coo_config(selected, data.dtype, data.device, cfg_input)
    elif cfg_input:
        raise ValueError("legacy COO algorithms do not accept config")
    def execute(phases):
        a, r, c = data, col if trans else row, row if trans else col
        starts = None
        with phases.measure("process_gpu_ms"):
            a, r, c = a.contiguous(), r.contiguous(), c.contiguous()
            vector = x
            if data.dtype == torch.float16:
                a, vector = a.float(), x.float()
            if "atomic" not in selected:
                a, r, c, starts = _coo_sorted_runs(a, r, c)
        with phases.measure("compute_ms"):
            reason = None
            if selected in _COO_NEW_ALGORITHMS:
                def launch(rr, cc, ss):
                    return _launch_coo_extension(a, rr, cc, ss, vector[:, None], shape,
                                                 selected, cfg, prepared.op == "conj")[:, 0]
                try:
                    y = launch(r, c, starts)
                except Exception as exc:
                    from ._spmv_csr_config import is_index_compatibility_error
                    view = _PreparedCooLaunch(a, r, c, shape, starts, 0,
                                             index_fallback_policy=prepared.index_fallback_policy)
                    if prepared.index_fallback_policy != "auto" or not is_index_compatibility_error(exc) or not _spmv_coo_uses_int64_indices(view):
                        raise
                    fallback = _spmv_coo_prepared_with_int32_indices(view, exc)
                    r, c, starts = fallback.row, fallback.col, fallback.seg_starts
                    reason = str(exc)
                    y = launch(r, c, starts)
            else:
                if prepared.op == "conj" and a.is_complex():
                    a = a.conj().resolve_conj()
                view = _PreparedCooLaunch(a, r, c, shape, starts, 0,
                                         index_fallback_policy=prepared.index_fallback_policy)
                y = _run_spmv_coo_prepared_with_fallback(view, vector, 256, 4, 128)
            y = y.to(data.dtype)
            if out is not None:
                out.copy_(y)
                y = out
        return y, dict(info, alg=selected, alg_requested=requested, alg_resolved=selected,
                       op=prepared.op, compute_dtype=str(torch.float32 if data.dtype == torch.float16 else data.dtype),
                       component_dtype="float64" if data.dtype in (torch.float64, torch.complex128) else "float32",
                       backend=_backend_name(),
                       input_indices=(str(row.dtype), str(col.dtype)),
                       execution_indices=(str(r.dtype), str(c.dtype)), fallback_reason=reason,
                       transpose_strategy="index_roles", implementation_version=1,
                       timing_contract_version=2)
    return _coo_measure_run(execute, return_time, return_meta, timing)


def _coo_index_compatibility_error(exc):
    from ._spmv_csr_config import is_index_compatibility_error
    return is_index_compatibility_error(exc)
