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

Storage: sorted ``data, row, col`` plus optional ``seg_starts`` vector. Prepared
gather routes additionally store row pointers; callers still provide COO arrays.
"""

from ._common import *
from . import _common as _common_mod
from . import mixed_spmx as _mixed_spmx

import time
from functools import lru_cache

import triton
import triton.language as tl

SUPPORTED_SPMV_COO_VALUE_DTYPES = (
    torch.float16,
    torch.bfloat16,
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
    return "COO SpMV supports float32, float64, complex64, and complex128"


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
        "original_inputs",
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
        "transpose_csr_plan",
        "non_csr_plan",
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
        self.transpose_csr_plan = None
        self.non_csr_plan = None
        if self.sort_by_row and self.data_trans.dtype in (
            torch.float32,
            torch.complex64,
        ):
            from . import _spmv_csr_transpose

            counts = torch.bincount(self.row_trans.long(), minlength=self.n_cols)
            ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(
                self.row_trans.dtype
            )
            self.transpose_csr_plan = _spmv_csr_transpose.make_plan(
                self.col_trans, ptr, None, self.n_cols
            )
            if (
                _backend_name() == "cuda" or _is_maca_runtime() or _is_rocm_runtime()
            ) and self.data_non.dtype == torch.complex64:
                counts = torch.bincount(self.row_non.long(), minlength=self.n_rows)
                ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(
                    self.row_non.dtype
                )
                self.non_csr_plan = _spmv_csr_transpose.make_plan(
                    self.col_non, ptr, None, self.n_rows
                )
                if _is_rocm_runtime():
                    # gfx936's wave64 complex gather is register-pressure
                    # bound at the default 8-lane row tile. Four lanes and a
                    # 128-row batch improve the C32 COO delivery corpus;
                    # keep this override local to ROCm and this route.
                    cols, plan_ptr, order, rows, _lanes, _batch, cache = (
                        self.non_csr_plan
                    )
                    self.non_csr_plan = (cols, plan_ptr, order, rows, 4, 128, cache)
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
        "csr_plan",
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
        csr_plan=None,
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
        self.csr_plan = csr_plan


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
    USE_MASKED_SELECT: tl.constexpr,
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
        # Masked loads already return zero, so the select only adds an extra
        # instruction to the FP32 reduction.
        acc += tl.sum(tl.where(m, v * xv, 0.0)) if USE_MASKED_SELECT else tl.sum(v * xv)
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
    USE_MASKED_SELECT: tl.constexpr,
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
        # Masked loads return zero for inactive lanes, so no select is needed
        # around the FP64 product before reduction.
        acc += tl.sum(tl.where(m, v * xv, 0.0)) if USE_MASKED_SELECT else tl.sum(v * xv)
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


@triton.jit
def _spmv_coo_run_add(row_a, val_a, row_b, val_b):
    # The flags represent segment starts, not row IDs. Propagating a start
    # makes this operation associative even for unsorted runs such as 0,1,0.
    return row_a | row_b, tl.where(row_b, val_b, val_a + val_b)


@triton.jit
def _spmv_coo_atomic_runs_real(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
    ACC: tl.constexpr,
):
    """Reduce adjacent COO entries before the global atomic update.

    This also handles unsorted COO: only adjacent equal rows are combined;
    disjoint runs still contribute through separate atomics.
    """
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    offs = pid * BLOCK + lane
    valid = offs < nnz
    row = tl.load(row_ptr + offs, valid, 0)
    col = tl.load(col_ptr + offs, valid, 0)
    value = tl.load(data_ptr + offs, valid, 0.0).to(ACC)
    vector = tl.load(x_ptr + col, valid, 0.0).to(ACC)
    previous = tl.load(row_ptr + offs - 1, valid & (lane > 0), -1)
    head = (lane == 0) | (row != previous) | ~valid
    _, run_sum = tl.associative_scan(
        (head, value * vector), axis=0, combine_fn=_spmv_coo_run_add
    )
    next_row = tl.load(row_ptr + offs + 1, valid & (offs + 1 < nnz), -1)
    boundary = valid & ((row != next_row) | (lane == BLOCK - 1))
    tl.atomic_add(y_ptr + row, run_sum, mask=boundary, sem="relaxed")


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


def _prepare_coo_tensors(data, row, col, shape, sort_by_row, *, value_dtypes=None):
    if not all(torch.is_tensor(t) for t in (data, row, col)):
        raise TypeError("data, row, col must all be torch.Tensor")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if not all(_is_accel_tensor(t) for t in (data, row, col)):
        raise ValueError("data, row, col must be CUDA tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have the same length")
    if data.dtype not in (SUPPORTED_SPMV_COO_VALUE_DTYPES if value_dtypes is None else value_dtypes):
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
    from .mixed_spmx import remember_registered_inputs
    original_inputs = (data, row, col)
    if alg is not None:
        selected = "coo_rowrun" if alg == "auto" else alg
        spec = get_spmv_coo_algorithm_spec(selected)
        if sort_by_row is not None and bool(sort_by_row) != (selected in ("coo_rowrun", "coo_rowrun_subgroup")):
            raise ValueError("sort_by_row conflicts with the selected algorithm")
        if data.dtype not in spec["supported_dtypes"]:
            raise TypeError(f"{selected} does not support {data.dtype}")
        code = _normalize_spmv_coo_op(op, transpose=transpose)
        if transpose and code == SPMV_COO_OP_NON:
            raise ValueError("transpose conflicts with op")
        if len(shape) != 2 or min(shape) < 0:
            raise ValueError("shape must contain two nonnegative dimensions")
        if row.device != data.device or col.device != data.device:
            raise ValueError("COO inputs must share a device")
        _prepare_coo_tensors(data, row, col, shape, False, value_dtypes=spec["supported_dtypes"])
        return remember_registered_inputs(PreparedCooRoute(data, row, col, shape, code, alg, config,
                               _normalize_spmv_coo_index_fallback_policy(index_fallback_policy)), original_inputs)
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
    return remember_registered_inputs(PreparedCoo(
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
    ), original_inputs)


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
            csr_plan=prepared.non_csr_plan,
        )
    data = prepared.data_trans
    # The csr_plan path (built whenever dtype qualifies, independent of backend --
    # see _PreparedCooLaunch.__init__) folds conj in internally, so the direct
    # conjugation below is normally skipped when a plan exists. Ascend's
    # _ascend_spmv_coo_index_add never touches csr_plan at all (plain index_add_
    # over launch.data/.row/.col), so it always needs the direct conj here
    # regardless of whether a plan was built -- same bug shape already fixed in
    # spmv_csr/spmm_csr/sddmm_csr's Ascend fallbacks.
    if (
        op_code == SPMV_COO_OP_CONJ_TRANS
        and _is_complex_dtype(data.dtype)
        and (prepared.transpose_csr_plan is None or _is_ascend_runtime())
    ):
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
        csr_plan=prepared.transpose_csr_plan,
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
    if prepared.csr_plan is not None:
        from . import _spmv_csr_transpose

        return _spmv_csr_transpose.gather(
            prepared.data,
            x,
            prepared.csr_plan,
            prepared.n_rows,
            prepared.op == SPMV_COO_OP_CONJ_TRANS,
        )
    dtype = prepared.data.dtype
    # A run-compressed launch stores exactly one value for every non-empty row.
    # When every row has a run, avoid a full output memset before the kernel;
    # retain zero initialization for genuinely empty rows.
    if (
        _is_rocm_runtime()
        and prepared.use_seg_kernel
        and prepared.n_segs == prepared.n_rows
    ):
        y = torch.empty(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    else:
        y = torch.zeros(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    nnz = prepared.nnz
    if nnz == 0:
        return y
    if _is_xpu_runtime() and not _is_complex_dtype(dtype):
        y.index_add_(0, prepared.row, prepared.data * x[prepared.col])
        return y
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.view_as_real(y).reshape(-1)
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
            USE_MASKED_SELECT=not _is_rocm_runtime(),
            num_warps=1,
        )
        return y
    ker = _spmv_coo_atomic_f64 if dtype == torch.float64 else _spmv_coo_atomic_f32
    grid = (triton.cdiv(nnz, block_size),)
    if _is_rocm_runtime():
        _spmv_coo_atomic_runs_real[grid](
            prepared.data,
            prepared.row,
            prepared.col,
            x,
            y,
            nnz,
            BLOCK=block_size,
            ACC=tl.float64 if dtype == torch.float64 else tl.float32,
            num_warps=num_warps,
        )
        return y
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
        if prepared.index_fallback_policy != "auto" or not _spmv_coo_uses_int64_indices(
            prepared
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


def _ascend_spmv_coo_index_add(launch, x):
    """Run COO SpMV through torch_npu primitives instead of Triton lowering."""
    y = torch.zeros(launch.n_rows, dtype=launch.data.dtype, device=launch.data.device)
    if launch.nnz:
        row = launch.row.to(torch.int64)
        col = launch.col.to(torch.int64)
        # conj (for op=conj_trans) is already folded into launch.data by
        # _resolve_spmv_coo_launch / _prepare_spmv_coo_launch.
        _index_add_values(y, 0, row, launch.data * _gather_values(x, col))
    return y


@lru_cache(maxsize=128)
def _cached_rocm_coo_launch(device, dtype, nnz, block_size, num_warps):
    # Device properties and launch clipping do not change between executions.
    # Cache only immutable launch values, never tensors or output buffers.
    launch = _spmv_rocm_launch_overrides(
        fmt="coo",
        dtype=dtype,
        nnz=nnz,
        block_size=block_size,
        num_warps=num_warps,
        device=device,
    )
    return int(launch["block_size"]), int(launch["num_warps"])


def _resolve_spmv_coo_kernel_launch(prepared, block_size, num_warps):
    # gfx936's atomic COO kernel is memory/atomic bound. A 256-element tile
    # with two waves outperformed the former 512-element tile on the delivery
    # matrices. Preserve explicit callers'
    # launch choices; this policy is only the public default.
    if block_size is None:
        block_size = 256 if _is_rocm_runtime() else 256
    if num_warps is None:
        num_warps = 2 if _is_rocm_runtime() else 4
    if _is_rocm_runtime():
        return _cached_rocm_coo_launch(
            prepared.data.device,
            prepared.data.dtype,
            prepared.nnz,
            int(block_size),
            int(num_warps),
        )
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
    block_size=None,
    num_warps=None,
    block_inner=128,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
    *,
    out_dtype=None,
    alg=None,
    config=None,
    return_meta=False,
    timing=False,
):
    """COO SpMV with no CSR indptr. See module docstring.

    ``block_inner``: tile for the row-run kernel (``sort_by_row=True``).
    ``block_size`` / ``num_warps``: grid over NNZ when ``sort_by_row=False`` (atomics).
    float16 / bfloat16 / int8 matrices and widened outputs (``out_dtype`` or
    ``out.dtype``: half -> float32, int8 -> int32 / float32) run in ``mixed_spmx``.
    """
    from .mixed_spmx import validate_registered_output_alias
    validate_registered_output_alias(prepared, x, out, (data, row, col))
    if alg is not None or config is not None or isinstance(prepared, PreparedCooRoute):
        if block_size is not None or num_warps is not None or block_inner != 128:
            raise ValueError("use config for registered COO launch parameters")
        if transpose is not None and op is not None and bool(transpose) != _spmv_coo_op_transposes(_normalize_spmv_coo_op(op)):
            raise ValueError("transpose conflicts with op")
        if prepared is None:
            prepared = prepare_spmv_coo(data, row, col, shape, op=op,
                transpose=bool(transpose), sort_by_row=sort_by_row, alg="auto" if alg is None else alg,
                config=config, index_fallback_policy=index_fallback_policy)
        elif shape is not None and tuple(shape) != prepared.shape:
            raise ValueError("shape conflicts with prepared COO")
        if sort_by_row is not None:
            selected = alg or prepared.alg
            selected = "coo_rowrun" if selected == "auto" else selected
            if bool(sort_by_row) != (selected in ("coo_rowrun", "coo_rowrun_subgroup")):
                raise ValueError("sort_by_row conflicts with the selected algorithm")
        requested_op = op if op is not None else ("trans" if transpose else "non") if transpose is not None else None
        return flagsparse_spmv_coo_run(prepared, x, alg=alg, config=config, op=requested_op,
            out=out, out_dtype=out_dtype, return_time=return_time, return_meta=return_meta, timing=timing)
    if return_meta or timing:
        raise ValueError("legacy COO diagnostics use the existing benchmark interface; select alg for registered diagnostics")
    if (
        prepared is None
        and torch.is_tensor(data)
        and _mixed_spmx.spmv_needs_mixed(
            data.dtype,
            x.dtype if torch.is_tensor(x) else None,
            out,
            out_dtype,
            coo=True,
        )
    ):
        return _mixed_spmx.spmv_coo_mixed(
            data,
            row,
            col,
            x,
            shape,
            op=op,
            transpose=transpose,
            out=out,
            out_dtype=out_dtype,
            return_time=return_time,
        )
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
    output_dtype = _mixed_spmx.resolve_registered_types(
        launch.data.dtype, x.dtype, out=out, out_dtype=out_dtype
    )["output_dtype"]
    if out is not None:
        if out.shape != (launch.n_rows,) or out.dtype != output_dtype or out.device != launch.data.device:
            raise ValueError("out shape/dtype/device mismatch")
        if any(torch._C._overlaps(out, value) for value in
               (prepared.data_non, prepared.data_trans, prepared.row_non, prepared.col_non, x)):
            raise ValueError("out overlaps an input")
        from ._spmm_csr_runtime import output_has_internal_overlap
        if output_has_internal_overlap(out):
            raise ValueError("out must have non-overlapping storage")
    x = _validate_x_coo(x, launch)
    if num_warps is not None and num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a power of 2 in [1, 32]")
    if block_inner <= 0 or (block_inner & (block_inner - 1)) != 0:
        raise ValueError("block_inner must be a positive power of 2")
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if launch.data.dtype in (torch.float16, torch.bfloat16):
        from copy import copy

        launch = copy(launch)
        launch.data = launch.data.float()
        x = x.float()
    if _is_ascend_runtime():
        y = _ascend_spmv_coo_index_add(launch, x)
    else:
        # The gather kernel owns its R/K/warp geometry. COO atomic launch
        # overrides are unused on this route, so avoid resolving them per call.
        if _is_rocm_runtime() and launch.csr_plan is not None:
            block_size = 256 if block_size is None else block_size
            num_warps = 2 if num_warps is None else num_warps
        else:
            block_size, num_warps = _resolve_spmv_coo_kernel_launch(
                launch, block_size, num_warps
            )
        y = _run_spmv_coo_prepared_with_fallback(
            launch,
            x,
            block_size=block_size,
            num_warps=num_warps,
            block_inner=block_inner,
        )
    y = y.to(output_dtype)
    if out is not None:
        out.copy_(y)
        y = out
    elapsed_ms = None
    if return_time:
        _ACCEL.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
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
        return a * b, tl.full(mask.shape, 0, acc_type)


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
def _coo_stream_panel_kernel(A, Row, Col, B, C, NNZ, N,
                             bs0, bs1, cs0, cs1,
                             T: tl.constexpr, BN: tl.constexpr, BLOCKS: tl.constexpr,
                             COMPLEX: tl.constexpr, CONJ: tl.constexpr,
                             FP64: tl.constexpr, SEGMENT: tl.constexpr):
    lane = tl.arange(0, T)
    n = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    base = tl.program_id(0).to(tl.int64) * T * BLOCKS
    acc_type = tl.float64 if FP64 else tl.float32
    # Carry has the reduction's slice layout from its first definition onward.
    # Static unrolling avoids dynamic loop/branch phi layout conversion on HIP.
    carry_r = tl.sum(tl.zeros((T, BN), acc_type), 0)
    carry_i = tl.sum(tl.zeros((T, BN), acc_type), 0)
    carry_row = tl.full((), -1, tl.int64)
    for block in tl.static_range(BLOCKS):
        first = base + block * T
        k = first + lane
        row = tl.load(Row + k, k < NNZ, -1).to(tl.int64)
        col = tl.load(Col + k, k < NNZ, 0).to(tl.int64)
        valid = (k[:, None] < NNZ) & (n[None, :] < N)
        real, imag = _coo_product(A, B, k[:, None], col[:, None], n[None, :],
                                 bs0, bs1, valid, COMPLEX, CONJ, FP64)
        if SEGMENT:
            first_row = tl.load(Row + first, first < NNZ, -1).to(tl.int64)
            same = carry_row == first_row
            flush = (first < NNZ) & (carry_row >= 0) & ~same & (n < N)
            dst = carry_row * cs0 + n * cs1
            if COMPLEX:
                tl.atomic_add(C + 2 * dst, carry_r, flush, sem="relaxed")
                tl.atomic_add(C + 2 * dst + 1, carry_i, flush, sem="relaxed")
            else:
                tl.atomic_add(C + dst, carry_r, flush, sem="relaxed")
            real += tl.where((lane[:, None] == 0) & same, carry_r[None, :], 0)
            if COMPLEX:
                imag += tl.where((lane[:, None] == 0) & same, carry_i[None, :], 0)
            prev = tl.load(Row + k - 1, (lane > 0) & (k < NNZ), -1)
            flags = tl.broadcast_to(((lane == 0) | (row != prev))[:, None], (T, BN))
            real, _ = tl.associative_scan((real, flags), 0, _coo_scan_pair)
            if COMPLEX:
                imag, _ = tl.associative_scan((imag, flags), 0, _coo_scan_pair)
            last_lane = tl.maximum(0, tl.minimum(T, NNZ - first) - 1)
            nxt = tl.load(Row + k + 1, lane < last_lane, -1)
            valid = valid & ((row != nxt) & (lane < last_lane))[:, None]
            carry_row = tl.where(first < NNZ,
                tl.load(Row + first + last_lane, first < NNZ, -1).to(tl.int64), carry_row)
            carry_r = tl.where(first < NNZ,
                tl.sum(tl.where(lane[:, None] == last_lane, real, 0), 0), carry_r)
            if COMPLEX:
                carry_i = tl.where(first < NNZ,
                    tl.sum(tl.where(lane[:, None] == last_lane, imag, 0), 0), carry_i)
        dst = row[:, None] * cs0 + n[None, :] * cs1
        if COMPLEX:
            tl.atomic_add(C + 2 * dst, real, valid, sem="relaxed")
            tl.atomic_add(C + 2 * dst + 1, imag, valid, sem="relaxed")
        else:
            tl.atomic_add(C + dst, real, valid, sem="relaxed")
    if SEGMENT:
        dst = carry_row * cs0 + n * cs1
        valid = (carry_row >= 0) & (n < N)
        if COMPLEX:
            tl.atomic_add(C + 2 * dst, carry_r, valid, sem="relaxed")
            tl.atomic_add(C + 2 * dst + 1, carry_i, valid, sem="relaxed")
        else:
            tl.atomic_add(C + dst, carry_r, valid, sem="relaxed")


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
    "coo_segmented_stream_atomic", "coo_segmented_stream_panel_atomic",
)
_COO_ATOMIC_ALGORITHMS = ("coo_atomic", "coo_segmented_atomic", "coo_segmented_panel_atomic",
                          "coo_segmented_stream_atomic", "coo_segmented_stream_panel_atomic")
SPMV_COO_ALGORITHMS = ("coo_rowrun", "coo_atomic", "coo_segmented_atomic",
                       "coo_rowrun_subgroup", "coo_segmented_stream_atomic")


def _resolve_coo_config(alg, dtype, device, config=None, n=1):
    from ._spmm_csr_runtime import backend_caps
    from ._spmx_column_config import resolve_coo_config
    return resolve_coo_config(alg, str(dtype).removeprefix("torch."), backend_caps(device), config, n)


def _coo_sorted_runs(data, row, col):
    order = torch.argsort(row, stable=True)
    data, row, col = data[order], row[order], col[order]
    starts = _seg_starts_from_sorted_rows(row, row.numel(), row.device)
    return data, row, col, starts


def _launch_coo_extension(data, row, col, starts, B, shape, alg, cfg, conj=False, compute_dtype=None):
    # B is always a 2D view; SpMV uses a singleton output panel.
    compute_dtype = data.dtype if compute_dtype is None else compute_dtype
    C = torch.zeros((shape[0], B.shape[1]), device=data.device, dtype=compute_dtype)
    if not data.numel() or not C.numel():
        return C
    complex_ = data.is_complex()
    a = torch.view_as_real(data).reshape(-1) if complex_ else data
    b = torch.view_as_real(B) if complex_ else B
    c = torch.view_as_real(C) if complex_ else C
    args = (a, row, col)
    common = dict(COMPLEX=complex_, CONJ=conj,
                  FP64=compute_dtype in (torch.float64, torch.complex128),
                  num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if alg == "coo_segmented_atomic" and not complex_ and cfg["local_reduce"] == "segment":
        _spmv_coo_atomic_runs_real[(triton.cdiv(data.numel(), cfg["block_nnz"]),)](
            data, row, col, B[:, 0].contiguous(), C[:, 0], data.numel(),
            BLOCK=cfg["block_nnz"], ACC=tl.float64 if compute_dtype == torch.float64 else tl.float32,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )
        return C
    if alg in _COO_ATOMIC_ALGORITHMS:
        bn = cfg.get("block_n", 1)
        stream = alg in ("coo_segmented_stream_atomic", "coo_segmented_stream_panel_atomic")
        kernel = _coo_stream_panel_kernel if stream else _coo_segmented_panel_kernel
        extra = {"BLOCKS": cfg["blocks_per_program"]} if stream else {}
        kernel[(triton.cdiv(data.numel(), cfg["block_nnz"] * cfg.get("blocks_per_program", 1)),
                                     triton.cdiv(B.shape[1], bn))](
            *args, b, c, data.numel(), B.shape[1], *B.stride(), *C.stride(),
            T=cfg["block_nnz"], BN=bn, SEGMENT=cfg["local_reduce"] == "segment", **extra, **common)
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
                supported_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128),
                half_output_dtypes={dtype: (dtype, torch.float32)
                                    for dtype in (torch.float16, torch.bfloat16)},
                implementation_version=3, timing_contract_version=2,
                requires_sort=alg not in _COO_ATOMIC_ALGORITHMS)


def list_spmv_coo_algorithms(op=None, dtype=None, backend=None, *, rhs_dtype=None, out_dtype=None):
    if rhs_dtype is not None or out_dtype is not None:
        if dtype is None:
            raise ValueError("dtype is required for type-combination filtering")
        try:
            _mixed_spmx.resolve_registered_types(
                dtype, dtype if rhs_dtype is None else rhs_dtype, out_dtype=out_dtype)
        except TypeError:
            return ()
    if op is not None:
        _normalize_spmv_coo_op(op)
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16, *SUPPORTED_SPMV_COO_VALUE_DTYPES):
        return ()
    # Runtime capabilities are resolved against the actual device at run time.
    if backend is not None and backend not in tuple(spec.name for spec in _common_mod.backend_specs()):
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
                            out=None, out_dtype=None, return_time=False, return_meta=False, timing=False):
    from .mixed_spmx import validate_registered_output_alias
    validate_registered_output_alias(prepared, x, out)
    if not isinstance(prepared, PreparedCooRoute):
        if alg is not None or config is not None:
            raise ValueError("prepare with alg before switching a legacy COO prepared route")
        if return_meta or timing:
            raise ValueError("legacy COO prepared diagnostics use the existing benchmark interface")
        return flagsparse_spmv_coo(prepared=prepared, x=x, op=op, out=out,
                                   out_dtype=out_dtype, return_time=return_time)
    if op is not None and _spmv_coo_op_to_name(op) != prepared.op:
        raise ValueError("op conflicts with prepared COO")
    requested = prepared.alg if alg is None else alg
    selected = "coo_rowrun" if requested == "auto" else requested
    spec = get_spmv_coo_algorithm_spec(selected)
    if prepared.data.dtype not in spec["supported_dtypes"]:
        raise TypeError(f"{selected} does not support {prepared.data.dtype}")
    cfg_input = config if config is not None else (prepared.config if requested == prepared.alg else {})
    data, row, col = prepared.data, prepared.row, prepared.col
    trans = prepared.op != "non"
    shape = prepared.shape[::-1] if trans else prepared.shape
    if not torch.is_tensor(x):
        raise TypeError("x must be a tensor")
    if x.ndim != 1 or x.numel() != shape[1] or x.dtype != data.dtype or x.device != data.device:
        raise ValueError("x shape/dtype/device does not match COO operation")
    output_dtype = _mixed_spmx.resolve_registered_types(
        data.dtype, x.dtype, out=out, out_dtype=out_dtype
    )["output_dtype"]
    if out is not None:
        if out.shape != (shape[0],) or out.dtype != output_dtype or out.device != data.device:
            raise ValueError("out shape/dtype/device does not match COO operation")
        if any(torch._C._overlaps(out, t) for t in (data, row, col, x)):
            raise ValueError("out overlaps an input")
        from ._spmm_csr_runtime import output_has_internal_overlap
        if output_has_internal_overlap(out):
            raise ValueError("out must have non-overlapping storage")
    cfg, info = ({}, {})
    if selected in _COO_NEW_ALGORITHMS:
        cfg, info = _resolve_coo_config(selected, data.dtype, data.device, cfg_input)
    elif cfg_input:
        raise ValueError("legacy COO algorithms do not accept config")
    def execute(phases):
        a, r, c = data, col if trans else row, row if trans else col
        starts = None
        with phases.measure("process_gpu_ms"):
            a, r, c = a.resolve_conj().resolve_neg().contiguous(), r.contiguous(), c.contiguous()
            vector = x.resolve_conj().resolve_neg()
            if data.dtype in (torch.float16, torch.bfloat16):
                a, vector = a.float(), x.float()
            if selected not in _COO_ATOMIC_ALGORITHMS:
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
            y = y.to(output_dtype)
            if out is not None:
                out.copy_(y)
                y = out
        return y, dict(info, alg=selected, alg_requested=requested, alg_resolved=selected,
                       a_dtype=str(data.dtype), rhs_dtype=str(x.dtype), output_dtype=str(output_dtype),
                       op=prepared.op, compute_dtype=str(torch.float32 if data.dtype in (torch.float16, torch.bfloat16) else data.dtype),
                       component_dtype="float64" if data.dtype in (torch.float64, torch.complex128) else "float32",
                       backend=_backend_name(),
                       input_indices=(str(row.dtype), str(col.dtype)),
                       execution_indices=(str(r.dtype), str(c.dtype)), fallback_reason=reason,
                       transpose_strategy="index_roles", implementation_version=2,
                       blocks_per_program=cfg.get("blocks_per_program", 1),
                       timing_contract_version=2)
    return _coo_measure_run(execute, return_time, return_meta, timing)


def _coo_index_compatibility_error(exc):
    from ._spmv_csr_config import is_index_compatibility_error
    return is_index_compatibility_error(exc)
