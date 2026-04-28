"""Sparse triangular matrix-matrix solve (SpSM) for CSR/COO."""

from collections import OrderedDict

from ._common import *


SUPPORTED_SPSM_VALUE_DTYPES = (torch.float32, torch.float64)
SUPPORTED_SPSM_INDEX_DTYPES = (torch.int32, torch.int64)
SPSM_NON_TRANS_PRIMARY_COMBOS = (
    ("csr", torch.float32, torch.int32),
    ("csr", torch.float64, torch.int32),
    ("coo", torch.float32, torch.int32),
    ("coo", torch.float64, torch.int32),
)
_SPSM_PREPROCESS_CACHE = OrderedDict()
_SPSM_PREPROCESS_CACHE_SIZE = 8


def _clear_spsm_preprocess_cache():
    _SPSM_PREPROCESS_CACHE.clear()


def _is_non_transpose(op):
    op_str = str(op).upper()
    return op_str in ("NON_TRANS", "NON_TRANSPOSE", "N")


def _validate_spsm_non_trans_combo(fmt_name, value_dtype, index_dtype):
    combo = (str(fmt_name).lower(), value_dtype, index_dtype)
    if combo not in SPSM_NON_TRANS_PRIMARY_COMBOS:
        raise TypeError(
            f"{fmt_name} SpSM currently supports NON_TRANS combinations with int32 kernel indices: "
            "(float32, int32), (float64, int32)"
        )


def _validate_spsm_op_and_layout(opA, opB, major):
    if not _is_non_transpose(opA):
        raise NotImplementedError("Only op(A)=NON_TRANS is supported")
    if not _is_non_transpose(opB):
        raise NotImplementedError("Only op(B)=NON_TRANS is supported")
    if str(major).lower() != "row":
        raise NotImplementedError("Only row-major dense layout is supported")


def _prepare_spsm_csr_inputs(data, indices, indptr, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, B)):
        raise TypeError("data, indices, indptr, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, B)):
        raise ValueError("data, indices, indptr, B must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense matrix")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if B.shape[0] != n_rows:
        raise ValueError(f"B.shape[0] must equal n_rows={n_rows}")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32 or float64")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if indices.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    indices64 = indices.to(torch.int64).contiguous()
    indptr64 = indptr.to(torch.int64).contiguous()
    if data.numel() > 0:
        if bool(torch.any(indices64 < 0).item()):
            raise IndexError("indices must be non-negative")
        if int(indices64.max().item()) >= n_cols:
            raise IndexError("indices out of range")
        if int(indices64.max().item()) > _INDEX_LIMIT_INT32:
            raise ValueError("index value exceeds int32 kernel range")
    if indptr64.numel() > 0:
        if int(indptr64[0].item()) != 0:
            raise ValueError("indptr[0] must be 0")
        if int(indptr64[-1].item()) != int(data.numel()):
            raise ValueError("indptr[-1] must equal nnz")
        if bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
            raise ValueError("indptr must be non-decreasing")
    _validate_spsm_non_trans_combo("csr", data.dtype, torch.int32)
    return data.contiguous(), indices64, indptr64, B.contiguous(), n_rows, n_cols


def _prepare_spsm_coo_inputs(data, row, col, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, row, col, B)):
        raise TypeError("data, row, col, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, B must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have same length")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if B.ndim != 2 or B.shape[0] != n_rows:
        raise ValueError("B must be 2D and B.shape[0] == n_rows")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32 or float64")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSM_INDEX_DTYPES or col.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("row/col dtype must be torch.int32 or torch.int64")
    row64 = row.to(torch.int64).contiguous()
    col64 = col.to(torch.int64).contiguous()
    if row64.numel() > 0:
        if bool(torch.any(row64 < 0).item()) or bool(torch.any(col64 < 0).item()):
            raise IndexError("row/col must be non-negative")
        if int(row64.max().item()) >= n_rows:
            raise IndexError("row index out of range")
        if int(col64.max().item()) >= n_cols:
            raise IndexError("col index out of range")
        if max(int(row64.max().item()), int(col64.max().item())) > _INDEX_LIMIT_INT32:
            raise ValueError("row/col value exceeds int32 kernel range")
    _validate_spsm_non_trans_combo("coo", data.dtype, torch.int32)
    return data.contiguous(), row64, col64, B.contiguous(), n_rows, n_cols


def _tensor_cache_token(tensor):
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        int(tensor.numel()),
        storage_ptr,
        int(getattr(tensor, "_version", 0)),
    )


def _spsm_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsm_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _spsm_preprocess_cache_key(fmt_name, tensors, shape, lower, unit_diagonal):
    return (
        str(fmt_name).lower(),
        bool(lower),
        bool(unit_diagonal),
        int(shape[0]),
        int(shape[1]),
        *(_tensor_cache_token(t) for t in tensors),
    )


def _prepare_spsm_diag(data, indices64, indptr64, n_rows, unit_diagonal=False):
    diag = torch.ones(n_rows, dtype=data.dtype, device=data.device)
    if unit_diagonal or n_rows == 0 or data.numel() == 0:
        return diag
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    diag_mask = indices64 == row_ids
    if bool(torch.any(diag_mask).item()):
        diag.scatter_(0, row_ids[diag_mask], data[diag_mask])
    return diag


def _prepare_spsm_inv_diag(diag):
    if diag.numel() == 0:
        return diag
    eps = 1e-12 if diag.dtype == torch.float64 else 1e-6
    safe_diag = torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    return torch.reciprocal(safe_diag)


def _prepare_spsm_kernel_row_ptr(indptr64):
    if indptr64.numel() == 0:
        return indptr64.to(torch.int32)
    if int(indptr64[-1].item()) <= _INDEX_LIMIT_INT32:
        return indptr64.to(torch.int32)
    return indptr64


def _csr_row_ids_from_indptr(indptr64, n_rows):
    if n_rows == 0 or indptr64.numel() <= 1:
        return torch.empty(0, dtype=torch.int64, device=indptr64.device)
    return torch.repeat_interleave(
        torch.arange(n_rows, device=indptr64.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )


def _csr_rows_sorted_by_col(indices64, indptr64, n_rows):
    if indices64.numel() <= 1:
        return True
    row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    if row_ids.numel() <= 1:
        return True
    same_row = row_ids[1:] == row_ids[:-1]
    if not bool(torch.any(same_row).item()):
        return True
    return bool(torch.all(indices64[1:][same_row] >= indices64[:-1][same_row]).item())


def _prepare_spsm_csr_sorted_view(data, indices64, indptr64, n_rows, n_cols):
    row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    if data.numel() <= 1:
        return data, indices64, row_ids
    if _csr_rows_sorted_by_col(indices64, indptr64, n_rows):
        return data, indices64, row_ids
    key = row_ids * max(1, n_cols) + indices64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order].contiguous(), indices64[order].contiguous(), row_ids


def _prepare_spsm_csr_dependency_bounds(indices64, indptr64, n_rows, lower, row_ids=None):
    row_begin = indptr64[:-1].clone()
    row_end = indptr64[1:].clone()
    if n_rows == 0 or indices64.numel() == 0:
        return row_begin, row_begin if lower else row_end
    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    dep_counts = torch.bincount(row_ids[dep_mask], minlength=n_rows)
    if lower:
        dep_begin = row_begin
        dep_end = row_begin + dep_counts
    else:
        dep_begin = row_end - dep_counts
        dep_end = row_end
    return dep_begin, dep_end


def _prepare_spsm_csr_dependency_view(
    data, indices64, indptr64, n_rows, lower, row_ids=None
):
    dep_begin64, dep_end64 = _prepare_spsm_csr_dependency_bounds(
        indices64, indptr64, n_rows, lower=lower, row_ids=row_ids
    )
    dep_counts = dep_end64 - dep_begin64
    dep_ptr64 = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr64.device)
    if n_rows > 0:
        dep_ptr64[1:] = torch.cumsum(dep_counts, dim=0)
    total_dep_nnz = int(dep_ptr64[-1].item()) if dep_ptr64.numel() > 0 else 0
    if total_dep_nnz == 0:
        empty_data = torch.empty(0, dtype=data.dtype, device=data.device)
        empty_indices = torch.empty(0, dtype=torch.int64, device=indices64.device)
        return empty_data, empty_indices, dep_ptr64

    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    dep_data = data[dep_mask].contiguous()
    dep_indices64 = indices64[dep_mask].contiguous()
    return dep_data, dep_indices64, dep_ptr64


def _alpha_is_one(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            return False
        return bool((alpha.detach().cpu() == 1).item())
    return alpha == 1 or alpha == 1.0


def _alpha_to_host_scalar(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("alpha tensor must be scalar")
        return float(alpha.detach().cpu().item())
    return float(alpha)


@triton.jit
def _spsm_csr_diag_only_kernel_real(
    inv_diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_offsets < n_rhs

    b_ptrs = b_ptr + row * stride_b0 + rhs_offsets
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    if USE_FP64_ACC:
        rhs = rhs.to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
    else:
        rhs = rhs.to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)

    inv_diag = tl.load(inv_diag_ptr + row)
    inv_diag = inv_diag.to(tl.float64) if USE_FP64_ACC else inv_diag.to(tl.float32)
    out = rhs * alpha_val * inv_diag
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row * stride_x0 + rhs_offsets
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_offsets < n_rhs

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)

        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        x_ptrs = x_ptr + col[:, None] * stride_x0 + rhs_offsets[None, :]
        x_mask = nnz_mask[:, None] & rhs_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        if USE_FP64_ACC:
            x_vals = x_vals.to(tl.float64)
        else:
            x_vals = x_vals.to(tl.float32)

        contrib = tl.where(nnz_mask[:, None], a[:, None] * x_vals, 0.0)
        acc += tl.sum(contrib, axis=0)

    b_ptrs = b_ptr + row * stride_b0 + rhs_offsets
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    if USE_FP64_ACC:
        rhs = rhs.to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
    else:
        rhs = rhs.to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)

    inv_diag = tl.load(inv_diag_ptr + row)
    inv_diag = inv_diag.to(tl.float64) if USE_FP64_ACC else inv_diag.to(tl.float32)
    out = (rhs * alpha_val - acc) * inv_diag
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row * stride_x0 + rhs_offsets
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_single_segment_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_offsets < n_rhs

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    nnz_offsets = start + tl.arange(0, BLOCK_NNZ)
    nnz_mask = nnz_offsets < end

    a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
    col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)
    if USE_FP64_ACC:
        a = a.to(tl.float64)
    else:
        a = a.to(tl.float32)

    x_ptrs = x_ptr + col[:, None] * stride_x0 + rhs_offsets[None, :]
    x_mask = nnz_mask[:, None] & rhs_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
    if USE_FP64_ACC:
        x_vals = x_vals.to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
    else:
        x_vals = x_vals.to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)
    contrib = tl.where(nnz_mask[:, None], a[:, None] * x_vals, 0.0)
    acc = tl.sum(contrib, axis=0)

    b_ptrs = b_ptr + row * stride_b0 + rhs_offsets
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)

    inv_diag = tl.load(inv_diag_ptr + row)
    inv_diag = inv_diag.to(tl.float64) if USE_FP64_ACC else inv_diag.to(tl.float32)
    out = (rhs * alpha_val - acc) * inv_diag
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row * stride_x0 + rhs_offsets
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_staged_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ptr,
    work_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_work0,
    alpha,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    RHS_TILE_GROUPS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs_group = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_group_base = pid_rhs_group * BLOCK_RHS * RHS_TILE_GROUPS
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        inv_diag = tl.load(inv_diag_ptr + row).to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
        acc0 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc1 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc2 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc3 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        inv_diag = tl.load(inv_diag_ptr + row).to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)
        acc0 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)
        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        for tile_id in range(RHS_TILE_GROUPS):
            rhs_offsets = rhs_group_base + tile_id * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
            rhs_mask = rhs_offsets < n_rhs
            x_ptrs = work_ptr + col[:, None] * stride_work0 + rhs_offsets[None, :]
            x_mask = nnz_mask[:, None] & rhs_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
            if USE_FP64_ACC:
                x_vals = x_vals.to(tl.float64)
            else:
                x_vals = x_vals.to(tl.float32)
            contrib = tl.where(nnz_mask[:, None], a[:, None] * x_vals, 0.0)
            tile_acc = tl.sum(contrib, axis=0)
            if tile_id == 0:
                acc0 += tile_acc
            elif tile_id == 1:
                acc1 += tile_acc
            elif tile_id == 2:
                acc2 += tile_acc
            else:
                acc3 += tile_acc

    for tile_id in range(RHS_TILE_GROUPS):
        rhs_offsets = rhs_group_base + tile_id * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
        rhs_mask = rhs_offsets < n_rhs
        work_ptrs = work_ptr + row * stride_work0 + rhs_offsets
        rhs = tl.load(work_ptrs, mask=rhs_mask, other=0.0)
        rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)
        if tile_id == 0:
            acc_tile = acc0
        elif tile_id == 1:
            acc_tile = acc1
        elif tile_id == 2:
            acc_tile = acc2
        else:
            acc_tile = acc3
        out = (rhs * alpha_val - acc_tile) * inv_diag
        out = tl.where(out == out, out, 0.0)
        tl.store(work_ptrs, out, mask=rhs_mask)


def _build_spsm_levels(indptr, indices, n_rows, lower=True):
    if n_rows == 0:
        return []
    indptr_h = indptr.to(torch.int64).cpu()
    indices_h = indices.to(torch.int64).cpu()
    levels = [0] * n_rows

    if lower:
        for i in range(n_rows):
            start = int(indptr_h[i].item())
            end = int(indptr_h[i + 1].item())
            level = 0
            for p in range(start, end):
                col = int(indices_h[p].item())
                if col < i:
                    level = max(level, levels[col] + 1)
            levels[i] = level
    else:
        for i in range(n_rows - 1, -1, -1):
            start = int(indptr_h[i].item())
            end = int(indptr_h[i + 1].item())
            level = 0
            for p in range(start, end):
                col = int(indices_h[p].item())
                if col > i:
                    level = max(level, levels[col] + 1)
            levels[i] = level

    max_level = max(levels)
    buckets = [[] for _ in range(max_level + 1)]
    for row, level in enumerate(levels):
        buckets[level].append(row)

    device = indptr.device
    return [torch.tensor(rows, dtype=torch.int32, device=device) for rows in buckets if rows]


def _spsm_block_nnz_for_row_length(max_nnz_per_row):
    max_nnz_per_row = int(max_nnz_per_row)
    if max_nnz_per_row <= 32:
        return 32
    if max_nnz_per_row <= 64:
        return 64
    if max_nnz_per_row <= 128:
        return 128
    if max_nnz_per_row <= 256:
        return 256
    if max_nnz_per_row <= 512:
        return 512
    if max_nnz_per_row <= 1024:
        return 1024
    if max_nnz_per_row <= 4096:
        return 1024
    return 1024


def _spsm_single_segment_block_nnz(limit):
    return _spsm_block_nnz_for_row_length(limit)


def _bucketize_spsm_launch_groups(levels, indptr_like):
    if not levels:
        return []
    row_lengths = (indptr_like[1:] - indptr_like[:-1]).to(torch.int64)
    bucket_limits = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    grouped = []
    for rows_lv in levels:
        if rows_lv.numel() == 0:
            continue
        rows_i64 = rows_lv.to(torch.int64)
        lv_lengths = row_lengths.index_select(0, rows_i64)
        remaining = torch.ones(rows_lv.numel(), dtype=torch.bool, device=rows_lv.device)
        zero_mask = remaining & (lv_lengths == 0)
        if bool(torch.any(zero_mask).item()):
            grouped.append(
                {
                    "rows": rows_lv[zero_mask],
                    "diag_only": True,
                }
            )
            remaining = remaining & (~zero_mask)
        for limit in bucket_limits:
            mask = remaining & (lv_lengths <= limit)
            if bool(torch.any(mask).item()):
                bucket_rows = rows_lv[mask]
                # Keep single-segment buckets fully covered while staying on hardware-friendly powers of two.
                block_nnz = _spsm_single_segment_block_nnz(limit)
                grouped.append(
                    {
                        "rows": bucket_rows,
                        "block_nnz": block_nnz,
                        "max_segments": 1,
                        "diag_only": False,
                        "single_segment": True,
                        "staged": False,
                    }
                )
                remaining = remaining & (~mask)
        if bool(torch.any(remaining).item()):
            bucket_rows = rows_lv[remaining]
            bucket_lengths = lv_lengths[remaining]
            bucket_max = int(bucket_lengths.max().item()) if bucket_lengths.numel() > 0 else 0
            block_nnz = _spsm_block_nnz_for_row_length(bucket_max)
            max_segments = max((bucket_max + block_nnz - 1) // block_nnz, 1)
            grouped.append(
                {
                    "rows": bucket_rows,
                    "block_nnz": block_nnz,
                    "max_segments": max_segments,
                    "diag_only": False,
                    "single_segment": max_segments == 1,
                    "staged": bucket_max >= 256 or max_segments > 1,
                }
            )
    return grouped


def _auto_spsm_launch_config(indptr, block_nnz=None, max_segments=None):
    if indptr.numel() <= 1:
        max_nnz_per_row = 0
    else:
        row_lengths = indptr[1:] - indptr[:-1]
        max_nnz_per_row = int(row_lengths.max().item())

    auto_block = block_nnz is None
    if block_nnz is None:
        if max_nnz_per_row <= 64:
            block_nnz_use = 64
        elif max_nnz_per_row <= 256:
            block_nnz_use = 128
        elif max_nnz_per_row <= 1024:
            block_nnz_use = 256
        elif max_nnz_per_row <= 4096:
            block_nnz_use = 512
        else:
            block_nnz_use = 1024
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be positive")

    required_segments = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be positive")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )
    return block_nnz_use, max_segments_use


def _auto_rhs_block(n_rhs):
    n_rhs = int(n_rhs)
    if n_rhs <= 4:
        return 4
    if n_rhs <= 8:
        return 8
    if n_rhs <= 16:
        return 16
    if n_rhs <= 32:
        return 32
    return 64


def _choose_group_block_rhs(
    n_rhs,
    *,
    block_nnz,
    max_segments,
    diag_only,
    single_segment,
    value_dtype,
):
    base = _auto_rhs_block(n_rhs)
    if diag_only:
        return min(base, 32 if value_dtype == torch.float32 else 16)
    if block_nnz <= 64 and single_segment:
        return min(base, 32 if value_dtype == torch.float32 else 16)
    if block_nnz <= 128 and single_segment:
        return min(base, 16 if value_dtype == torch.float32 else 8)
    if max_segments > 2 or block_nnz >= 512:
        return min(base, 8)
    if max_segments > 1 or block_nnz >= 256:
        return min(base, 16 if value_dtype == torch.float32 else 8)
    return base


def _choose_group_rhs_tile_groups(n_rhs, block_rhs_use, staged):
    if not staged:
        return 1
    if block_rhs_use <= 0:
        return 1
    if n_rhs >= block_rhs_use * 4:
        return 4
    if n_rhs >= block_rhs_use * 2:
        return 2
    return 1


def _prepare_spsm_rhs_work_buffer(rhs):
    # Library-main solves through a dedicated RHS work buffer. In our current
    # row-major NON_TRANS path that buffer already matches the final layout, so
    # we keep a contiguous in-place work copy rather than doing an extra transpose.
    return rhs.contiguous().clone()


def _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols):
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device),
        )
    key = row64 * max(1, n_cols) + col64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_s = key[order]
    data_s = data[order]
    unique_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    out_nnz = unique_key.numel()
    data_u = torch.zeros(out_nnz, dtype=data.dtype, device=data.device)
    data_u.scatter_add_(0, inverse, data_s)
    row_u = torch.div(unique_key, max(1, n_cols), rounding_mode="floor")
    col_u = unique_key - row_u * max(1, n_cols)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    if out_nnz > 0:
        nnz_per_row = torch.bincount(row_u, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_u, col_u.to(torch.int64), indptr


def _prepare_spsm_csr_system(data, indices64, indptr64, n_rows, lower, unit_diagonal):
    sorted_data, sorted_indices64, row_ids = _prepare_spsm_csr_sorted_view(
        data, indices64, indptr64, n_rows, n_rows
    )
    dep_data, dep_indices64, dep_ptr64 = _prepare_spsm_csr_dependency_view(
        sorted_data, sorted_indices64, indptr64, n_rows, lower=lower, row_ids=row_ids
    )
    levels = _build_spsm_levels(dep_ptr64, dep_indices64, n_rows, lower=lower)
    launch_groups = _bucketize_spsm_launch_groups(levels, dep_ptr64)
    diag = _prepare_spsm_diag(
        sorted_data, sorted_indices64, indptr64, n_rows, unit_diagonal=unit_diagonal
    )
    default_block_nnz, default_max_segments = _auto_spsm_launch_config(dep_ptr64)
    return {
        "kernel_dep_data": dep_data,
        "kernel_dep_indices32": dep_indices64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(dep_ptr64),
        "kernel_inv_diag": _prepare_spsm_inv_diag(diag),
        "launch_groups": launch_groups,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "lower_eff": bool(lower),
    }


def _prepare_spsm_coo_system(data, row64, col64, n_rows, n_cols, lower, unit_diagonal):
    data_u, col_u64, row_ptr = _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols)
    dep_data, dep_indices64, dep_ptr64 = _prepare_spsm_csr_dependency_view(
        data_u, col_u64, row_ptr, n_rows, lower=lower
    )
    levels = _build_spsm_levels(dep_ptr64, dep_indices64, n_rows, lower=lower)
    launch_groups = _bucketize_spsm_launch_groups(levels, dep_ptr64)
    diag = _prepare_spsm_diag(
        data_u, col_u64, row_ptr, n_rows, unit_diagonal=unit_diagonal
    )
    default_block_nnz, default_max_segments = _auto_spsm_launch_config(dep_ptr64)
    return {
        "kernel_dep_data": dep_data,
        "kernel_dep_indices32": dep_indices64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(dep_ptr64),
        "kernel_inv_diag": _prepare_spsm_inv_diag(diag),
        "launch_groups": launch_groups,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "lower_eff": bool(lower),
    }


def _resolve_spsm_csr_runtime(data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major):
    data, indices64, indptr64, B, n_rows, n_cols = _prepare_spsm_csr_inputs(
        data, indices, indptr, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "csr",
        (data, indices64, indptr64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_csr_system(
            data, indices64, indptr64, n_rows, lower, unit_diagonal
        )
        _spsm_cache_put(_SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE)
    return data, B, n_rows, n_cols, solve_plan


def _resolve_spsm_coo_runtime(data, row, col, B, shape, lower, unit_diagonal, opA, opB, major):
    data, row64, col64, B, n_rows, n_cols = _prepare_spsm_coo_inputs(
        data, row, col, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "coo",
        (data, row64, col64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_coo_system(
            data, row64, col64, n_rows, n_cols, lower, unit_diagonal
        )
        _spsm_cache_put(_SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE)
    return data, B, n_rows, n_cols, solve_plan


def _run_spsm_csr_core(
    data,
    indices32,
    indptr,
    inv_diag,
    rhs,
    n_rows,
    *,
    alpha=1.0,
    lower=True,
    block_nnz=None,
    max_segments=None,
    block_rhs=None,
    launch_groups=None,
    block_nnz_use=None,
    max_segments_use=None,
):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    rhs = rhs.contiguous()
    if rhs.shape[0] != n_rows:
        raise ValueError("rhs first dim must equal n_rows")
    n_rhs = int(rhs.shape[1])
    rhs_work = _prepare_spsm_rhs_work_buffer(rhs)
    if n_rows == 0 or n_rhs == 0:
        return rhs_work

    if launch_groups is None:
        levels = _build_spsm_levels(indptr, indices32, n_rows, lower=lower)
        launch_groups = _bucketize_spsm_launch_groups(levels, indptr)
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsm_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    use_fp64 = data.dtype == torch.float64

    for group in launch_groups:
        if isinstance(group, dict):
            rows_lv = group["rows"]
            diag_only = bool(group.get("diag_only", False))
            single_segment = bool(group.get("single_segment", False))
            staged = bool(group.get("staged", False))
            block_nnz_lv = int(group.get("block_nnz", block_nnz_use))
            max_segments_lv = int(group.get("max_segments", max_segments_use))
        else:
            rows_lv = group
            diag_only = False
            single_segment = False
            staged = False
            block_nnz_lv = block_nnz_use
            max_segments_lv = max_segments_use
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        block_rhs_use = (
            int(block_rhs)
            if block_rhs is not None
            else _choose_group_block_rhs(
                n_rhs,
                block_nnz=block_nnz_lv,
                max_segments=max_segments_lv,
                diag_only=diag_only,
                single_segment=single_segment,
                value_dtype=data.dtype,
            )
        )
        if block_rhs_use <= 0:
            raise ValueError("block_rhs must be positive")
        rhs_tile_groups = _choose_group_rhs_tile_groups(n_rhs, block_rhs_use, staged)
        if diag_only:
            grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
            _spsm_csr_diag_only_kernel_real[grid](
                inv_diag,
                rhs_work,
                rhs_work,
                rows_lv,
                n_level_rows=n_lv,
                n_rhs=n_rhs,
                stride_b0=rhs_work.stride(0),
                stride_x0=rhs_work.stride(0),
                alpha=alpha,
                BLOCK_RHS=block_rhs_use,
                USE_FP64_ACC=use_fp64,
            )
            continue
        if single_segment:
            grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
            _spsm_csr_level_kernel_single_segment_real[grid](
                data,
                indices32,
                indptr,
                inv_diag,
                rhs_work,
                rhs_work,
                rows_lv,
                n_level_rows=n_lv,
                n_rhs=n_rhs,
                stride_b0=rhs_work.stride(0),
                stride_x0=rhs_work.stride(0),
                alpha=alpha,
                BLOCK_NNZ=block_nnz_lv,
                BLOCK_RHS=block_rhs_use,
                USE_FP64_ACC=use_fp64,
            )
            continue
        if staged:
            grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use * rhs_tile_groups))
            _spsm_csr_level_kernel_staged_real[grid](
                data,
                indices32,
                indptr,
                inv_diag,
                rhs_work,
                rows_lv,
                n_level_rows=n_lv,
                n_rhs=n_rhs,
                stride_work0=rhs_work.stride(0),
                alpha=alpha,
                BLOCK_NNZ=block_nnz_lv,
                BLOCK_RHS=block_rhs_use,
                RHS_TILE_GROUPS=rhs_tile_groups,
                MAX_SEGMENTS=max_segments_lv,
                USE_FP64_ACC=use_fp64,
            )
            continue
        grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
        _spsm_csr_level_kernel_real[grid](
            data,
            indices32,
            indptr,
            inv_diag,
            rhs_work,
            rhs_work,
            rows_lv,
            n_level_rows=n_lv,
            n_rhs=n_rhs,
            stride_b0=rhs_work.stride(0),
            stride_x0=rhs_work.stride(0),
            alpha=alpha,
            BLOCK_NNZ=block_nnz_lv,
            BLOCK_RHS=block_rhs_use,
            MAX_SEGMENTS=max_segments_lv,
            USE_FP64_ACC=use_fp64,
        )
    return rhs_work


def flagsparse_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_csr_runtime(
        data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major
    )
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    x = _run_spsm_csr_core(
        solve_plan["kernel_dep_data"],
        solve_plan["kernel_dep_indices32"],
        solve_plan["kernel_dep_ptr"],
        solve_plan["kernel_inv_diag"],
        B,
        n_rows,
        alpha=alpha_value,
        lower=solve_plan["lower_eff"],
        launch_groups=solve_plan["launch_groups"],
        block_nnz_use=solve_plan["default_block_nnz"],
        max_segments_use=solve_plan["default_max_segments"],
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out
    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_coo_runtime(
        data, row, col, B, shape, lower, unit_diagonal, opA, opB, major
    )
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    x = _run_spsm_csr_core(
        solve_plan["kernel_dep_data"],
        solve_plan["kernel_dep_indices32"],
        solve_plan["kernel_dep_ptr"],
        solve_plan["kernel_inv_diag"],
        B,
        n_rows,
        alpha=alpha_value,
        lower=solve_plan["lower_eff"],
        launch_groups=solve_plan["launch_groups"],
        block_nnz_use=solve_plan["default_block_nnz"],
        max_segments_use=solve_plan["default_max_segments"],
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out
    if return_time:
        return x, elapsed_ms
    return x


def _analyze_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_csr_runtime(data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def _analyze_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_coo_runtime(data, row, col, B, shape, lower, unit_diagonal, opA, opB, major)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def benchmark_spsm_case(
    fmt="csr",
    n_rows=1024,
    n_rhs=32,
    nnz=8192,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    warmup=10,
    iters=50,
):
    """Pure FlagSparse SpSM benchmark entry for one configuration."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_rows, nnz, value_dtype, index_dtype, device
    )
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    col_ids = indices.to(torch.int64)
    tri_mask = (col_ids <= row_ids) if lower else (col_ids >= row_ids)
    data = data[tri_mask]
    row_ids = row_ids[tri_mask]
    col_ids = col_ids[tri_mask]
    data, col_ids, indptr = _coo_to_csr_sorted_unique(data, row_ids, col_ids, n_rows, n_rows)
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    diag_mask = col_ids == row_ids
    diag_present = torch.zeros(n_rows, dtype=torch.bool, device=device)
    if diag_mask.numel() > 0 and bool(torch.any(diag_mask).item()):
        diag_present[row_ids[diag_mask]] = True
    missing_diag = torch.nonzero(~diag_present, as_tuple=False).reshape(-1).to(torch.int64)
    if missing_diag.numel() > 0:
        diag_values = torch.ones(missing_diag.numel(), dtype=value_dtype, device=device)
        data = torch.cat([data, diag_values], dim=0)
        row_ids = torch.cat([row_ids, missing_diag], dim=0)
        col_ids = torch.cat([col_ids, missing_diag], dim=0)
        data, col_ids, indptr = _coo_to_csr_sorted_unique(data, row_ids, col_ids, n_rows, n_rows)
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=device).contiguous()
    shape = (n_rows, n_rows)

    if str(fmt).lower() == "coo":
        row_ids = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        analysis_ms = _analyze_spsm_coo(
            data,
            row_ids,
            col_ids,
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_coo(
            data,
            row_ids,
            col_ids,
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    else:
        analysis_ms = _analyze_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr,
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr,
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    C_fs, solve_ms = _benchmark_cuda_op(solve_call, warmup=warmup, iters=iters)
    return {
        "parameters": {
            "format": str(fmt).lower(),
            "n_rows": n_rows,
            "n_rhs": n_rhs,
            "nnz": int(data.numel()),
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "opA": "NON_TRANS",
            "opB": "NON_TRANS",
            "major": "row",
        },
        "performance": {
            "triton_analysis_ms": analysis_ms,
            "triton_solve_ms": solve_ms,
            "triton_time_total_ms": (
                analysis_ms + solve_ms if analysis_ms is not None and solve_ms is not None else None
            ),
        },
        "samples": {"flagsparse": C_fs},
    }
