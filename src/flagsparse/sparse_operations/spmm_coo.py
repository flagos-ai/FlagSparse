"""Native COO SpMM kernels, route helpers, and internal benchmark entry points."""

import ctypes

from . import _common as _common_mod
from ._common import *
from .spmm_csr import (
    SUPPORTED_SPMM_VALUE_DTYPES,
    _select_spmm_alg1_warp_and_factor,
    _spmm_coo_reference_tolerance,
    _spmm_relative_threshold,
    _spmm_validation_metrics,
)

hip = _common_mod.hip
hipsparse = _common_mod.hipsparse
HipPointer = _common_mod.HipPointer
_hip_check_result = _common_mod._hip_check_result
_hipsparse_unavailable_reason = _common_mod._hipsparse_unavailable_reason
_hipsparse_value_type = _common_mod._hipsparse_value_type
_hipsparse_scalar = _common_mod._hipsparse_scalar
_hipsparse_index_type = _common_mod._hipsparse_index_type
_hipsparse_lookup = _common_mod._hipsparse_lookup
_hipsparse_create_coo_descriptor = _common_mod._hipsparse_create_coo_descriptor
_hipsparse_create_dnmat_descriptor = _common_mod._hipsparse_create_dnmat_descriptor
def _spmm_coo_compute_dtype(value_dtype):
    if _is_complex_dtype(value_dtype):
        return torch.complex128 if value_dtype == torch.complex64 else value_dtype
    if value_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if value_dtype == torch.float32:
        return torch.float64
    return value_dtype


def _sort_coo_lex_inplace(data, row, col, n_cols):
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    if data.numel() == 0:
        return data.contiguous(), row64, col64
    key = row64 * max(1, int(n_cols)) + col64
    order = torch.argsort(key)
    return (
        data[order].contiguous(),
        row64[order].contiguous(),
        col64[order].contiguous(),
    )


def _coalesce_coo_entries(data, row, col, shape):
    """Merge duplicate (row, col) by summing values (PyTorch COO coalesce)."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        z = torch.empty(0, dtype=torch.int64, device=data.device)
        return data.contiguous(), z, z.clone()
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    coo = torch.sparse_coo_tensor(
        torch.stack([row64, col64]),
        data,
        size=(n_rows, n_cols),
        device=data.device,
        dtype=data.dtype,
    ).coalesce()
    idx = coo.indices()
    return coo.values().contiguous(), idx[0].contiguous(), idx[1].contiguous()


def _build_torch_sparse_coo(data, row, col, shape):
    """Coalesced CUDA COO tensor for ``torch.sparse.mm`` (indices int64)."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        empty_idx = torch.empty((2, 0), dtype=torch.int64, device=data.device)
        return torch.sparse_coo_tensor(
            empty_idx,
            data,
            size=(n_rows, n_cols),
            device=data.device,
            dtype=data.dtype,
        )
    row_i = row.to(torch.int64)
    col_i = col.to(torch.int64)
    indices = torch.stack([row_i, col_i])
    return torch.sparse_coo_tensor(
        indices,
        data,
        size=(n_rows, n_cols),
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _build_random_coo(n_rows, n_cols, nnz, value_dtype, index_dtype, device):
    nnz = int(nnz)
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if int(n_rows) < 0 or int(n_cols) < 0:
        raise ValueError("matrix dimensions must be non-negative")
    if index_dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("index_dtype must be torch.int32 or torch.int64")
    if value_dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError("value_dtype is not supported by COO SpMM")

    data = _build_random_dense(nnz, value_dtype, device)
    if nnz == 0:
        row = torch.empty((0,), dtype=index_dtype, device=device)
        col = torch.empty((0,), dtype=index_dtype, device=device)
        return data, row, col
    if n_rows == 0 or n_cols == 0:
        raise ValueError("nnz must be 0 when either matrix dimension is zero")
    row = torch.randint(0, int(n_rows), (nnz,), dtype=index_dtype, device=device)
    col = torch.randint(0, int(n_cols), (nnz,), dtype=index_dtype, device=device)
    return data, row, col


def _prepare_spmm_coo_canonical_prepared(data, row, col, B, n_rows, n_cols, n_dense_cols):
    output_dtype = data.dtype
    compute_dtype = _spmm_coo_compute_dtype(output_dtype)
    data_compute = data if compute_dtype == output_dtype else data.to(compute_dtype)
    B_compute = B if compute_dtype == output_dtype else B.to(compute_dtype)
    canonical_data, canonical_row, canonical_col = _coalesce_coo_entries(
        data_compute,
        row,
        col,
        (n_rows, n_cols),
    )
    canonical_data, canonical_row, canonical_col = _sort_coo_lex_inplace(
        canonical_data,
        canonical_row,
        canonical_col,
        n_cols,
    )
    return (
        canonical_data,
        canonical_row,
        canonical_col,
        B_compute,
        n_rows,
        n_cols,
        n_dense_cols,
        output_dtype,
        compute_dtype,
    )


def _prepare_spmm_coo_canonical_inputs(data, row, col, B, shape):
    data, kernel_row, kernel_col, B, n_rows, n_cols, n_dense_cols = _prepare_spmm_coo_inputs(
        data, row, col, B, shape
    )
    return _prepare_spmm_coo_canonical_prepared(
        data,
        kernel_row,
        kernel_col,
        B,
        n_rows,
        n_cols,
        n_dense_cols,
    )


def _spmm_coo_sparse_ref_backend(value_dtype, index_dtype):
    if _is_rocm_runtime():
        reason = _hipsparse_spmm_coo_skip_reason(value_dtype, index_dtype)
        if reason is None:
            return "hipsparse", None
        return None, reason
    if cp is None or cpx_sparse is None:
        return None, "CuPy/cuSPARSE is not available"
    if value_dtype in (torch.float16, torch.bfloat16):
        return None, "float16/bfloat16 not supported by CuPy sparse; skipped"
    return "cupy_cusparse", None


def _hipsparse_spmm_coo_skip_reason(value_dtype, index_dtype):
    if not _is_rocm_runtime():
        return "hipSPARSE COO SpMM reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCoo",
        "hipsparseCreateDnMat",
        "hipsparseDestroyDnMat",
        "hipsparseDestroySpMat",
        "hipsparseSpMM_bufferSize",
        "hipsparseSpMM_preprocess",
        "hipsparseSpMM",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE COO SpMM direct API is unavailable: missing {symbol}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_scalar(value_dtype, 0.0, 0.0)
        _ = _hipsparse_index_type(index_dtype, "hipSPARSE COO SpMM indices")
        _ = _common_mod._hipsparse_spmm_order("row", "hipSPARSE COO SpMM")
        _ = _common_mod._hipsparse_spmm_algorithm("coo")
        _ = _hipsparse_lookup(
            "hipsparseOperation_t",
            ("HIPSPARSE_OPERATION_NON_TRANSPOSE",),
        )
    except Exception as exc:
        return str(exc)
    return None


def _prepare_spmm_coo_ref_hipsparse(
    data,
    row,
    col,
    B,
    shape,
    out=None,
):
    skip_reason = _hipsparse_spmm_coo_skip_reason(data.dtype, row.dtype)
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(torch.is_tensor(t) for t in (data, row, col, B)):
        raise TypeError("data, row, col, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, B must all be CUDA tensors")
    if not all(t.device == data.device for t in (row, col, B)):
        raise ValueError("data, row, col, B must be on the same CUDA device")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must all be 1D tensors")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have the same length")
    if B.ndim != 2:
        raise ValueError("hipSPARSE COO SpMM reference expects a 2D dense RHS")

    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if int(B.shape[0]) != n_cols:
        raise ValueError(f"B.shape[0] must equal n_cols={n_cols}")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match sparse value dtype for direct hipSPARSE COO SpMM")
    if row.dtype != col.dtype:
        raise TypeError("row and col must use the same index dtype for direct hipSPARSE COO SpMM")
    if not B.is_contiguous():
        raise ValueError("hipSPARSE COO SpMM direct reference expects contiguous row-major B")

    n_dense_cols = int(B.shape[1])
    if n_dense_cols == 0:
        return {
            "backend": "hipsparse",
            "buffer_size": 0,
            "format": "coo",
            "C": torch.empty((n_rows, 0), dtype=data.dtype, device=data.device),
            "empty": True,
        }

    data = data.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    B = B.contiguous()
    value_type = _hipsparse_value_type(data.dtype)
    alpha = _hipsparse_scalar(data.dtype, 1.0, 0.0)
    beta = _hipsparse_scalar(data.dtype, 0.0, 0.0)
    index_type = _hipsparse_index_type(row.dtype, "hipSPARSE COO SpMM indices")
    op_enum = _hipsparse_lookup(
        "hipsparseOperation_t",
        ("HIPSPARSE_OPERATION_NON_TRANSPOSE",),
    )
    order = _common_mod._hipsparse_spmm_order("row", "hipSPARSE COO SpMM")
    alg = _common_mod._hipsparse_spmm_algorithm("coo")

    C = out
    if C is None:
        C = torch.empty((n_rows, n_dense_cols), dtype=data.dtype, device=data.device)
    else:
        if not torch.is_tensor(C):
            raise TypeError("out must be a torch.Tensor")
        if not C.is_cuda or C.device != data.device:
            raise ValueError("out must be a CUDA tensor on the same device as data")
        if C.dtype != data.dtype or C.shape != (n_rows, n_dense_cols):
            raise ValueError("out must match the result shape and dtype")
        if not C.is_contiguous():
            raise ValueError("out must be contiguous row-major")

    handle = None
    spmat = None
    matb = None
    matc = None
    workspace = 0
    workspace_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)

        spmat = ptr_type()
        matb = ptr_type()
        matc = ptr_type()
        spmat_ref = spmat.createRef()
        matb_ref = matb.createRef()
        matc_ref = matc.createRef()

        row_ptr = HipPointer.fromObj(row.data_ptr())
        col_ptr = HipPointer.fromObj(col.data_ptr())
        values_ptr = HipPointer.fromObj(data.data_ptr())
        b_ptr = HipPointer.fromObj(B.data_ptr())
        c_ptr = HipPointer.fromObj(C.data_ptr())

        index_base = _hipsparse_lookup(
            "hipsparseIndexBase_t",
            ("HIPSPARSE_INDEX_BASE_ZERO",),
        )

        _hipsparse_create_coo_descriptor(
            spmat_ref,
            n_rows,
            n_cols,
            int(data.numel()),
            row_ptr,
            col_ptr,
            values_ptr,
            index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_dnmat_descriptor(
            matb_ref,
            n_cols,
            n_dense_cols,
            int(B.stride(0)),
            b_ptr,
            value_type,
            order,
        )
        _hipsparse_create_dnmat_descriptor(
            matc_ref,
            n_rows,
            n_dense_cols,
            int(C.stride(0)),
            c_ptr,
            value_type,
            order,
        )

        size_out = ctypes.c_size_t()
        _hip_check_result(
            hipsparse.hipsparseSpMM_bufferSize(
                handle,
                op_enum,
                op_enum,
                alpha,
                spmat,
                matb,
                beta,
                matc,
                value_type,
                alg,
                size_out,
            ),
            "hipsparseSpMM_bufferSize",
        )
        buffer_size = int(size_out.value)
        if buffer_size > 0:
            workspace = _hip_check_result(hip.hipMalloc(buffer_size), "hipMalloc")
            workspace_allocated = True
        else:
            workspace = 0
        _hip_check_result(
            hipsparse.hipsparseSpMM_preprocess(
                handle,
                op_enum,
                op_enum,
                alpha,
                spmat,
                matb,
                beta,
                matc,
                value_type,
                alg,
                workspace,
            ),
            "hipsparseSpMM_preprocess",
        )
        return {
            "backend": "hipsparse",
            "buffer_size": buffer_size,
            "format": "coo",
            "handle": handle,
            "spmat": spmat,
            "matb": matb,
            "matc": matc,
            "workspace": workspace,
            "workspace_allocated": workspace_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "beta": beta,
            "value_type": value_type,
            "alg": alg,
            "C": C,
            "empty": False,
        }
    finally:
        if handle is None and matc is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnMat(matc), "hipsparseDestroyDnMat(C)"
                )
            except Exception:
                pass
        if handle is None and matb is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnMat(matb), "hipsparseDestroyDnMat(B)"
                )
            except Exception:
                pass
        if handle is None and spmat is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
                )
            except Exception:
                pass
        if handle is None and workspace_allocated:
            try:
                _hip_check_result(hip.hipFree(workspace), "hipFree")
            except Exception:
                pass
 

def _run_spmm_coo_ref_hipsparse_prepared(state):
    if state.get("empty"):
        return state["C"]
    _hip_check_result(
        hipsparse.hipsparseSpMM(
            state["handle"],
            state["op_enum"],
            state["op_enum"],
            state["alpha"],
            state["spmat"],
            state["matb"],
            state["beta"],
            state["matc"],
            state["value_type"],
            state["alg"],
            state["workspace"],
        ),
        "hipsparseSpMM",
    )
    return state["C"]


def _destroy_spmm_coo_ref_hipsparse_prepared(state):
    matc = state.get("matc")
    matb = state.get("matb")
    spmat = state.get("spmat")
    workspace_allocated = bool(state.get("workspace_allocated"))
    workspace = state.get("workspace", 0)
    handle = state.get("handle")
    if matc is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnMat(matc), "hipsparseDestroyDnMat(C)"
            )
        except Exception:
            pass
    if matb is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnMat(matb), "hipsparseDestroyDnMat(B)"
            )
        except Exception:
            pass
    if spmat is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
            )
        except Exception:
            pass
    if workspace_allocated:
        try:
            _hip_check_result(hip.hipFree(workspace), "hipFree")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def _spmm_coo_ref_hipsparse(
    data,
    row,
    col,
    B,
    shape,
    out=None,
    return_metadata=False,
):
    state = _prepare_spmm_coo_ref_hipsparse(data, row, col, B, shape, out=out)
    try:
        C = _run_spmm_coo_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse",
            "buffer_size": int(state.get("buffer_size", 0)),
            "format": "coo",
        }
        if return_metadata:
            return C, metadata
        return C
    finally:
        _destroy_spmm_coo_ref_hipsparse_prepared(state)


def _benchmark_spmm_coo_sparse_ref(data, row, col, B, shape, warmup, iters):
    backend, reason = _spmm_coo_sparse_ref_backend(data.dtype, row.dtype)
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result
    if backend == "hipsparse":
        values, ms = _common_mod._benchmark_prepared_cuda_op(
            lambda: _prepare_spmm_coo_ref_hipsparse(data, row, col, B, shape),
            _run_spmm_coo_ref_hipsparse_prepared,
            _destroy_spmm_coo_ref_hipsparse_prepared,
            warmup=warmup,
            iters=iters,
        )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
        return result

    data_cp = _cupy_from_torch(data)
    row_cp = _cupy_from_torch(row.to(torch.int64))
    col_cp = _cupy_from_torch(col.to(torch.int64))
    B_cp = _cupy_from_torch(B)
    A_coo = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    values_cp, ms = _benchmark_cuda_op(
        lambda: A_coo @ B_cp,
        warmup=warmup,
        iters=iters,
    )
    result["values"] = _torch_from_cupy(values_cp)
    result["ms"] = ms
    result["reason"] = None
    return result
def _seg_starts_from_sorted_rows(row_i32, nnz, device):
    if nnz == 0:
        return None
    diff = row_i32[1:] != row_i32[:-1]
    breaks = torch.nonzero(diff, as_tuple=False).flatten().to(torch.int32) + 1
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            breaks,
            torch.tensor([nnz], dtype=torch.int32, device=device),
        ]
    )


@triton.jit
def _spmm_coo_rowrun_real_kernel(
    data_ptr,
    row_ptr,
    col_ptr,
    b_ptr,
    c_ptr,
    seg_starts_ptr,
    n_segs,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    seg = tl.program_id(0)
    pid_n = tl.program_id(1)
    if seg >= n_segs:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_nnz = end - start
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(col_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)

    tl.store(c_ptr + row_id * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_coo_rowrun_complex_kernel(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    b_ri_ptr,
    c_ri_ptr,
    seg_starts_ptr,
    n_segs,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    seg = tl.program_id(0)
    pid_n = tl.program_id(1)
    if seg >= n_segs:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_nnz = end - start
    row_id = tl.load(row_ptr + start)
    acc_re = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc_im = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_re = tl.load(data_ri_ptr + idx * 2, mask=valid, other=0.0)
            a_im = tl.load(data_ri_ptr + idx * 2 + 1, mask=valid, other=0.0)
            a_col = tl.load(col_ptr + idx, mask=valid, other=0)
            b_re = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            b_im = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn + stride_br,
                mask=mask_n & valid,
                other=0.0,
            )
            acc_re = acc_re + a_re.to(ACC_DTYPE) * b_re.to(ACC_DTYPE) - a_im.to(ACC_DTYPE) * b_im.to(ACC_DTYPE)
            acc_im = acc_im + a_re.to(ACC_DTYPE) * b_im.to(ACC_DTYPE) + a_im.to(ACC_DTYPE) * b_re.to(ACC_DTYPE)

    tl.store(c_ri_ptr + row_id * stride_cm + offs_n * stride_cn, acc_re, mask=mask_n)
    tl.store(
        c_ri_ptr + row_id * stride_cm + offs_n * stride_cn + stride_cr,
        acc_im,
        mask=mask_n,
    )

@triton.jit
def _spmm_coo_atomic_real_kernel(
    data_ptr,
    row_ptr,
    col_ptr,
    b_ptr,
    c_ptr,
    nnz,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
):
    idx = tl.program_id(0)
    dense_col = tl.program_id(1)
    if idx >= nnz or dense_col >= n_dense_cols:
        return

    row_k = tl.load(row_ptr + idx)
    col_k = tl.load(col_ptr + idx)
    val_k = tl.load(data_ptr + idx)
    b_val = tl.load(b_ptr + col_k * stride_bk + dense_col * stride_bn)
    tl.atomic_add(
        c_ptr + row_k * stride_cm + dense_col * stride_cn,
        val_k.to(ACC_DTYPE) * b_val.to(ACC_DTYPE),
        sem="relaxed",
    )
@triton.jit
def _spmm_coo_atomic_complex_kernel(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    b_ri_ptr,
    c_ri_ptr,
    nnz,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    ACC_DTYPE: tl.constexpr,
):
    idx = tl.program_id(0)
    dense_col = tl.program_id(1)
    if idx >= nnz or dense_col >= n_dense_cols:
        return

    row_k = tl.load(row_ptr + idx)
    col_k = tl.load(col_ptr + idx)
    a_re = tl.load(data_ri_ptr + idx * 2)
    a_im = tl.load(data_ri_ptr + idx * 2 + 1)
    b_re = tl.load(b_ri_ptr + col_k * stride_bk + dense_col * stride_bn)
    b_im = tl.load(b_ri_ptr + col_k * stride_bk + dense_col * stride_bn + stride_br)
    contrib_re = a_re.to(ACC_DTYPE) * b_re.to(ACC_DTYPE) - a_im.to(ACC_DTYPE) * b_im.to(ACC_DTYPE)
    contrib_im = a_re.to(ACC_DTYPE) * b_im.to(ACC_DTYPE) + a_im.to(ACC_DTYPE) * b_re.to(ACC_DTYPE)
    tl.atomic_add(
        c_ri_ptr + row_k * stride_cm + dense_col * stride_cn,
        contrib_re,
        sem="relaxed",
    )
    tl.atomic_add(
        c_ri_ptr + row_k * stride_cm + dense_col * stride_cn + stride_cr,
        contrib_im,
        sem="relaxed",
    )
def _prepare_spmm_coo_inputs(data, row, col, B, shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple: (n_rows, n_cols)")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, and col must be 1D tensors")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, and col must have the same length (nnz)")
    if B.shape[0] != n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={n_cols}, got {B.shape[0]}")

    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, and B must be CUDA tensors")
    if not all(t.device == data.device for t in (row, col, B)):
        raise ValueError("data, row, col, and B must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if row.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")

    nnz = data.numel()
    if nnz > _INDEX_LIMIT_INT32:
        raise ValueError("nnz exceeds the int32 range supported by the Triton COO kernel")
    if nnz > 0:
        min_row = int(row.min().item())
        max_row = int(row.max().item())
        min_col = int(col.min().item())
        max_col = int(col.max().item())
        if min_row < 0 or max_row >= n_rows:
            raise IndexError("row indices out of range for n_rows")
        if min_col < 0 or max_col >= n_cols:
            raise IndexError("col indices out of range for n_cols")
        if max_row > _INDEX_LIMIT_INT32:
            raise ValueError(
                "row indices exceed the int32 range supported by the Triton kernel"
            )
        if max_col > _INDEX_LIMIT_INT32:
            raise ValueError(
                "column indices exceed the int32 range supported by the Triton kernel"
            )

    data = data.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    B = B.contiguous()

    kernel_row = row.to(torch.int32) if row.dtype == torch.int64 else row
    kernel_col = col.to(torch.int32) if col.dtype == torch.int64 else col
    return data, kernel_row, kernel_col, B, n_rows, n_cols, int(B.shape[1])


def _resolve_spmm_coo_launch_config(n_dense_cols, nnz, block_n=None, block_nnz=256):
    warp_size, factor = _select_spmm_alg1_warp_and_factor(n_dense_cols)

    if block_n is None:
        block_n = warp_size * factor
    if block_nnz is None:
        block_nnz = 256

    if block_n <= 0 or block_nnz <= 0:
        raise ValueError("block_n and block_nnz must be positive when provided")

    return {
        "block_n": int(block_n),
        "block_nnz": int(block_nnz),
        "required_nnz_tiles": int(triton.cdiv(nnz, block_nnz) if nnz > 0 else 0),
        "heuristic_warp_size": int(warp_size),
        "heuristic_factor": int(factor),
    }


def _triton_spmm_coo_rowrun_impl(
    data,
    row,
    col,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
    output_dtype,
):
    device = data.device
    dtype = data.dtype
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0 or data.numel() == 0:
        return torch.zeros((n_rows, n_dense_cols), dtype=output_dtype, device=device)

    seg_starts = _seg_starts_from_sorted_rows(row, int(data.numel()), device)
    n_segs = int(seg_starts.numel()) - 1 if seg_starts is not None else 0
    if n_segs == 0:
        return torch.zeros((n_rows, n_dense_cols), dtype=output_dtype, device=device)

    grid = (n_segs, triton.cdiv(n_dense_cols, block_n))
    if not _is_complex_dtype(dtype):
        C_compute = torch.zeros((n_rows, n_dense_cols), dtype=dtype, device=device)
        acc_dtype = tl.float64 if dtype == torch.float64 else tl.float32
        _spmm_coo_rowrun_real_kernel[grid](
            data,
            row,
            col,
            B,
            C_compute,
            seg_starts,
            n_segs,
            n_dense_cols,
            B.stride(0),
            B.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            BLOCK_N=block_n,
            BLOCK_NNZ=block_nnz,
            ACC_DTYPE=acc_dtype,
        )
        return C_compute if dtype == output_dtype else C_compute.to(output_dtype)

    data_ri = torch.view_as_real(data).contiguous().reshape(-1)
    B_ri = torch.view_as_real(B).contiguous()
    C_ri = torch.zeros((n_rows, n_dense_cols, 2), dtype=B_ri.dtype, device=device)
    acc_dtype = tl.float64 if B_ri.dtype == torch.float64 else tl.float32
    _spmm_coo_rowrun_complex_kernel[grid](
        data_ri,
        row,
        col,
        B_ri,
        C_ri,
        seg_starts,
        n_segs,
        n_dense_cols,
        B_ri.stride(0),
        B_ri.stride(1),
        B_ri.stride(2),
        C_ri.stride(0),
        C_ri.stride(1),
        C_ri.stride(2),
        BLOCK_N=block_n,
        BLOCK_NNZ=block_nnz,
        ACC_DTYPE=acc_dtype,
    )
    C = torch.view_as_complex(C_ri.contiguous())
    return C if dtype == output_dtype else C.to(output_dtype)

def _triton_spmm_coo_atomic_impl(
    data,
    row,
    col,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
    output_dtype,
):
    device = data.device
    dtype = data.dtype
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0 or data.numel() == 0:
        return torch.zeros((n_rows, n_dense_cols), dtype=output_dtype, device=device)

    nnz = int(data.numel())
    if nnz == 0:
        return torch.zeros((n_rows, n_dense_cols), dtype=output_dtype, device=device)

    if not _is_complex_dtype(dtype):
        C_compute = torch.zeros((n_rows, n_dense_cols), dtype=dtype, device=device)
        acc_dtype = tl.float64 if dtype == torch.float64 else tl.float32
        _spmm_coo_atomic_real_kernel[(nnz, n_dense_cols)](
            data,
            row,
            col,
            B,
            C_compute,
            nnz,
            n_dense_cols,
            B.stride(0),
            B.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            ACC_DTYPE=acc_dtype,
        )
        return C_compute if dtype == output_dtype else C_compute.to(output_dtype)

    data_ri = torch.view_as_real(data).contiguous().reshape(-1)
    B_ri = torch.view_as_real(B).contiguous()
    C_ri = torch.zeros((n_rows, n_dense_cols, 2), dtype=B_ri.dtype, device=device)
    acc_dtype = tl.float64 if B_ri.dtype == torch.float64 else tl.float32
    _spmm_coo_atomic_complex_kernel[(nnz, n_dense_cols)](
        data_ri,
        row,
        col,
        B_ri,
        C_ri,
        nnz,
        n_dense_cols,
        B_ri.stride(0),
        B_ri.stride(1),
        B_ri.stride(2),
        C_ri.stride(0),
        C_ri.stride(1),
        C_ri.stride(2),
        ACC_DTYPE=acc_dtype,
    )
    C = torch.view_as_complex(C_ri.contiguous())
    return C if dtype == output_dtype else C.to(output_dtype)

def _normalize_spmm_coo_route(route):
    route = "rowrun" if route is None else str(route).lower()
    if route not in ("rowrun", "atomic"):
        raise ValueError("route must be 'rowrun' or 'atomic'")
    return route


def _triton_spmm_coo_impl(
    data,
    row,
    col,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
    route="rowrun",
    output_dtype=None,
):
    route = _normalize_spmm_coo_route(route)
    resolved_output_dtype = output_dtype if output_dtype is not None else data.dtype
    if route == "rowrun":
        return _triton_spmm_coo_rowrun_impl(
            data,
            row,
            col,
            B,
            n_rows,
            n_dense_cols,
            block_n,
            block_nnz,
            output_dtype=resolved_output_dtype,
        )
    return _triton_spmm_coo_atomic_impl(
        data,
        row,
        col,
        B,
        n_rows,
        n_dense_cols,
        block_n,
        block_nnz,
        output_dtype=resolved_output_dtype,
    )


def _run_spmm_coo_canonical_route(
    canonical_data,
    canonical_row,
    canonical_col,
    canonical_B,
    n_rows,
    n_dense_cols,
    output_dtype,
    block_n=None,
    block_nnz=256,
    out=None,
    return_time=False,
    route="rowrun",
):
    route = _normalize_spmm_coo_route(route)
    launch = _resolve_spmm_coo_launch_config(
        n_dense_cols,
        canonical_data.numel(),
        block_n=block_n,
        block_nnz=block_nnz,
    )

    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != canonical_data.device:
            raise ValueError("out must be on the same CUDA device as the inputs")
        if out.shape != (n_rows, n_dense_cols) or out.dtype != output_dtype:
            raise ValueError("out shape/dtype must match result")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = _triton_spmm_coo_impl(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        route=route,
        output_dtype=output_dtype,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if out is not None:
        out.copy_(C)
        C = out
    if return_time:
        return C, elapsed_ms
    return C

def _run_spmm_coo_route(
    data,
    row,
    col,
    B,
    shape,
    block_n=None,
    block_nnz=256,
    out=None,
    return_time=False,
    route="rowrun",
):
    route = _normalize_spmm_coo_route(route)
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n must be positive when provided")
    if block_nnz is not None and block_nnz <= 0:
        raise ValueError("block_nnz must be positive when provided")

    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        _,
        n_dense_cols,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_inputs(data, row, col, B, shape)

    return _run_spmm_coo_canonical_route(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        output_dtype,
        block_n=block_n,
        block_nnz=block_nnz,
        out=out,
        return_time=return_time,
        route=route,
    )

def flagsparse_spmm_coo(
    data,
    row,
    col,
    B,
    shape,
    block_n=None,
    block_nnz=256,
    out=None,
    return_time=False,
):
    """COO SpMM: C = A @ B using a native Triton COO row-run kernel by default."""
    return _run_spmm_coo_route(
        data,
        row,
        col,
        B,
        shape,
        block_n=block_n,
        block_nnz=block_nnz,
        out=out,
        return_time=return_time,
        route="rowrun",
    )


def _build_spmm_coo_pytorch_reference_from_canonical(
    canonical_data,
    canonical_row,
    canonical_col,
    canonical_B,
    shape,
    output_dtype,
):
    canonical_coo = _build_torch_sparse_coo(
        canonical_data,
        canonical_row,
        canonical_col,
        shape,
    )
    expected = torch.sparse.mm(canonical_coo, canonical_B)
    return expected if expected.dtype == output_dtype else expected.to(output_dtype)



def _build_spmm_coo_pytorch_reference(data, row, col, B, shape):
    native_data, native_row, native_col, native_B, n_rows, n_cols, n_dense_cols = _prepare_spmm_coo_inputs(
        data, row, col, B, shape
    )
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        _,
        _,
        _,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_prepared(
        native_data,
        native_row,
        native_col,
        native_B,
        n_rows,
        n_cols,
        n_dense_cols,
    )
    native_coo = _build_torch_sparse_coo(native_data, native_row, native_col, shape)
    pytorch_format = "COO"
    pytorch_reason = None
    pytorch_op = lambda: torch.sparse.mm(native_coo, native_B)
    expected = _build_spmm_coo_pytorch_reference_from_canonical(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        shape,
        output_dtype,
    )
    return expected, pytorch_op, pytorch_format, pytorch_reason


def _benchmark_spmm_coo_canonical_route(
    canonical_data,
    canonical_row,
    canonical_col,
    canonical_B,
    n_rows,
    n_dense_cols,
    output_dtype,
    warmup,
    iters,
    block_n,
    block_nnz,
    route,
):
    route = _normalize_spmm_coo_route(route)
    op = lambda: _run_spmm_coo_canonical_route(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        output_dtype,
        block_n=block_n,
        block_nnz=block_nnz,
        return_time=False,
        route=route,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = op()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000.0
    values, steady_ms = _benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return values, steady_ms, first_call_ms



def _benchmark_spmm_coo_route(
    data,
    row,
    col,
    B,
    shape,
    warmup,
    iters,
    block_n,
    block_nnz,
    route,
):
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        _,
        n_dense_cols,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_inputs(data, row, col, B, shape)
    return _benchmark_spmm_coo_canonical_route(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        output_dtype,
        warmup,
        iters,
        block_n,
        block_nnz,
        route,
    )
def _spmm_coo_pairwise_summary(candidate, reference, value_dtype):
    metrics = _spmm_validation_metrics(candidate, reference)
    atol, rtol = _spmm_coo_reference_tolerance(value_dtype)
    if candidate.numel() == 0:
        error_ratio = 0.0
    else:
        diff = torch.abs(candidate - reference)
        denom = atol + rtol * torch.abs(reference)
        error_ratio = float(torch.max(diff / denom).item())
    return {
        "match": torch.allclose(candidate, reference, atol=atol, rtol=rtol),
        "error_ratio": error_ratio,
        "max_abs_error": metrics["max_abs_error"],
        "max_relative_error": metrics["max_relative_error"],
        "sum_relative_error": metrics["sum_relative_error"],
    }


def benchmark_spmm_coo_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=256,
    run_cusparse=True,
    route="rowrun",
    compare_routes=False,
):
    """Benchmark native COO SpMM vs PyTorch COO sparse.mm and CuPy/cuSPARSE COO @ dense."""
    selected_route = _normalize_spmm_coo_route(route)
    device = torch.device("cuda")
    data, row, col = _build_random_coo(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    B = _build_random_dense((n_cols, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)

    native_data, native_row, native_col, native_B, _, _, _ = _prepare_spmm_coo_inputs(
        data, row, col, B, shape
    )
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_cols,
        n_dense_cols,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_prepared(
        native_data,
        native_row,
        native_col,
        native_B,
        n_rows,
        n_cols,
        native_B.shape[1],
    )
    launch = _resolve_spmm_coo_launch_config(
        n_dense_cols,
        canonical_data.numel(),
        block_n=block_n,
        block_nnz=block_nnz,
    )
    seg_starts = _seg_starts_from_sorted_rows(canonical_row, canonical_data.numel(), device)
    n_row_runs = int(seg_starts.numel()) - 1 if seg_starts is not None else 0

    expected = _build_spmm_coo_pytorch_reference_from_canonical(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        shape,
        output_dtype,
    )
    pytorch_coo = _build_torch_sparse_coo(native_data, native_row, native_col, shape)
    pytorch_op = lambda: torch.sparse.mm(pytorch_coo, native_B)
    pytorch_format = "COO"
    pytorch_reason = None

    triton_C, triton_ms, triton_first_call_ms = _benchmark_spmm_coo_canonical_route(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        output_dtype,
        warmup,
        iters,
        launch["block_n"],
        launch["block_nnz"],
        selected_route,
    )
    triton_summary = _spmm_coo_pairwise_summary(triton_C, expected, value_dtype)
    triton_match = triton_summary["match"]

    pytorch_values = expected
    pytorch_ms = None
    try:
        pytorch_values, pytorch_ms = _benchmark_cuda_op(
            pytorch_op, warmup=warmup, iters=iters
        )
    except Exception as exc:
        pytorch_reason = str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"

    cusparse_ms = None
    cusparse_match = None
    cusparse_reason = None
    cusparse_values = None
    cusparse_summary = None
    if run_cusparse:
        try:
            sparse_ref = _benchmark_spmm_coo_sparse_ref(
                native_data,
                native_row,
                native_col,
                native_B,
                shape,
                warmup,
                iters,
            )
            cusparse_reason = sparse_ref["reason"]
            cusparse_ms = sparse_ref["ms"]
            cusparse_values = sparse_ref["values"]
            if cusparse_values is not None:
                cusparse_summary = _spmm_coo_pairwise_summary(
                    cusparse_values,
                    expected,
                    value_dtype,
                )
                cusparse_match = cusparse_summary["match"]
        except Exception as exc:
            cusparse_reason = str(exc)

    route_results = None
    parity = None
    route_samples = None
    if compare_routes:
        route_outputs = {selected_route: triton_C}
        route_results = {
            selected_route: {
                "route": selected_route,
                "ms": triton_ms,
                "first_call_ms": triton_first_call_ms,
                "match_reference": triton_summary["match"],
                "error_ratio": triton_summary["error_ratio"],
                "max_abs_error": triton_summary["max_abs_error"],
                "max_relative_error": triton_summary["max_relative_error"],
                "match_cusparse": (
                    None if cusparse_values is None else torch.allclose(
                        triton_C,
                        cusparse_values,
                        atol=_spmm_coo_reference_tolerance(value_dtype)[0],
                        rtol=_spmm_coo_reference_tolerance(value_dtype)[1],
                    )
                ),
                "error": None,
            }
        }

        for extra_route in ("rowrun", "atomic"):
            if extra_route in route_outputs:
                continue
            try:
                extra_values, extra_ms, extra_first_call_ms = _benchmark_spmm_coo_canonical_route(
                    canonical_data,
                    canonical_row,
                    canonical_col,
                    canonical_B,
                    n_rows,
                    n_dense_cols,
                    output_dtype,
                    warmup,
                    iters,
                    launch["block_n"],
                    launch["block_nnz"],
                    extra_route,
                )
                extra_summary = _spmm_coo_pairwise_summary(extra_values, expected, value_dtype)
                route_outputs[extra_route] = extra_values
                route_results[extra_route] = {
                    "route": extra_route,
                    "ms": extra_ms,
                    "first_call_ms": extra_first_call_ms,
                    "match_reference": extra_summary["match"],
                    "error_ratio": extra_summary["error_ratio"],
                    "max_abs_error": extra_summary["max_abs_error"],
                    "max_relative_error": extra_summary["max_relative_error"],
                    "match_cusparse": (
                        None if cusparse_values is None else torch.allclose(
                            extra_values,
                            cusparse_values,
                            atol=_spmm_coo_reference_tolerance(value_dtype)[0],
                            rtol=_spmm_coo_reference_tolerance(value_dtype)[1],
                        )
                    ),
                    "error": None,
                }
            except Exception as exc:
                route_results[extra_route] = {
                    "route": extra_route,
                    "ms": None,
                    "first_call_ms": None,
                    "match_reference": False,
                    "error_ratio": None,
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "match_cusparse": None,
                    "error": str(exc),
                }

        def _safe_parity(lhs, rhs):
            if lhs in route_outputs and rhs in route_outputs:
                return _spmm_coo_pairwise_summary(route_outputs[lhs], route_outputs[rhs], value_dtype)
            return {
                "match": None,
                "error_ratio": None,
                "max_abs_error": None,
                "max_relative_error": None,
                "sum_relative_error": None,
            }

        parity = {
            "rowrun_vs_atomic": _safe_parity("rowrun", "atomic"),
        }
        route_samples = route_outputs
    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if (pytorch_ms is not None and triton_ms > 0) else None
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms if (cusparse_ms is not None and triton_ms > 0) else None
    )
    threshold = _spmm_relative_threshold(value_dtype)
    return {
        "parameters": {
            "format": "coo",
            "internal_format": f"native-{selected_route}",
            "route": selected_route,
            "compare_routes": bool(compare_routes),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "required_nnz_tiles": launch["required_nnz_tiles"],
            "heuristic_warp_size": launch["heuristic_warp_size"],
            "heuristic_factor": launch["heuristic_factor"],
            "n_row_runs": n_row_runs,
            "run_cusparse": run_cusparse,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "triton_first_call_ms": triton_first_call_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_summary["max_abs_error"],
            "triton_max_abs_error": triton_summary["max_abs_error"],
            "triton_max_relative_error": triton_summary["max_relative_error"],
            "triton_sum_relative_error": triton_summary["sum_relative_error"],
            "triton_relative_threshold": threshold,
            "triton_strict_allclose_match": triton_match,
            "pytorch_match_reference": True,
            "pytorch_max_error": 0.0,
            "pytorch_max_abs_error": 0.0,
            "pytorch_max_relative_error": 0.0,
            "pytorch_sum_relative_error": 0.0,
            "pytorch_relative_threshold": threshold,
            "cusparse_match_reference": cusparse_match,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": (cusparse_summary["max_abs_error"] if cusparse_summary is not None else None),
            "cusparse_max_abs_error": (cusparse_summary["max_abs_error"] if cusparse_summary is not None else None),
            "cusparse_max_relative_error": (cusparse_summary["max_relative_error"] if cusparse_summary is not None else None),
            "cusparse_sum_relative_error": (cusparse_summary["sum_relative_error"] if cusparse_summary is not None else None),
            "cusparse_relative_threshold": threshold,
            "cusparse_strict_allclose_match": cusparse_match,
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "pytorch_sparse_format": pytorch_format,
            "cusparse_unavailable_reason": cusparse_reason,
            "flagsparse_internal_route": f"coo-native-{selected_route}",
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_C,
            "reference": expected,
            "cusparse": cusparse_values,
        },
        "route_results": route_results,
        "parity": parity,
        "route_samples": route_samples,
    }

def comprehensive_spmm_coo_test(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=256,
    run_cusparse=True,
):
    """Full COO SpMM benchmark entry for one configuration."""
    return benchmark_spmm_coo_case(
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        n_dense_cols=n_dense_cols,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        block_n=block_n,
        block_nnz=block_nnz,
        run_cusparse=run_cusparse,
    )
