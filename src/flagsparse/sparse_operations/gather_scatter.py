"""Gather and scatter (Triton kernels + cuSPARSE-style baselines)."""

from ._common import *

import triton
import triton.language as tl

@triton.jit
def _gather_real_kernel(
    sparse_values_ptr,
    dense_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    gathered_values = tl.load(dense_values_ptr + indices, mask=mask, other=0.0)
    tl.store(sparse_values_ptr + offsets, gathered_values, mask=mask)


@triton.jit
def _gather_complex_kernel(
    sparse_values_ri_ptr,
    dense_values_ri_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    dense_offsets = indices * 2
    sparse_offsets = offsets * 2

    gathered_real = tl.load(dense_values_ri_ptr + dense_offsets, mask=mask, other=0.0)
    gathered_imag = tl.load(dense_values_ri_ptr + dense_offsets + 1, mask=mask, other=0.0)

    tl.store(sparse_values_ri_ptr + sparse_offsets, gathered_real, mask=mask)
    tl.store(sparse_values_ri_ptr + sparse_offsets + 1, gathered_imag, mask=mask)


@triton.jit
def _scatter_real_kernel(
    dense_values_ptr,
    sparse_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    values = tl.load(sparse_values_ptr + offsets, mask=mask, other=0.0)
    tl.store(dense_values_ptr + indices, values, mask=mask)


@triton.jit
def _scatter_complex_kernel(
    dense_values_ri_ptr,
    sparse_values_ri_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    dense_offsets = indices * 2
    sparse_offsets = offsets * 2

    values_real = tl.load(sparse_values_ri_ptr + sparse_offsets, mask=mask, other=0.0)
    values_imag = tl.load(sparse_values_ri_ptr + sparse_offsets + 1, mask=mask, other=0.0)

    tl.store(dense_values_ri_ptr + dense_offsets, values_real, mask=mask)
    tl.store(dense_values_ri_ptr + dense_offsets + 1, values_imag, mask=mask)


def _triton_gather_impl(dense_vector, kernel_indices, block_size=1024):
    nnz = kernel_indices.numel()
    if nnz == 0:
        return torch.empty(0, dtype=dense_vector.dtype, device=dense_vector.device)

    grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)

    if not _is_complex_dtype(dense_vector.dtype):
        sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
        _gather_real_kernel[grid](
            sparse_values,
            dense_vector,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
        )
        return sparse_values

    sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
    dense_values_ri = torch.view_as_real(dense_vector).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)

    _gather_complex_kernel[grid](
        sparse_values_ri,
        dense_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
    )
    return sparse_values


def _triton_scatter_impl(
    sparse_values,
    kernel_indices,
    dense_size,
    out=None,
    block_size=1024,
    reset_output=True,
    index_fallback_policy="auto",
    return_metadata=False,
):
    index_fallback_policy = str(index_fallback_policy).lower()
    if index_fallback_policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")

    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        if reset_output:
            dense_values.zero_()

    nnz = kernel_indices.numel()
    scatter_meta = {
        "index_fallback_applied": False,
        "index_fallback_reason": None,
        "kernel_index_dtype": str(kernel_indices.dtype).replace("torch.", ""),
    }
    if nnz == 0:
        if return_metadata:
            return dense_values, scatter_meta
        return dense_values

    try:
        _launch_triton_scatter_kernel(
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            block_size=block_size,
        )
    except Exception as exc:
        if kernel_indices.dtype != torch.int64 or index_fallback_policy != "auto":
            raise RuntimeError(
                f"Triton scatter failed for index dtype {kernel_indices.dtype}: "
                f"{exc.__class__.__name__}: {str(exc)}"
            ) from exc

        max_index = int(kernel_indices.max().item()) if nnz > 0 else -1
        if max_index > _INDEX_LIMIT_INT32:
            raise RuntimeError(
                "Triton scatter failed for int64 indices, and int32 fallback is invalid: "
                f"max index {max_index} exceeds int32 range"
            ) from exc

        fallback_indices = kernel_indices.to(torch.int32)
        try:
            _launch_triton_scatter_kernel(
                dense_values,
                sparse_values,
                fallback_indices,
                nnz,
                block_size=block_size,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "Triton scatter failed for int64 indices, and int32 fallback also failed: "
                f"{fallback_exc.__class__.__name__}: {str(fallback_exc)}"
            ) from fallback_exc

        scatter_meta["index_fallback_applied"] = True
        scatter_meta["index_fallback_reason"] = (
            f"int64 kernel launch failed: {exc.__class__.__name__}: {str(exc)}"
        )
        scatter_meta["kernel_index_dtype"] = "int32"

    if return_metadata:
        return dense_values, scatter_meta
    return dense_values


def _launch_triton_scatter_kernel(
    dense_values, sparse_values, kernel_indices, nnz, block_size=1024
):
    grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)
    if not _is_complex_dtype(sparse_values.dtype):
        _scatter_real_kernel[grid](
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
        )
        return
    dense_values_ri = torch.view_as_real(dense_values).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)
    _scatter_complex_kernel[grid](
        dense_values_ri,
        sparse_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
    )


def _validate_gather_value_dtype(dense_vector, op_name):
    return None


def _cusparse_spmv(selector_matrix, dense_vector):
    if cp is not None and cpx_sparse is not None and isinstance(selector_matrix, cpx_sparse.spmatrix):
        if torch.is_tensor(dense_vector):
            out_dtype = dense_vector.dtype
            dense_for_compute = (
                dense_vector.to(torch.float32)
                if dense_vector.dtype == torch.bfloat16
                else dense_vector
            )
            dense_cp = _cupy_from_torch(dense_for_compute)
            out_cp = selector_matrix @ dense_cp
            out_torch = _torch_from_cupy(out_cp)
            if out_dtype == torch.bfloat16:
                out_torch = out_torch.to(torch.bfloat16)
            return out_torch

        if cp is not None and isinstance(dense_vector, cp.ndarray):
            return selector_matrix @ dense_vector

        raise TypeError("dense_vector must be torch.Tensor or cupy.ndarray")

    # Fallback path: torch sparse SpMV (still CUDA-backed).
    if torch.is_tensor(selector_matrix) and selector_matrix.is_sparse:
        if not torch.is_tensor(dense_vector):
            raise TypeError("dense_vector must be torch.Tensor for torch sparse fallback")
        out_dtype = dense_vector.dtype
        dense_for_compute = (
            dense_vector.to(torch.float32)
            if dense_vector.dtype == torch.bfloat16
            else dense_vector
        )
        out = torch.sparse.mm(selector_matrix, dense_for_compute.unsqueeze(1)).squeeze(1)
        if out_dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)
        return out

    if cp is None or cpx_sparse is None:
        raise RuntimeError(
            "CuPy is not available and torch sparse fallback selector is not provided"
        )
    raise TypeError(
        "selector_matrix must be a cupyx sparse matrix or torch sparse tensor"
    )


def _make_gather_selector_matrix(indices, dense_size, value_dtype):
    if cp is not None and cpx_sparse is not None:
        rows_cp = cp.arange(indices.numel(), dtype=cp.int64)
        cols_cp = _cupy_from_torch(indices.to(torch.int64))
        vals_cp = cp.ones(indices.numel(), dtype=_cupy_dtype_from_torch(value_dtype))
        return cpx_sparse.coo_matrix(
            (vals_cp, (rows_cp, cols_cp)),
            shape=(indices.numel(), dense_size),
        )

    rows = torch.arange(indices.numel(), dtype=torch.int64, device=indices.device)
    cols = indices.to(torch.int64)
    coords = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.numel(), dtype=value_dtype, device=indices.device)
    return torch.sparse_coo_tensor(
        coords, values, size=(indices.numel(), dense_size), device=indices.device
    ).coalesce()


def _make_scatter_selector_matrix(indices, dense_size, value_dtype):
    if cp is not None and cpx_sparse is not None:
        rows_cp = _cupy_from_torch(indices.to(torch.int64))
        cols_cp = cp.arange(indices.numel(), dtype=cp.int64)
        vals_cp = cp.ones(indices.numel(), dtype=_cupy_dtype_from_torch(value_dtype))
        return cpx_sparse.coo_matrix(
            (vals_cp, (rows_cp, cols_cp)),
            shape=(dense_size, indices.numel()),
        )

    rows = indices.to(torch.int64)
    cols = torch.arange(indices.numel(), dtype=torch.int64, device=indices.device)
    coords = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.numel(), dtype=value_dtype, device=indices.device)
    return torch.sparse_coo_tensor(
        coords, values, size=(dense_size, indices.numel()), device=indices.device
    ).coalesce()


def _pytorch_scatter_impl(sparse_values, indices, dense_size, out=None, reset_output=True):
    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        if reset_output:
            dense_values.zero_()
    dense_values.index_copy_(0, indices.to(torch.int64), sparse_values)
    return dense_values


def flagsparse_gather(a, indices, out=None, mode="raise", block_size=1024, return_time=False):
    """CuPy-style gather (take): out = a[indices]."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_vector, dense_backend = _to_torch_tensor(a, "a")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    dense_vector, indices_tensor, kernel_indices = _prepare_inputs(dense_vector, indices_tensor)
    _validate_gather_value_dtype(dense_vector, "flagsparse_gather")

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if out is not None:
        out_tensor, _ = _to_torch_tensor(out, "out")
        if out_tensor.shape != sparse_values.shape:
            raise ValueError("out shape must match gather output shape")
        if out_tensor.dtype != sparse_values.dtype:
            raise TypeError("out dtype must match gather output dtype")
        out_tensor.copy_(sparse_values)
        result = out if dense_backend == "cupy" else out_tensor
    else:
        result = _to_backend_like(sparse_values, a)

    if return_time:
        return result, execution_time_ms
    return result


def flagsparse_scatter(
    a,
    indices,
    values,
    mode="raise",
    block_size=1024,
    return_time=False,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    """CuPy-style scatter (put): a[indices] = values (in-place)."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_tensor, dense_backend = _to_torch_tensor(a, "a")
    values_tensor, _ = _to_torch_tensor(values, "values")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    values_tensor, _, kernel_indices, dense_size, _ = _prepare_scatter_inputs(
        values_tensor,
        indices_tensor,
        dense_size=dense_tensor.numel(),
        out=dense_tensor,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    _ = _triton_scatter_impl(
        values_tensor,
        kernel_indices,
        dense_size=dense_size,
        out=dense_tensor,
        block_size=block_size,
        reset_output=reset_output,
        index_fallback_policy=index_fallback_policy,
    )
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if dense_backend == "cupy":
        # DLPack view updates dense_tensor and cupy array shares memory.
        pass

    if return_time:
        return execution_time_ms
    return None


# Backward compatibility wrappers.
def triton_cusparse_gather(dense_vector, indices, block_size=1024):
    return flagsparse_gather(
        dense_vector, indices, block_size=block_size, return_time=True
    )


def triton_cusparse_scatter(
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    block_size=1024,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    sparse_values_t, sparse_backend = _to_torch_tensor(sparse_values, "sparse_values")
    indices_t, _ = _to_torch_tensor(indices, "indices")
    if out is None:
        if dense_size is None:
            dense_size = int(indices_t.max().item()) + 1 if indices_t.numel() > 0 else 0
        out = torch.zeros(
            int(dense_size), dtype=sparse_values_t.dtype, device=sparse_values_t.device
        )
    elapsed_ms = flagsparse_scatter(
        out,
        indices_t,
        sparse_values_t,
        block_size=block_size,
        return_time=True,
        reset_output=reset_output,
        dtype_policy=dtype_policy,
        index_fallback_policy=index_fallback_policy,
    )
    if sparse_backend == "cupy":
        return _to_backend_like(out, sparse_values), elapsed_ms
    return out, elapsed_ms


def pytorch_index_gather(dense_vector, indices):
    """Baseline gather using PyTorch native indexing."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = dense_vector[indices]
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    return sparse_values, execution_time_ms


def pytorch_index_scatter(
    sparse_values, indices, dense_size=None, out=None, reset_output=True, dtype_policy="auto"
):
    """Baseline scatter using PyTorch index_copy_."""
    sparse_values, indices, _, dense_size, _ = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=out,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    dense_values = _pytorch_scatter_impl(
        sparse_values, indices, dense_size, out=out, reset_output=reset_output
    )
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    return dense_values, execution_time_ms


def cusparse_spmv_gather(dense_vector, indices, selector_matrix=None):
    """Equivalent gather baseline via cuSPARSE-backed COO SpMV."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    _validate_gather_value_dtype(dense_vector, "cusparse_spmv_gather")
    skip_reason = _cusparse_baseline_skip_reason(dense_vector.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_gather_selector_matrix(
            indices, dense_vector.numel(), dense_vector.dtype
        )

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        sparse_values = _cusparse_spmv(selector_matrix, dense_vector)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE gather baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return sparse_values, execution_time_ms, selector_matrix


def cusparse_spmv_scatter(
    sparse_values, indices, dense_size=None, selector_matrix=None, dtype_policy="auto"
):
    """Equivalent scatter baseline via cuSPARSE-backed COO SpMV."""
    sparse_values, indices, _, dense_size, _ = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=None,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    skip_reason = _cusparse_baseline_skip_reason(sparse_values.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_scatter_selector_matrix(indices, dense_size, sparse_values.dtype)

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        dense_values = _cusparse_spmv(selector_matrix, sparse_values)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE scatter baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return dense_values, execution_time_ms, selector_matrix


_CUPY_GATHER_KERNEL_CACHE = {}
_CUPY_RAW_KIND_TO_DTYPE = {
    16: cp.uint16 if cp is not None else None,
    32: cp.uint32 if cp is not None else None,
    64: cp.uint64 if cp is not None else None,
}
_CUPY_RAW_KIND_TO_CTYPE = {
    16: "unsigned short",
    32: "unsigned int",
    64: "unsigned long long",
}
_CUPY_INDEX_TO_CTYPE = {
    "i32": "int",
    "i64": "long long",
}


def _cupy_gather_detect_layout(dense_vector):
    if dense_vector.ndim == 1 and dense_vector.dtype in (torch.float16, torch.bfloat16):
        return "scalar16"
    if dense_vector.ndim == 1 and dense_vector.dtype == torch.complex64:
        return "complex64"
    raise TypeError(
        "Unsupported gather input format. Expected one of: "
        "1D float16/bfloat16 or 1D complex64."
    )


def _cupy_gather_dense_size(dense_vector, layout):
    if layout in ("scalar16", "complex64"):
        return int(dense_vector.shape[0])
    raise RuntimeError(f"Unknown gather layout: {layout}")


def _cupy_gather_validate_inputs(dense_vector, indices):
    if not dense_vector.is_cuda or not indices.is_cuda:
        raise ValueError("dense_vector and indices must both be CUDA tensors")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")

    layout = _cupy_gather_detect_layout(dense_vector)
    dense_size = _cupy_gather_dense_size(dense_vector, layout)

    if indices.numel() > 0:
        if torch.any(indices < 0).item():
            raise IndexError("indices must be non-negative")
        max_index = int(indices.max().item())
        if max_index >= dense_size:
            raise IndexError(
                f"indices out of range: max index {max_index}, dense size {dense_size}"
            )

    return layout, dense_size


def _cupy_gather_validate_combo(dense_vector, indices, layout):
    # Keep only the required extra gather combos:
    # Half+Int64, Bfloat16+Int32/Int64, Complex64+Int32/Int64
    if layout == "scalar16":
        if dense_vector.dtype == torch.float16:
            if indices.dtype != torch.int64:
                raise TypeError("float16 gather_cupy supports only int64 indices")
            return
        if dense_vector.dtype == torch.bfloat16:
            return
        raise TypeError("scalar16 gather_cupy supports only float16/bfloat16")

    if layout == "complex64":
        return

    raise TypeError(f"Unsupported gather_cupy layout: {layout}")


def _cupy_gather_layout_raw_kind(layout):
    if layout == "scalar16":
        return 16
    if layout == "complex64":
        return 64
    raise RuntimeError(f"Unknown gather layout: {layout}")


def _cupy_gather_index_tag(index_dtype):
    return "i32" if index_dtype == torch.int32 else "i64"


def _cupy_gather_kernel_name(raw_kind, index_tag):
    return f"flagsparse_gather_cupy_raw{raw_kind}_{index_tag}"


def _cupy_gather_build_source(raw_kind, index_tag):
    value_ctype = _CUPY_RAW_KIND_TO_CTYPE[raw_kind]
    index_ctype = _CUPY_INDEX_TO_CTYPE[index_tag]
    kernel_name = _cupy_gather_kernel_name(raw_kind, index_tag)
    return f"""
extern "C" __global__
void {kernel_name}(const {value_ctype}* y, const {index_ctype}* x_ind, {value_ctype}* x_val, long long nnz)
{{
    long long tid = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = tid; i < nnz; i += stride)
    {{
        long long dense_idx = (long long)x_ind[i];
        x_val[i] = y[dense_idx];
    }}
}}
"""


def _cupy_gather_get_kernel(raw_kind, index_dtype):
    _require_cupy()
    index_tag = _cupy_gather_index_tag(index_dtype)
    key = (raw_kind, index_tag)
    if key not in _CUPY_GATHER_KERNEL_CACHE:
        src = _cupy_gather_build_source(raw_kind, index_tag)
        name = _cupy_gather_kernel_name(raw_kind, index_tag)
        _CUPY_GATHER_KERNEL_CACHE[key] = cp.RawKernel(src, name, options=("--std=c++14",))
    return _CUPY_GATHER_KERNEL_CACHE[key]


def _cupy_gather_dense_to_raw_torch(dense_t, layout):
    if layout == "scalar16":
        return dense_t.reshape(-1).view(torch.uint16)
    if layout == "complex64":
        return dense_t.reshape(-1).view(torch.uint64)
    raise RuntimeError(f"Unknown gather layout: {layout}")


def _cupy_gather_raw_to_dense_torch(out_raw_t, layout, dense_t_dtype):
    if layout == "scalar16":
        return out_raw_t.view(dense_t_dtype).reshape(-1)
    if layout == "complex64":
        return out_raw_t.view(torch.complex64).reshape(-1)
    raise RuntimeError(f"Unknown gather layout: {layout}")


def _cupy_gather_empty(layout, dense_dtype, device):
    if layout in ("scalar16", "complex64"):
        return torch.empty(0, dtype=dense_dtype, device=device)
    raise RuntimeError(f"Unknown gather layout: {layout}")


def _cupy_gather_selector_dtype(layout, dense_dtype):
    if layout == "scalar16":
        return dense_dtype
    if layout == "complex64":
        return torch.complex64
    raise RuntimeError(f"Unknown gather layout: {layout}")


def _cupy_gather_prepare_dense(dense_vector, indices):
    layout, dense_size = _cupy_gather_validate_inputs(dense_vector, indices)
    _cupy_gather_validate_combo(dense_vector, indices, layout)
    return dense_vector, layout, dense_size, None


def _cupy_gather_restore_output(gathered_t, restore_mode):
    return gathered_t


def _launch_cupy_gather_kernel(gather_kernel, dense_raw_cp, indices_cp, out_raw_cp, nnz):
    thread_per_block = 256
    block_per_grid = min(2, (nnz + thread_per_block - 1) // thread_per_block)
    if block_per_grid <= 0:
        block_per_grid = 1
    gather_kernel(
        (int(block_per_grid),),
        (int(thread_per_block),),
        (dense_raw_cp, indices_cp, out_raw_cp, cp.int64(nnz)),
    )


def flagsparse_gather_cupy(
    a,
    indices,
    out=None,
    mode="raise",
    index_fallback_policy="auto",
    return_time=False,
    return_metadata=False,
):
    """CuPy-style gather (take) for extra gather dtype/index combinations."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")
    index_fallback_policy = str(index_fallback_policy).lower()
    if index_fallback_policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")

    _require_cupy()
    dense_t, dense_backend = _to_torch_tensor(a, "a")
    indices_t, _ = _to_torch_tensor(indices, "indices")

    dense_t = dense_t.contiguous()
    indices_t = indices_t.contiguous()
    runtime_dense_t, layout, _, restore_mode = _cupy_gather_prepare_dense(dense_t, indices_t)
    gather_meta = {
        "index_fallback_applied": False,
        "index_fallback_reason": None,
        "kernel_index_dtype": str(indices_t.dtype).replace("torch.", ""),
    }

    nnz = int(indices_t.numel())
    if nnz == 0:
        gathered_t = _cupy_gather_empty(layout, runtime_dense_t.dtype, runtime_dense_t.device)
        execution_time_ms = 0.0
    else:
        raw_kind = _cupy_gather_layout_raw_kind(layout)
        raw_dtype = _CUPY_RAW_KIND_TO_DTYPE[raw_kind]
        indices_cp = _cupy_from_torch(indices_t)
        dense_raw_t = _cupy_gather_dense_to_raw_torch(runtime_dense_t, layout)
        dense_raw_cp = _cupy_from_torch(dense_raw_t)
        out_raw_cp = cp.empty(nnz, dtype=raw_dtype)
        gather_kernel = _cupy_gather_get_kernel(raw_kind, indices_t.dtype)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        stream_ptr = torch.cuda.current_stream(device=runtime_dense_t.device).cuda_stream
        try:
            with cp.cuda.ExternalStream(stream_ptr):
                _launch_cupy_gather_kernel(
                    gather_kernel,
                    dense_raw_cp,
                    indices_cp,
                    out_raw_cp,
                    nnz,
                )
        except Exception as exc:
            if indices_t.dtype != torch.int64 or index_fallback_policy != "auto":
                raise RuntimeError(
                    f"CuPy gather failed for index dtype {indices_t.dtype}: "
                    f"{exc.__class__.__name__}: {str(exc)}"
                ) from exc

            max_index = int(indices_t.max().item()) if nnz > 0 else -1
            if max_index > _INDEX_LIMIT_INT32:
                raise RuntimeError(
                    "CuPy gather failed for int64 indices, and int32 fallback is invalid: "
                    f"max index {max_index} exceeds int32 range"
                ) from exc

            fallback_indices_t = indices_t.to(torch.int32)
            fallback_indices_cp = _cupy_from_torch(fallback_indices_t)
            fallback_kernel = _cupy_gather_get_kernel(raw_kind, torch.int32)
            try:
                with cp.cuda.ExternalStream(stream_ptr):
                    _launch_cupy_gather_kernel(
                        fallback_kernel,
                        dense_raw_cp,
                        fallback_indices_cp,
                        out_raw_cp,
                        nnz,
                    )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "CuPy gather failed for int64 indices, and int32 fallback also failed: "
                    f"{fallback_exc.__class__.__name__}: {str(fallback_exc)}"
                ) from fallback_exc
            gather_meta["index_fallback_applied"] = True
            gather_meta["index_fallback_reason"] = (
                f"int64 kernel launch failed: {exc.__class__.__name__}: {str(exc)}"
            )
            gather_meta["kernel_index_dtype"] = "int32"
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        out_raw_t = _torch_from_cupy(out_raw_cp)
        gathered_t = _cupy_gather_raw_to_dense_torch(out_raw_t, layout, runtime_dense_t.dtype)

    gathered_t = _cupy_gather_restore_output(gathered_t, restore_mode)

    if out is not None:
        out_t, _ = _to_torch_tensor(out, "out")
        if out_t.shape != gathered_t.shape:
            raise ValueError("out shape must match gather output shape")
        if out_t.dtype != gathered_t.dtype:
            raise TypeError("out dtype must match gather output dtype")
        if not out_t.is_cuda:
            raise ValueError("out must be a CUDA tensor/array")
        out_t.copy_(gathered_t)
        result = out if dense_backend == "cupy" else out_t
    else:
        result = _to_backend_like(gathered_t, a)

    if return_time and return_metadata:
        return result, execution_time_ms, gather_meta
    if return_time:
        return result, execution_time_ms
    if return_metadata:
        return result, gather_meta
    return result


def cusparse_spmv_gather_cupy(dense_vector, indices, selector_matrix=None):
    """Equivalent gather baseline via cuSPARSE-backed COO SpMV for cupy gather path."""
    _require_cupy()
    dense_t, _ = _to_torch_tensor(dense_vector, "dense_vector")
    indices_t, _ = _to_torch_tensor(indices, "indices")

    dense_t = dense_t.contiguous()
    indices_t = indices_t.contiguous()
    runtime_dense_t, layout, dense_size, restore_mode = _cupy_gather_prepare_dense(
        dense_t, indices_t
    )

    selector_dtype = _cupy_gather_selector_dtype(layout, runtime_dense_t.dtype)
    if selector_matrix is None:
        selector_matrix = _make_gather_selector_matrix(indices_t, dense_size, selector_dtype)

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        if layout in ("scalar16", "complex64"):
            gathered_t = _cusparse_spmv(selector_matrix, runtime_dense_t)
        else:
            raise RuntimeError(f"Unknown gather layout: {layout}")
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE gather baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    gathered_t = _cupy_gather_restore_output(gathered_t, restore_mode)
    result = _to_backend_like(gathered_t, dense_vector)
    return result, execution_time_ms, selector_matrix
