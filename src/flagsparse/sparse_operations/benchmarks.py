"""Benchmarks for gather, scatter, SpMV, SpMM, SpGEMM, SDDMM, and SpSM."""

from ._common import *

from .gather_scatter import (
    SUPPORTED_SCATTER_VALUE_DTYPES,
    _scatter_dtype_error_message,
    _pytorch_scatter_impl,
    _triton_gather_impl,
    _triton_scatter_impl,
    benchmark_hipsparse_gather,
    benchmark_hipsparse_scatter,
    hipsparse_gather,
    hipsparse_scatter,
)
from .spmv_csr import (
    SPMV_OP_NON,
    SPMV_OP_TRANS,
    _normalize_spmv_op,
    _spmv_op_to_name,
    _spmv_op_transposes,
    flagsparse_spmv_csr,
    prepare_spmv_csr,
)
from .spmm_csr import (
    benchmark_spmm_case,
    benchmark_spmm_opt_case,
    comprehensive_spmm_test,
)
from .spmm_csr_opt_alg2 import benchmark_spmm_opt_alg2_case
from .spgemm_csr import benchmark_spgemm_case
from .sddmm_csr import benchmark_sddmm_case
from .spsm import benchmark_spsm_case


def _normalize_dtype_name(value):
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).replace("torch.", "")


def _resolve_scatter_benchmark_dtype(value_dtype, dtype_policy):
    requested_name = _normalize_dtype_name(value_dtype)
    if requested_name in ("complex32", "chalf"):
        raise TypeError(_scatter_dtype_error_message())
    effective_dtype, fallback_applied, fallback_reason = _resolve_scatter_value_dtype(
        value_dtype, dtype_policy=dtype_policy
    )
    if effective_dtype not in SUPPORTED_SCATTER_VALUE_DTYPES:
        raise TypeError(_scatter_dtype_error_message())
    return requested_name, effective_dtype, bool(fallback_applied), fallback_reason


def _build_gather_selector_csr(indices, dense_size, value_dtype):
    nnz = int(indices.numel())
    data = torch.ones(nnz, dtype=value_dtype, device=indices.device)
    selector_indices = indices.contiguous()
    indptr = torch.arange(nnz + 1, dtype=torch.int64, device=indices.device)
    return data, selector_indices, indptr, (nnz, dense_size)


def _build_scatter_selector_csr(indices, dense_size, value_dtype):
    nnz = int(indices.numel())
    data = torch.ones(nnz, dtype=value_dtype, device=indices.device)
    if nnz == 0:
        indptr = torch.zeros(dense_size + 1, dtype=torch.int64, device=indices.device)
        selector_indices = torch.empty(0, dtype=indices.dtype, device=indices.device)
        return data, selector_indices, indptr, (dense_size, 0)

    order = torch.argsort(indices.to(torch.int64), stable=True)
    sorted_rows = indices[order]
    selector_indices = torch.arange(nnz, dtype=indices.dtype, device=indices.device)[order]
    row_counts = torch.bincount(sorted_rows.to(torch.int64), minlength=dense_size)
    indptr = torch.zeros(dense_size + 1, dtype=torch.int64, device=indices.device)
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return data, selector_indices.contiguous(), indptr, (dense_size, nnz)


def benchmark_gather_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
):
    """Benchmark Triton vs PyTorch indexing vs the active sparse reference backend."""
    device = torch.device("cuda")
    dense_vector = _build_random_dense(dense_size, value_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=False)

    dense_vector, indices, kernel_indices = _prepare_inputs(dense_vector, indices)
    expected = dense_vector[indices]

    pytorch_op = lambda: dense_vector[indices]
    triton_op = lambda: _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if nnz > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    sparse_ref_backend = None
    if run_cusparse:
        if _is_rocm_runtime():
            try:
                cusparse_values, cusparse_ms = benchmark_hipsparse_gather(
                    dense_vector,
                    indices,
                    warmup=warmup,
                    iters=iters,
                )
                sparse_ref_backend = "hipsparse"
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if nnz > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)
        else:
            selector_data, selector_indices, selector_indptr, selector_shape = (
                _build_gather_selector_csr(indices, dense_vector.numel(), dense_vector.dtype)
            )
            try:
                sparse_ref = _benchmark_spmv_csr_sparse_ref(
                    selector_data,
                    selector_indices,
                    selector_indptr,
                    dense_vector,
                    selector_shape,
                    warmup=warmup,
                    iters=iters,
                    op="non",
                )
                sparse_ref_backend = sparse_ref["backend"]
                if sparse_ref_backend is not None:
                    cusparse_values = sparse_ref["values"]
                    cusparse_ms = sparse_ref["ms"]
                    cusparse_match = torch.allclose(
                        cusparse_values, expected, atol=atol, rtol=rtol
                    )
                    cusparse_max_error = (
                        float(torch.max(torch.abs(cusparse_values - expected)).item())
                        if nnz > 0
                        else 0.0
                    )
                else:
                    cusparse_reason = sparse_ref["reason"]
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
            "sparse_ref_backend": sparse_ref_backend,
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_scatter_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
    unique_indices=True,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    """Benchmark Triton scatter vs PyTorch index_copy vs the active sparse reference backend."""
    device = torch.device("cuda")
    requested_value_dtype, requested_effective_dtype, fallback_applied, fallback_reason = (
        _resolve_scatter_benchmark_dtype(value_dtype, dtype_policy)
    )
    sparse_values = _build_random_dense(nnz, requested_effective_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=unique_indices)

    sparse_values, indices, kernel_indices, dense_size, prep_meta = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=None,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    effective_value_dtype = prep_meta["effective_value_dtype"]
    fallback_applied = fallback_applied or prep_meta["fallback_applied"]
    fallback_reason = fallback_reason or prep_meta["fallback_reason"]

    base_out = _build_random_dense(dense_size, sparse_values.dtype, device)
    expected = _pytorch_scatter_impl(
        sparse_values,
        indices,
        dense_size,
        out=base_out.clone(),
        reset_output=reset_output,
    )

    pytorch_out = base_out.clone()
    triton_probe_out = base_out.clone()
    _, triton_index_meta = _triton_scatter_impl(
        sparse_values,
        kernel_indices,
        dense_size=dense_size,
        out=triton_probe_out,
        block_size=block_size,
        reset_output=reset_output,
        index_fallback_policy=index_fallback_policy,
        return_metadata=True,
    )
    effective_kernel_indices = kernel_indices
    if triton_index_meta["index_fallback_applied"] and kernel_indices.dtype == torch.int64:
        effective_kernel_indices = kernel_indices.to(torch.int32)
    triton_out = base_out.clone()

    pytorch_op = lambda: _pytorch_scatter_impl(
        sparse_values,
        indices,
        dense_size,
        out=pytorch_out,
        reset_output=reset_output,
    )
    triton_op = lambda: _triton_scatter_impl(
        sparse_values,
        effective_kernel_indices,
        dense_size=dense_size,
        out=triton_out,
        block_size=block_size,
        reset_output=reset_output,
        index_fallback_policy="strict",
    )

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(effective_value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if dense_size > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    sparse_ref_backend = None
    if run_cusparse:
        if not unique_indices:
            skip_reason = "scatter sparse reference only matches unique_indices=True semantics"
        elif _is_rocm_runtime():
            skip_reason = None
            try:
                cusparse_out = base_out.clone()
                cusparse_values, cusparse_ms = benchmark_hipsparse_scatter(
                    sparse_values,
                    indices,
                    dense_size=dense_size,
                    out=cusparse_out,
                    reset_output=reset_output,
                    dtype_policy="strict",
                    warmup=warmup,
                    iters=iters,
                )
                sparse_ref_backend = "hipsparse"
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if dense_size > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)
        elif not reset_output:
            skip_reason = "cuSPARSE scatter baseline only matches reset_output=True semantics"
        else:
            selector_data, selector_indices, selector_indptr, selector_shape = (
                _build_scatter_selector_csr(indices, dense_size, sparse_values.dtype)
            )
            skip_reason = None
        if skip_reason:
            cusparse_reason = skip_reason
        elif not _is_rocm_runtime():
            try:
                sparse_ref = _benchmark_spmv_csr_sparse_ref(
                    selector_data,
                    selector_indices,
                    selector_indptr,
                    sparse_values,
                    selector_shape,
                    warmup=warmup,
                    iters=iters,
                    op="non",
                )
                sparse_ref_backend = sparse_ref["backend"]
                if sparse_ref_backend is not None:
                    cusparse_values = sparse_ref["values"]
                    cusparse_ms = sparse_ref["ms"]
                    cusparse_match = torch.allclose(
                        cusparse_values, expected, atol=atol, rtol=rtol
                    )
                    cusparse_max_error = (
                        float(torch.max(torch.abs(cusparse_values - expected)).item())
                        if dense_size > 0
                        else 0.0
                    )
                else:
                    cusparse_reason = sparse_ref["reason"]
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": requested_value_dtype,
            "effective_value_dtype": str(effective_value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "unique_indices": unique_indices,
            "reset_output": bool(reset_output),
            "dtype_policy": str(dtype_policy).lower(),
            "fallback_applied": bool(fallback_applied),
            "index_fallback_policy": str(index_fallback_policy).lower(),
            "kernel_index_dtype": triton_index_meta["kernel_index_dtype"],
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
            "sparse_ref_backend": sparse_ref_backend,
            "fallback_reason": fallback_reason,
            "index_fallback_applied": bool(triton_index_meta["index_fallback_applied"]),
            "index_fallback_reason": triton_index_meta["index_fallback_reason"],
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_spmv_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_nnz=256,
    max_segments=None,
    run_cusparse=True,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    """Benchmark Triton CSR SpMV vs the active sparse reference backend."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    op_code = _normalize_spmv_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    transpose = _spmv_op_transposes(op_code)
    op_name = _spmv_op_to_name(op_code)
    x_size = n_rows if transpose else n_cols
    y_size = n_cols if transpose else n_rows
    x = _build_random_dense(x_size, value_dtype, device)
    shape = (n_rows, n_cols)
    prepared = prepare_spmv_csr(
        data,
        indices,
        indptr,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )
    triton_op = lambda: flagsparse_spmv_csr(
        x=x,
        prepared=prepared,
        return_time=False,
    )
    triton_y, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)
    expected, ref_meta = _spmv_csr_reference(
        data,
        indices,
        indptr,
        x,
        shape,
        out_dtype=value_dtype,
        op=op_code,
        return_metadata=True,
    )
    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_y, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_y - expected)).item())
        if y_size > 0
        else 0.0
    )
    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        sparse_ref = _benchmark_spmv_csr_sparse_ref(
            data,
            indices,
            indptr,
            x,
            shape,
            warmup=warmup,
            iters=iters,
            op=op_code,
        )
        if sparse_ref["backend"] is not None:
            cusparse_values = sparse_ref["values"]
            cusparse_ms = sparse_ref["ms"]
            cusparse_match = torch.allclose(
                cusparse_values, expected, atol=atol, rtol=rtol
            )
            cusparse_max_error = (
                float(torch.max(torch.abs(cusparse_values - expected)).item())
                if y_size > 0
                else 0.0
            )
        else:
            cusparse_reason = sparse_ref["reason"]
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )
    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "op": op_name,
            "transpose": transpose,
            "index_fallback_policy": str(index_fallback_policy).lower(),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_reference": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
            "sparse_ref_backend": ref_meta["backend"],
            "sparse_ref_fallback_reason": ref_meta["fallback_reason"],
        },
        "samples": {"triton": triton_y, "reference": expected},
    }


def benchmark_performance(
    dense_size=65536,
    nnz=4096,
    dtype=torch.float32,
    index_dtype=torch.int32,
):
    """Backward-compatible benchmark entry."""
    result = benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=10,
        iters=100,
        run_cusparse=False,
    )
    return {
        "triton_time_ms": result["performance"]["triton_ms"],
        "results_match": result["verification"]["triton_match_pytorch"],
        "dtype": str(dtype),
        "index_dtype": str(index_dtype),
        "dense_size": dense_size,
        "nnz": nnz,
    }


def comprehensive_gather_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
):
    """Full test entry for one configuration."""
    return benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
    )


def comprehensive_scatter_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
    unique_indices=True,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    """Full scatter test entry for one configuration."""
    return benchmark_scatter_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
        unique_indices=unique_indices,
        reset_output=reset_output,
        dtype_policy=dtype_policy,
        index_fallback_policy=index_fallback_policy,
    )


def comprehensive_spsm_test(
    fmt="csr",
    n_rows=1024,
    n_rhs=32,
    nnz=8192,
    dtype=torch.float32,
    index_dtype=torch.int32,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    warmup=10,
    iters=50,
):
    """Full SpSM benchmark entry for one configuration."""
    return benchmark_spsm_case(
        fmt=fmt,
        n_rows=n_rows,
        n_rhs=n_rhs,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        alpha=alpha,
        lower=lower,
        unit_diagonal=unit_diagonal,
        warmup=warmup,
        iters=iters,
    )
