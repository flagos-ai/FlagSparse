import pytest
import torch

from flagsparse import (
    cusparse_spmv_gather_cupy,
    flagsparse_gather,
    flagsparse_gather_cupy,
    flagsparse_scatter,
)
from flagsparse.sparse_operations import gather_scatter as gather_scatter_ops
from flagsparse.sparse_operations._common import cp

from tests.pytest.param_shapes import FLOAT_DTYPE_IDS, FLOAT_DTYPES, GATHER_SCATTER_SHAPES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
INDEX_DTYPES = [torch.int32, torch.int64]
INDEX_DTYPE_IDS = ["int32", "int64"]
RESET_OUTPUT_CASES = [True, False]
RESET_OUTPUT_IDS = ["reset", "inplace"]

def _scatter_dtype_cases():
    cases = [(str(dtype).replace("torch.", ""), dtype) for dtype in FLOAT_DTYPES]
    cases.append(("complex64", torch.complex64))
    cases.append(("complex128", torch.complex128))
    return cases


SCATTER_DTYPE_CASES = _scatter_dtype_cases()
SCATTER_DTYPE_IDS = [name for name, _ in SCATTER_DTYPE_CASES]


def _skip_unavailable_dtype(dtype_name, dtype):
    if dtype is None:
        pytest.skip(f"{dtype_name} dtype is unavailable in this torch build")
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")


def _build_random_values(size, dtype, device):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(size, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(size, dtype=torch.float32, device=device)
        imag = torch.randn(size, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(size, dtype=torch.float64, device=device)
        imag = torch.randn(size, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported dtype in test: {dtype}")


def _to_cupy(tensor):
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _to_torch(array):
    try:
        dlpack_capsule = array.toDlpack()
    except AttributeError:
        dlpack_capsule = array.to_dlpack()
    return torch.utils.dlpack.from_dlpack(dlpack_capsule)


def _to_backend_tensor(tensor, backend):
    if backend == "torch":
        return tensor
    return _to_cupy(tensor)


def _as_torch_tensor(value):
    return value if torch.is_tensor(value) else _to_torch(value)


def _build_extra_gather_dense(layout, value_dtype, dense_size, device):
    if layout == "scalar16":
        return torch.randn(dense_size, dtype=value_dtype, device=device)
    if layout == "complex16_pair":
        return torch.randn(dense_size, 2, dtype=value_dtype, device=device)
    if layout == "complex64":
        real = torch.randn(dense_size, dtype=torch.float32, device=device)
        imag = torch.randn(dense_size, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    raise RuntimeError(f"Unknown layout: {layout}")


def _extra_gather_tolerance(value_dtype):
    if value_dtype == torch.float16:
        return 5e-3, 5e-3
    if value_dtype == torch.bfloat16:
        return 1e-2, 1e-2
    if value_dtype == torch.complex64:
        return 1e-6, 1e-5
    return 1e-6, 1e-5


EXTRA_GATHER_CASES = [
    ("scalar16", torch.float16, torch.int64),
    ("scalar16", torch.bfloat16, torch.int32),
    ("scalar16", torch.bfloat16, torch.int64),
    ("complex64", torch.complex64, torch.int32),
    ("complex64", torch.complex64, torch.int64),
]
EXTRA_GATHER_CASE_IDS = [
    "half_i64",
    "bf16_i32",
    "bf16_i64",
    "c64_i32",
    "c64_i64",
]


@pytest.mark.gather
@pytest.mark.parametrize("dense_size, nnz", GATHER_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
def test_gather_matches_indexing(dense_size, nnz, dtype, index_dtype):
    _skip_unavailable_dtype(str(dtype).replace("torch.", ""), dtype)
    device = torch.device("cuda")
    nnz = min(nnz, dense_size)
    dense = torch.randn(dense_size, dtype=dtype, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    ref = dense[indices.to(torch.int64)]
    got = flagsparse_gather(dense, indices)
    assert torch.equal(ref, got)


@pytest.mark.gather
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
def test_gather_complex128_matches_indexing(index_dtype):
    device = torch.device("cuda")
    dense_size = 4096
    nnz = 1024
    real = torch.randn(dense_size, dtype=torch.float64, device=device)
    imag = torch.randn(dense_size, dtype=torch.float64, device=device)
    dense = torch.complex(real, imag)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    ref = dense.index_select(0, indices.to(torch.int64))
    got = flagsparse_gather(dense, indices)
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-8)


@pytest.mark.scatter
@pytest.mark.parametrize("dense_size, nnz", GATHER_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype_name,dtype", SCATTER_DTYPE_CASES, ids=SCATTER_DTYPE_IDS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
@pytest.mark.parametrize("reset_output", RESET_OUTPUT_CASES, ids=RESET_OUTPUT_IDS)
def test_scatter_matches_index_copy(
    dense_size, nnz, dtype_name, dtype, index_dtype, reset_output
):
    _skip_unavailable_dtype(dtype_name, dtype)
    device = torch.device("cuda")
    nnz = min(nnz, dense_size)
    vals = _build_random_values(nnz, dtype, device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    dense_ref = _build_random_values(dense_size, dtype, device)
    dense = dense_ref.clone()
    if reset_output:
        dense_ref.zero_()
    dense_ref.index_copy_(0, indices.to(torch.int64), vals)
    flagsparse_scatter(
        dense,
        indices,
        vals,
        reset_output=reset_output,
        dtype_policy="auto",
    )
    assert torch.equal(dense, dense_ref)


@pytest.mark.scatter
def test_scatter_int64_auto_fallback_to_int32(monkeypatch):
    device = torch.device("cuda")
    dense_size = 257
    nnz = 129
    vals = torch.randn(nnz, dtype=torch.float32, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int64)
    dense = torch.randn(dense_size, dtype=torch.float32, device=device)
    dense_ref = dense.clone()
    dense_ref.zero_()
    dense_ref.index_copy_(0, indices.to(torch.int64), vals)

    original_launch = gather_scatter_ops._launch_triton_scatter_kernel
    state = {"forced_once": False}

    def fake_launch(dense_values, sparse_values, kernel_indices, nnz, block_size=1024):
        if kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original_launch(
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            block_size=block_size,
        )

    monkeypatch.setattr(gather_scatter_ops, "_launch_triton_scatter_kernel", fake_launch)

    flagsparse_scatter(
        dense,
        indices,
        vals,
        reset_output=True,
        dtype_policy="auto",
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    assert torch.equal(dense, dense_ref)


@pytest.mark.scatter
def test_scatter_int64_strict_no_fallback(monkeypatch):
    device = torch.device("cuda")
    dense_size = 257
    nnz = 129
    vals = torch.randn(nnz, dtype=torch.float32, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int64)
    dense = torch.randn(dense_size, dtype=torch.float32, device=device)

    original_launch = gather_scatter_ops._launch_triton_scatter_kernel

    def fake_launch(dense_values, sparse_values, kernel_indices, nnz, block_size=1024):
        if kernel_indices.dtype == torch.int64:
            raise RuntimeError("forced int64 launch failure")
        return original_launch(
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            block_size=block_size,
        )

    monkeypatch.setattr(gather_scatter_ops, "_launch_triton_scatter_kernel", fake_launch)

    with pytest.raises(RuntimeError, match="Triton scatter failed for index dtype"):
        flagsparse_scatter(
            dense,
            indices,
            vals,
            reset_output=True,
            dtype_policy="auto",
            index_fallback_policy="strict",
        )


@pytest.mark.gather
@pytest.mark.skipif(cp is None, reason="CuPy required")
@pytest.mark.parametrize("backend", ["torch", "cupy"])
@pytest.mark.parametrize(
    "layout,value_dtype,index_dtype",
    EXTRA_GATHER_CASES,
    ids=EXTRA_GATHER_CASE_IDS,
)
def test_gather_cupy_extra_dtypes_match_torch_and_cusparse(
    layout,
    value_dtype,
    index_dtype,
    backend,
):
    if value_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this GPU")

    device = torch.device("cuda")
    dense_size = 65536
    nnz = 4096

    dense_t = _build_extra_gather_dense(layout, value_dtype, dense_size, device)
    indices_t = torch.arange(nnz, device=device, dtype=index_dtype) * 17 % dense_size
    reference = dense_t.index_select(0, indices_t.to(torch.int64))

    dense_in = _to_backend_tensor(dense_t, backend)
    indices_in = _to_backend_tensor(indices_t, backend)

    got = flagsparse_gather_cupy(dense_in, indices_in)
    cusparse_values, _, _ = cusparse_spmv_gather_cupy(dense_in, indices_in)

    got_t = _as_torch_tensor(got)
    cusparse_t = _as_torch_tensor(cusparse_values)

    atol, rtol = _extra_gather_tolerance(value_dtype)
    assert torch.allclose(got_t, reference, atol=atol, rtol=rtol)
    assert torch.allclose(cusparse_t, reference, atol=atol, rtol=rtol)


@pytest.mark.gather
@pytest.mark.skipif(cp is None, reason="CuPy required")
def test_gather_cupy_rejects_bfloat16_pair_layout():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this GPU")

    device = torch.device("cuda")
    dense_size = 4096
    nnz = 256
    dense = torch.randn(dense_size, 2, dtype=torch.bfloat16, device=device)
    indices = torch.arange(nnz, device=device, dtype=torch.int32) * 17 % dense_size

    with pytest.raises(TypeError, match="Unsupported gather input format"):
        flagsparse_gather_cupy(dense, indices)


@pytest.mark.gather
@pytest.mark.skipif(cp is None, reason="CuPy required")
@pytest.mark.parametrize("backend", ["torch", "cupy"])
def test_gather_cupy_same_backend_out_float16_i64(backend):
    device = torch.device("cuda")
    dense_size = 65536
    nnz = 4096
    dense_t = torch.randn(dense_size, dtype=torch.float16, device=device)
    indices_t = torch.arange(nnz, device=device, dtype=torch.int64) * 17 % dense_size
    reference = dense_t.index_select(0, indices_t)

    dense_in = _to_backend_tensor(dense_t, backend)
    indices_in = _to_backend_tensor(indices_t, backend)
    out = _to_backend_tensor(torch.empty_like(reference), backend)
    result = flagsparse_gather_cupy(dense_in, indices_in, out=out)

    assert result is out
    assert torch.allclose(_as_torch_tensor(out), reference, atol=5e-3, rtol=5e-3)


@pytest.mark.gather
@pytest.mark.skipif(cp is None, reason="CuPy required")
def test_gather_cupy_int64_auto_fallback_to_int32(monkeypatch):
    device = torch.device("cuda")
    dense_size = 257
    nnz = 129
    dense = torch.randn(dense_size, dtype=torch.float16, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int64)
    ref = dense.index_select(0, indices.to(torch.int64))

    original_launch = gather_scatter_ops._launch_cupy_gather_kernel
    state = {"forced_once": False}

    def fake_launch(gather_kernel, dense_raw_cp, indices_cp, out_raw_cp, nnz):
        if not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original_launch(gather_kernel, dense_raw_cp, indices_cp, out_raw_cp, nnz)

    monkeypatch.setattr(gather_scatter_ops, "_launch_cupy_gather_kernel", fake_launch)

    got, meta = flagsparse_gather_cupy(
        dense,
        indices,
        index_fallback_policy="auto",
        return_metadata=True,
    )
    assert state["forced_once"]
    assert meta["index_fallback_applied"]
    assert meta["kernel_index_dtype"] == "int32"
    assert torch.allclose(got, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.gather
@pytest.mark.skipif(cp is None, reason="CuPy required")
def test_gather_cupy_int64_strict_no_fallback(monkeypatch):
    device = torch.device("cuda")
    dense_size = 257
    nnz = 129
    dense = torch.randn(dense_size, dtype=torch.float16, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int64)

    def fake_launch(gather_kernel, dense_raw_cp, indices_cp, out_raw_cp, nnz):
        raise RuntimeError("forced int64 launch failure")

    monkeypatch.setattr(gather_scatter_ops, "_launch_cupy_gather_kernel", fake_launch)

    with pytest.raises(RuntimeError, match="CuPy gather failed for index dtype"):
        flagsparse_gather_cupy(
            dense,
            indices,
            index_fallback_policy="strict",
        )
