import pytest
import torch

from flagsparse import flagsparse_spmm_coo

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import MNK_SHAPES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _value_dtype_cases():
    cases = [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]
    return [(name, dtype) for name, dtype in cases if dtype is not None]


def _random_dense(shape, dtype, device):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _random_coo_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_coo().coalesce()


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spmm_coo
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize(
    "dtype_name,dtype",
    _value_dtype_cases(),
    ids=[name for name, _dtype in _value_dtype_cases()],
)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spmm_coo_matches_dense_reference(M, N, K, dtype_name, dtype, index_dtype):
    device = torch.device("cuda")
    Asp = _random_coo_mk(M, K, dtype, device)
    indices = Asp.indices()
    data = Asp.values()
    row = indices[0].to(index_dtype).contiguous()
    col = indices[1].to(index_dtype).contiguous()
    B = _random_dense((K, N), dtype, device)
    ref_dtype = _reference_dtype(dtype)
    ref = (Asp.to_dense().to(ref_dtype) @ B.to(ref_dtype)).to(dtype)
    out = flagsparse_spmm_coo(data, row, col, B, (M, K))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref_dtype), ref.to(ref_dtype), rtol=rtol, atol=atol)


@pytest.mark.spmm_coo
def test_spmm_coo_does_not_accept_op_argument():
    device = torch.device("cuda")
    Asp = _random_coo_mk(8, 12, torch.float32, device)
    indices = Asp.indices()
    B = torch.randn(12, 4, dtype=torch.float32, device=device)
    with pytest.raises(TypeError, match="unexpected keyword argument 'op'"):
        flagsparse_spmm_coo(
            Asp.values(),
            indices[0].to(torch.int32).contiguous(),
            indices[1].to(torch.int32).contiguous(),
            B,
            (8, 12),
            op="trans",
        )
