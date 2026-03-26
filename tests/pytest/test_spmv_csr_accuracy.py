import pytest
import torch

from flagsparse import flagsparse_spmv_csr

from tests.pytest.param_shapes import FLOAT_DTYPE_IDS, FLOAT_DTYPES, SPMV_MN_SHAPES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_csr_mn(M, N, dtype, device):
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, N, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _tol(dtype):
    if dtype == torch.float16:
        return 5e-3, 5e-3
    if dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-10, 1e-8


@pytest.mark.spmv_csr
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_spmv_csr_matches_torch(M, N, dtype):
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")
    device = torch.device("cuda")
    Asp = _random_csr_mn(M, N, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    x = torch.randn(N, dtype=dtype, device=device)
    ref = torch.sparse.mm(Asp, x.unsqueeze(1)).squeeze(1)
    out = flagsparse_spmv_csr(data, indices, indptr, x, shape=(M, N))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
