import pytest
import torch

from flagsparse import flagsparse_spmv_coo

from tests.pytest.param_shapes import (
    SPMV_COO_DTYPES,
    SPMV_COO_DTYPE_IDS,
    SPMV_MN_SHAPES,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_coo_mn(M, N, dtype, device):
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, N, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_coo()


def _tol(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-10, 1e-8


@pytest.mark.spmv_coo
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("dtype", SPMV_COO_DTYPES, ids=SPMV_COO_DTYPE_IDS)
def test_spmv_coo_matches_torch(M, N, dtype):
    device = torch.device("cuda")
    Asp = _random_coo_mn(M, N, dtype, device)
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].contiguous()
    col = indices[1].contiguous()
    x = torch.randn(N, dtype=dtype, device=device)
    ref = torch.sparse.mm(Asp, x.unsqueeze(1)).squeeze(1)
    out = flagsparse_spmv_coo(data, row, col, x, shape=(M, N))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
