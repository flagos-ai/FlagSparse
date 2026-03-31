import pytest
import torch

from flagsparse import flagsparse_spmm_csr

from tests.pytest.param_shapes import MNK_SHAPES, SPMM_FLOAT_DTYPES, SPMM_FLOAT_DTYPE_IDS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_csr_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _tol(dtype):
    if dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-10, 1e-8


@pytest.mark.spmm_csr
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_FLOAT_DTYPES, ids=SPMM_FLOAT_DTYPE_IDS)
def test_spmm_csr_matches_torch(M, N, K, dtype):
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")
    device = torch.device("cuda")
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    if dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=indptr,
            col_indices=indices,
            values=data.double(),
            size=(M, K),
            dtype=torch.float64,
            device=device,
        )
        ref = torch.sparse.mm(Asp64, B.double()).float()
    else:
        ref = torch.sparse.mm(Asp, B)
    out = flagsparse_spmm_csr(data, indices, indptr, B, (M, K))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
