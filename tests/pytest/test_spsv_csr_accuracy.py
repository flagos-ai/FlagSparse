import pytest
import torch

from flagsparse import flagsparse_spsv_csr

from tests.pytest.param_shapes import SPSV_N


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_spsv_csr_lower_matches_dense(n, dtype):
    device = torch.device("cuda")
    base = torch.tril(torch.randn(n, n, dtype=dtype, device=device))
    eye = torch.eye(n, dtype=dtype, device=device)
    A = base + eye * (float(n) * 0.5 + 2.0)
    b = torch.randn(n, dtype=dtype, device=device)
    # PyTorch 2.x requires B with rank >= 2 for solve_triangular.
    x_ref = torch.linalg.solve_triangular(
        A, b.unsqueeze(-1), upper=False
    ).squeeze(-1)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices().to(torch.int64)
    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    rtol = 1e-4 if dtype == torch.float32 else 1e-10
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)
