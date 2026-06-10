import pytest
import torch

from flagsparse import flagsparse_spsm_coo, flagsparse_spsm_csr

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import SPSM_N_RHS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


SPSM_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
SPSM_DTYPE_IDS = ("float32", "float64", "complex64", "complex128")


def _build_triangular_dense(n, dtype, device, lower, unit_diagonal):
    base = torch.randn(n, n, dtype=dtype, device=device) * 0.02
    base = torch.tril(base) if lower else torch.triu(base)
    eye = torch.eye(n, dtype=dtype, device=device)
    if unit_diagonal:
        return base * (1 - eye) + eye
    return base + eye * (float(n) * 0.5 + 2.0)


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spsm
@pytest.mark.spsm_csr
@pytest.mark.parametrize("n, n_rhs", SPSM_N_RHS)
@pytest.mark.parametrize("dtype", SPSM_DTYPES, ids=SPSM_DTYPE_IDS)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize(
    "unit_diagonal", [False, True], ids=["explicit_diag", "unit_diag"]
)
def test_spsm_csr_matches_dense(n, n_rhs, dtype, lower, unit_diagonal):
    device = torch.device("cuda")
    A = _build_triangular_dense(n, dtype, device, lower, unit_diagonal)
    B = torch.randn(n, n_rhs, dtype=dtype, device=device)
    ref = torch.linalg.solve_triangular(
        A,
        B,
        upper=not lower,
        unitriangular=unit_diagonal,
    )
    Acsr = A.to_sparse_csr()
    out = flagsparse_spsm_csr(
        Acsr.values(),
        Acsr.col_indices().to(torch.int32),
        Acsr.crow_indices().to(torch.int32),
        B,
        (n, n),
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spsm
@pytest.mark.spsm_coo
@pytest.mark.parametrize("n, n_rhs", SPSM_N_RHS)
@pytest.mark.parametrize("dtype", SPSM_DTYPES, ids=SPSM_DTYPE_IDS)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize(
    "unit_diagonal", [False, True], ids=["explicit_diag", "unit_diag"]
)
def test_spsm_coo_matches_dense(n, n_rhs, dtype, lower, unit_diagonal):
    device = torch.device("cuda")
    A = _build_triangular_dense(n, dtype, device, lower, unit_diagonal)
    B = torch.randn(n, n_rhs, dtype=dtype, device=device)
    ref = torch.linalg.solve_triangular(
        A,
        B,
        upper=not lower,
        unitriangular=unit_diagonal,
    )
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    out = flagsparse_spsm_coo(
        Acoo.values(),
        indices[0].to(torch.int32).contiguous(),
        indices[1].to(torch.int32).contiguous(),
        B,
        (n, n),
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spsm
@pytest.mark.spsm_csr
def test_spsm_csr_rejects_unsupported_index_dtype():
    device = torch.device("cuda")
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, device, lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=device)
    Acsr = A.to_sparse_csr()
    with pytest.raises(TypeError, match="indices dtype must be torch.int32"):
        flagsparse_spsm_csr(
            Acsr.values(),
            Acsr.col_indices().to(torch.int64),
            Acsr.crow_indices().to(torch.int64),
            B,
            (n, n),
        )


@pytest.mark.spsm
@pytest.mark.spsm_coo
def test_spsm_coo_rejects_unsupported_index_dtype():
    device = torch.device("cuda")
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, device, lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=device)
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    with pytest.raises(TypeError, match="row/col dtype must be torch.int32"):
        flagsparse_spsm_coo(
            Acoo.values(),
            indices[0].to(torch.int64).contiguous(),
            indices[1].to(torch.int64).contiguous(),
            B,
            (n, n),
        )


@pytest.mark.spsm
@pytest.mark.spsm_csr
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"opA": "TRANS"}, "Only op\\(A\\)=NON_TRANS is supported"),
        ({"opB": "TRANS"}, "Only op\\(B\\)=NON_TRANS is supported"),
        ({"major": "col"}, "Only row-major dense layout is supported"),
    ],
    ids=["opA_trans", "opB_trans", "col_major"],
)
def test_spsm_csr_rejects_unsupported_ops_and_layout(kwargs, match):
    device = torch.device("cuda")
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, device, lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=device)
    Acsr = A.to_sparse_csr()
    with pytest.raises(NotImplementedError, match=match):
        flagsparse_spsm_csr(
            Acsr.values(),
            Acsr.col_indices().to(torch.int32),
            Acsr.crow_indices().to(torch.int32),
            B,
            (n, n),
            **kwargs,
        )


@pytest.mark.spsm
@pytest.mark.spsm_coo
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"opA": "TRANS"}, "Only op\\(A\\)=NON_TRANS is supported"),
        ({"opB": "TRANS"}, "Only op\\(B\\)=NON_TRANS is supported"),
        ({"major": "col"}, "Only row-major dense layout is supported"),
    ],
    ids=["opA_trans", "opB_trans", "col_major"],
)
def test_spsm_coo_rejects_unsupported_ops_and_layout(kwargs, match):
    device = torch.device("cuda")
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, device, lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=device)
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    with pytest.raises(NotImplementedError, match=match):
        flagsparse_spsm_coo(
            Acoo.values(),
            indices[0].to(torch.int32).contiguous(),
            indices[1].to(torch.int32).contiguous(),
            B,
            (n, n),
            **kwargs,
        )
