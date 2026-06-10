import pytest
import torch

from flagsparse import flagsparse_sddmm_csr, flagsparse_spgemm_csr

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import (
    SDDMM_DTYPES,
    SDDMM_DTYPE_IDS,
    SDDMM_MNK_SHAPES,
    SPGEMM_DTYPES,
    SPGEMM_DTYPE_IDS,
    SPGEMM_MNK_SHAPES,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SYNTHETIC_VALUE_SCALE = 0.125


def _random_csr(rows, cols, dtype, device, value_scale=_SYNTHETIC_VALUE_SCALE):
    denom = max(rows * cols, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(rows, cols, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = (
        torch.randn(rows, cols, dtype=dtype, device=device)
        * value_scale
        * mask.to(dtype=dtype)
    )
    return vals.to_sparse_csr()


def _csr_to_dense(data, indices, indptr, shape):
    csr = torch.sparse_csr_tensor(
        indptr,
        indices,
        data,
        size=shape,
        dtype=data.dtype,
        device=data.device,
    )
    return csr.to_dense()


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spgemm_csr
@pytest.mark.parametrize("M, N, K", SPGEMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPGEMM_DTYPES, ids=SPGEMM_DTYPE_IDS)
@pytest.mark.parametrize(
    "indptr_dtype", [torch.int32, torch.int64], ids=["ptr32", "ptr64"]
)
def test_spgemm_csr_matches_torch(M, N, K, dtype, indptr_dtype):
    device = torch.device("cuda")
    A = _random_csr(M, K, dtype, device)
    B = _random_csr(K, N, dtype, device)
    c_data, c_indices, c_indptr, c_shape = flagsparse_spgemm_csr(
        A.values(),
        A.col_indices().to(torch.int32),
        A.crow_indices().to(indptr_dtype),
        (M, K),
        B.values(),
        B.col_indices().to(torch.int32),
        B.crow_indices().to(indptr_dtype),
        (K, N),
    )
    got = _csr_to_dense(c_data, c_indices, c_indptr, c_shape)
    ref = torch.sparse.mm(A, B.to_dense())
    rtol, atol = _tol(dtype)
    assert torch.allclose(got, ref, rtol=rtol, atol=atol)


@pytest.mark.spgemm_csr
def test_spgemm_csr_rejects_unsupported_index_and_value_dtypes():
    device = torch.device("cuda")
    A = _random_csr(8, 10, torch.float32, device)
    B = _random_csr(10, 6, torch.float32, device)
    with pytest.raises(TypeError, match="a_indices dtype must be torch.int32"):
        flagsparse_spgemm_csr(
            A.values(),
            A.col_indices().to(torch.int64),
            A.crow_indices(),
            (8, 10),
            B.values(),
            B.col_indices().to(torch.int32),
            B.crow_indices(),
            (10, 6),
        )

    A_complex = _random_csr(8, 10, torch.complex64, device)
    with pytest.raises(
        TypeError, match="a_data dtype must be torch.float32 or torch.float64"
    ):
        flagsparse_spgemm_csr(
            A_complex.values(),
            A_complex.col_indices().to(torch.int32),
            A_complex.crow_indices(),
            (8, 10),
            B.values(),
            B.col_indices().to(torch.int32),
            B.crow_indices(),
            (10, 6),
        )


@pytest.mark.sddmm_csr
@pytest.mark.parametrize("M, N, K", SDDMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SDDMM_DTYPES, ids=SDDMM_DTYPE_IDS)
@pytest.mark.parametrize(
    "indptr_dtype", [torch.int32, torch.int64], ids=["ptr32", "ptr64"]
)
def test_sddmm_csr_matches_sampled_dense_reference(M, N, K, dtype, indptr_dtype):
    device = torch.device("cuda")
    pattern = _random_csr(M, N, dtype, device)
    indices = pattern.col_indices().to(torch.int32)
    indptr = pattern.crow_indices().to(indptr_dtype)
    data = pattern.values()
    x = torch.randn(M, K, dtype=dtype, device=device) * _SYNTHETIC_VALUE_SCALE
    y = torch.randn(N, K, dtype=dtype, device=device) * _SYNTHETIC_VALUE_SCALE
    alpha = 1.25
    beta = 0.5

    got = flagsparse_sddmm_csr(
        data=data,
        indices=indices,
        indptr=indptr,
        x=x,
        y=y,
        shape=(M, N),
        alpha=alpha,
        beta=beta,
    )
    row_ids = torch.repeat_interleave(
        torch.arange(M, dtype=torch.int64, device=device),
        indptr[1:] - indptr[:-1],
    )
    ref = (
        alpha * torch.sum(x[row_ids] * y[indices.to(torch.int64)], dim=1) + beta * data
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(got, ref, rtol=rtol, atol=atol)


@pytest.mark.sddmm_csr
def test_sddmm_csr_rejects_unsupported_index_and_value_dtypes():
    device = torch.device("cuda")
    pattern = _random_csr(8, 10, torch.float32, device)
    data = pattern.values()
    indptr = pattern.crow_indices()
    x = torch.randn(8, 4, dtype=torch.float32, device=device)
    y = torch.randn(10, 4, dtype=torch.float32, device=device)
    with pytest.raises(TypeError, match="indices dtype must be torch.int32"):
        flagsparse_sddmm_csr(
            data=data,
            indices=pattern.col_indices().to(torch.int64),
            indptr=indptr,
            x=x,
            y=y,
            shape=(8, 10),
        )

    data_complex = data.to(torch.complex64)
    with pytest.raises(
        TypeError, match="x dtype must be torch.float32 or torch.float64"
    ):
        flagsparse_sddmm_csr(
            data=data_complex,
            indices=pattern.col_indices().to(torch.int32),
            indptr=indptr,
            x=x.to(torch.complex64),
            y=y.to(torch.complex64),
            shape=(8, 10),
        )
