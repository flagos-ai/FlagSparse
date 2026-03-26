import pytest
import torch

from flagsparse import flagsparse_gather, flagsparse_scatter

from tests.pytest.param_shapes import FLOAT_DTYPE_IDS, FLOAT_DTYPES, GATHER_SCATTER_SHAPES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.gather
@pytest.mark.parametrize("dense_size, nnz", GATHER_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_gather_matches_indexing(dense_size, nnz, dtype):
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")
    device = torch.device("cuda")
    nnz = min(nnz, dense_size)
    dense = torch.randn(dense_size, dtype=dtype, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int32)
    ref = dense[indices.to(torch.int64)]
    got = flagsparse_gather(dense, indices)
    assert torch.equal(ref, got)


@pytest.mark.scatter
@pytest.mark.parametrize("dense_size, nnz", GATHER_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_scatter_matches_index_copy(dense_size, nnz, dtype):
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")
    device = torch.device("cuda")
    nnz = min(nnz, dense_size)
    vals = torch.randn(nnz, dtype=dtype, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int32)
    dense_ref = torch.zeros(dense_size, dtype=dtype, device=device)
    dense_ref.index_copy_(0, indices.to(torch.int64), vals)
    dense = torch.zeros(dense_size, dtype=dtype, device=device)
    flagsparse_scatter(dense, indices, vals)
    assert torch.equal(dense, dense_ref)
