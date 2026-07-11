import importlib

import pytest
import torch

from flagsparse import flagsparse_spmv_bsr, prepare_spmv_bsr
from tests.pytest.accuracy_utils import close_tolerances


spmv_bsr_mod = importlib.import_module("flagsparse.sparse_operations.spmv_bsr")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BSR_MN_SHAPES = ((7, 8), (12, 9), (16, 32), (64, 96))


def _value_dtype_cases():
    cases = [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]
    return [(name, dtype) for name, dtype in cases if dtype is not None]


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        return torch.complex(
            torch.randn(shape, dtype=torch.float32, device=device),
            torch.randn(shape, dtype=torch.float32, device=device),
        )
    if dtype == torch.complex128:
        return torch.complex(
            torch.randn(shape, dtype=torch.float64, device=device),
            torch.randn(shape, dtype=torch.float64, device=device),
        )
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _dense_to_bsr(dense, index_dtype, block_dim):
    device = dense.device
    M, N = dense.shape
    n_block_rows = (M + block_dim - 1) // block_dim
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    blocks = {}
    for row, col in zip(rows.tolist(), cols.tolist()):
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            torch.zeros((block_dim, block_dim), dtype=dense.dtype, device=device),
        )
        block[inner_row, inner_col] = dense[row, col]
    row_blocks = [[] for _ in range(n_block_rows)]
    for key in sorted(blocks):
        row_blocks[key[0]].append(key)
    data = []
    indices = []
    indptr = [0]
    for keys in row_blocks:
        for key in keys:
            indices.append(key[1])
            data.append(blocks[key])
        indptr.append(len(indices))
    if data:
        data_tensor = torch.stack(data).contiguous()
    else:
        data_tensor = torch.empty((0, block_dim, block_dim), dtype=dense.dtype, device=device)
    return (
        data_tensor,
        torch.tensor(indices, dtype=index_dtype, device=device),
        torch.tensor(indptr, dtype=index_dtype, device=device),
    )


def _random_bsr_mn(M, N, dtype, index_dtype, block_dim, device):
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, N), dtype, device),
        torch.zeros((), dtype=dtype, device=device),
    )
    data, indices, indptr = _dense_to_bsr(dense, index_dtype, block_dim)
    return data, indices, indptr, dense


def _make_x(length, dtype, device):
    return _random_values((length,), dtype, device)


def _padded_rows(M, block_dim):
    return ((int(M) + int(block_dim) - 1) // int(block_dim)) * int(block_dim)


def _padded_cols(N, block_dim):
    return ((int(N) + int(block_dim) - 1) // int(block_dim)) * int(block_dim)


def _assert_close(actual, expected, dtype):
    rtol, atol = close_tolerances(dtype)
    ref_dtype = _reference_dtype(dtype)
    assert torch.allclose(
        actual.to(ref_dtype), expected.to(ref_dtype), rtol=rtol, atol=atol
    )


@pytest.mark.spmv_bsr
@pytest.mark.parametrize("M, N", BSR_MN_SHAPES)
@pytest.mark.parametrize(
    "name,dtype", _value_dtype_cases(), ids=[c[0] for c in _value_dtype_cases()]
)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("block_dim", [2, 4], ids=["block2", "block4"])
def test_spmv_bsr_matches_dense_reference(M, N, name, dtype, index_dtype, block_dim):
    device = torch.device("cuda")
    data, indices, indptr, dense = _random_bsr_mn(
        M, N, dtype, index_dtype, block_dim, device
    )
    x = _make_x(N, dtype, device)
    ref_dtype = _reference_dtype(dtype)
    ref = (dense.to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
    out = flagsparse_spmv_bsr(
        data,
        indices,
        indptr,
        x,
        shape=(M, N),
        block_dim=block_dim,
        index_fallback_policy="auto",
    )
    assert out.numel() == _padded_rows(M, block_dim)
    _assert_close(out[:M], ref, dtype)


@pytest.mark.spmv_bsr
def test_spmv_bsr_prepared_path_matches_dense_reference():
    device = torch.device("cuda")
    M, N = 7, 8
    dtype = torch.complex64
    block_dim = 4
    data, indices, indptr, dense = _random_bsr_mn(
        M, N, dtype, torch.int32, block_dim, device
    )
    prepared = prepare_spmv_bsr(data, indices, indptr, (M, N), block_dim, op="non")
    x = _make_x(N, dtype, device)
    ref = (dense.to(torch.complex128) @ x.to(torch.complex128)).to(dtype)
    out = flagsparse_spmv_bsr(x=x, prepared=prepared)
    assert out.numel() == _padded_rows(M, block_dim)
    _assert_close(out[:M], ref, dtype)


@pytest.mark.spmv_bsr
@pytest.mark.parametrize("op", ["trans", "conj"], ids=["trans", "conj"])
def test_spmv_bsr_unsupported_ops_are_rejected(op):
    device = torch.device("cuda")
    data, indices, indptr, _dense = _random_bsr_mn(
        8, 12, torch.float32, torch.int32, 2, device
    )
    x = torch.randn(8, dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="only supports op='non'"):
        flagsparse_spmv_bsr(data, indices, indptr, x, shape=(8, 12), block_dim=2, op=op)


@pytest.mark.spmv_bsr
def test_spmv_bsr_x_length_mismatch_rejected():
    device = torch.device("cuda")
    data, indices, indptr, _dense = _random_bsr_mn(
        8, 12, torch.float32, torch.int32, 2, device
    )
    prepared = prepare_spmv_bsr(data, indices, indptr, (8, 12), 2)
    x = torch.randn(8, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="x length must be 12"):
        flagsparse_spmv_bsr(x=x, prepared=prepared)


@pytest.mark.spmv_bsr
def test_spmv_bsr_int64_auto_fallback_to_int32(monkeypatch):
    device = torch.device("cuda")
    data, indices, indptr, dense = _random_bsr_mn(
        12, 9, torch.float32, torch.int64, 2, device
    )
    x = torch.randn(9, dtype=torch.float32, device=device)
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    state = {"forced_once": False}
    original = spmv_bsr_mod._triton_spmv_bsr_kernel

    def fail_int64_once(prepared, x_in, op_code):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, x_in, op_code)

    monkeypatch.setattr(spmv_bsr_mod, "_triton_spmv_bsr_kernel", fail_int64_once)
    out = flagsparse_spmv_bsr(
        data,
        indices,
        indptr,
        x,
        shape=(12, 9),
        block_dim=2,
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    assert out.numel() == _padded_rows(12, 2)
    _assert_close(out[:12], ref.to(torch.float32), torch.float32)


@pytest.mark.spmv_bsr
def test_spmv_bsr_int64_strict_no_fallback(monkeypatch):
    device = torch.device("cuda")
    data, indices, indptr, _dense = _random_bsr_mn(
        12, 10, torch.float32, torch.int64, 2, device
    )
    x = torch.randn(10, dtype=torch.float32, device=device)
    original = spmv_bsr_mod._triton_spmv_bsr_kernel

    def fail_int64(prepared, x_in, op_code):
        if prepared.kernel_indices.dtype == torch.int64:
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, x_in, op_code)

    monkeypatch.setattr(spmv_bsr_mod, "_triton_spmv_bsr_kernel", fail_int64)
    with pytest.raises(RuntimeError, match="forced int64 launch failure"):
        flagsparse_spmv_bsr(
            data,
            indices,
            indptr,
            x,
            shape=(12, 10),
            block_dim=2,
            index_fallback_policy="strict",
        )


@pytest.mark.spmv_bsr
def test_spmv_bsr_non_divisible_shape_matches_dense_reference():
    device = torch.device("cuda")
    M, N = 7, 8
    data, indices, indptr, dense = _random_bsr_mn(
        M, N, torch.float32, torch.int32, 4, device
    )
    x = torch.randn(8, dtype=torch.float32, device=device)
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    prepared = prepare_spmv_bsr(data, indices, indptr, (M, N), 4)
    out = flagsparse_spmv_bsr(x=x, prepared=prepared)
    assert out.numel() == _padded_rows(M, 4)
    _assert_close(out[:M], ref.to(torch.float32), torch.float32)


@pytest.mark.spmv_bsr
def test_spmv_bsr_accepts_logical_or_padded_x():
    device = torch.device("cuda")
    M, N = 7, 9
    block_dim = 4
    data, indices, indptr, dense = _random_bsr_mn(
        M, N, torch.float32, torch.int32, block_dim, device
    )
    x = torch.randn(N, dtype=torch.float32, device=device)
    x_padded = torch.zeros(_padded_cols(N, block_dim), dtype=torch.float32, device=device)
    x_padded[:N].copy_(x)
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    prepared = prepare_spmv_bsr(data, indices, indptr, (M, N), block_dim)
    out_logical_x = flagsparse_spmv_bsr(x=x, prepared=prepared)
    out_padded_x = flagsparse_spmv_bsr(x=x_padded, prepared=prepared)
    assert out_logical_x.numel() == _padded_rows(M, block_dim)
    assert out_padded_x.numel() == _padded_rows(M, block_dim)
    _assert_close(out_logical_x, out_padded_x, torch.float32)
    _assert_close(out_logical_x[:M], ref.to(torch.float32), torch.float32)
