# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flagsparse import (
    flagsparse_spmm_csr,
    flagsparse_spmm_csr_opt,
    prepare_spmm_csr_opt,
)
from flagsparse.sparse_operations import _common as common
from tests import reference_utils

from tests.pytest.param_shapes import (
    MNK_SHAPES,
    SPMM_FLOAT_DTYPES,
    SPMM_FLOAT_DTYPE_IDS,
    SPMM_OPT_DTYPES,
    SPMM_OPT_DTYPE_IDS,
)
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)

_SYNTHETIC_VALUE_SCALE = 0.125


def _random_csr_mk(M, K, dtype, device, value_scale=_SYNTHETIC_VALUE_SCALE):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = (
        torch.randn(M, K, dtype=dtype, device=device)
        * value_scale
        * mask.to(dtype=dtype)
    )
    return vals.to_sparse_csr()


def _plain_sparse_values(sparse_tensor):
    return sparse_tensor.values().clone()


def _csr_arrays(Asp, device, index_dtype=None):
    """Operator inputs on ``device`` from a CPU-side sparse CSR tensor.

    ``Asp`` itself stays on CPU: torch.sparse has no working matmul on MUSA
    (``aten::empty.memory_format`` is unregistered for ``SparseCsrmusa``), so the
    references in this file must be evaluated there.  See ``golden_device()``.
    """
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    if index_dtype is not None:
        indices = indices.to(index_dtype)
        indptr = indptr.to(index_dtype)
    return data.to(device), indices.to(device), indptr.to(device)


def _random_dense(shape, dtype, device, value_scale=_SYNTHETIC_VALUE_SCALE):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device) * value_scale
    if dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device) * value_scale
        imag = torch.randn(shape, dtype=torch.float32, device=device) * value_scale
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device) * value_scale
        imag = torch.randn(shape, dtype=torch.float64, device=device) * value_scale
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


def _apply_dense_op(dense, op):
    if op == "non":
        return dense
    if op == "trans":
        return dense.t()
    if op == "conj":
        return dense.conj().t()
    raise ValueError(f"unsupported op: {op}")


def _tol(dtype):
    return close_tolerances(dtype)


SPMM_OP_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
SPMM_OP_DTYPE_IDS = ("float32", "float64", "complex64", "complex128")


@pytest.mark.spmm_csr
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_FLOAT_DTYPES, ids=SPMM_FLOAT_DTYPE_IDS)
def test_spmm_csr_matches_torch(M, N, K, dtype):
    device = accelerator_device()
    golden = golden_device()
    Asp = _random_csr_mk(M, K, dtype, golden)
    B = _random_dense((K, N), dtype, golden)
    # Reference on CPU. This also subsumes the former MACA special case here: that
    # branch existed because the fp32 CSR reference ran on the card with int64
    # indices, which is exactly what no longer happens.
    if common._use_scipy_accuracy_reference():
        ref_dtype = _reference_dtype(dtype)
        matrix = reference_utils.scipy_csr(
            _plain_sparse_values(Asp), Asp.col_indices(), Asp.crow_indices(), (M, K), ref_dtype
        )
        ref = reference_utils.as_torch(
            reference_utils.spmm(matrix, B, ref_dtype), ref_dtype, golden
        ).to(dtype)
    elif dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=Asp.crow_indices(),
            col_indices=Asp.col_indices(),
            values=_plain_sparse_values(Asp).double(),
            size=(M, K),
            dtype=torch.float64,
            device=golden,
        )
        ref = torch.sparse.mm(Asp64, B.double()).float()
    else:
        ref = torch.sparse.mm(Asp, B)
    data, indices, indptr = _csr_arrays(Asp, device)
    out = flagsparse_spmm_csr(data, indices, indptr, B.to(device), (M, K))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spmm_csr
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_OP_DTYPES, ids=SPMM_OP_DTYPE_IDS)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmm_csr_op_matches_dense_reference(M, N, K, dtype, index_dtype, op):
    device = accelerator_device()
    golden = golden_device()
    Asp = _random_csr_mk(M, K, dtype, golden)
    data, indices, indptr = _csr_arrays(Asp, device, index_dtype)
    dense = Asp.to_dense()
    effective_cols = M if op in ("trans", "conj") else K
    B = _random_dense((effective_cols, N), dtype, golden)
    ref_dtype = _reference_dtype(dtype)
    if common._use_scipy_accuracy_reference():
        matrix = reference_utils.scipy_csr(
            _plain_sparse_values(Asp), Asp.col_indices(), Asp.crow_indices(), (M, K), ref_dtype
        )
        ref = reference_utils.as_torch(
            reference_utils.spmm(matrix, B, ref_dtype, op=op), ref_dtype, golden
        ).to(dtype)
    else:
        ref = (_apply_dense_op(dense, op).to(ref_dtype) @ B.to(ref_dtype)).to(dtype)
    out = flagsparse_spmm_csr(data, indices, indptr, B.to(device), (M, K), op=op)
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spmm_csr
def test_spmm_csr_return_meta_times_transpose_path():
    device = accelerator_device()
    Asp = _random_csr_mk(8, 12, torch.float32, golden_device())
    data, indices, indptr = _csr_arrays(Asp, device, torch.int64)
    B = torch.randn(8, 4, dtype=torch.float32, device=golden_device()).to(device)
    out, elapsed_ms, meta = flagsparse_spmm_csr(
        data,
        indices,
        indptr,
        B,
        (8, 12),
        op="trans",
        return_time=True,
        return_meta=True,
    )
    assert out.shape == (12, 4)
    assert meta["op"] == "trans"
    assert meta["symbolic_ms"] >= 0.0
    assert meta["compute_ms"] >= 0.0
    assert elapsed_ms == pytest.approx(meta["op_total_ms"])
    assert meta["op_total_ms"] == pytest.approx(
        meta["symbolic_ms"] + meta["compute_ms"]
    )


@pytest.mark.spmm_csr
def test_spmm_csr_non_meta_has_zero_symbolic_time():
    device = accelerator_device()
    Asp = _random_csr_mk(8, 12, torch.float32, golden_device())
    data, indices, indptr = _csr_arrays(Asp, device)
    B = torch.randn(12, 4, dtype=torch.float32, device=golden_device()).to(device)
    out, meta = flagsparse_spmm_csr(
        data,
        indices,
        indptr,
        B,
        (8, 12),
        op="non",
        return_meta=True,
    )
    assert out.shape == (8, 4)
    assert meta["op"] == "non"
    assert meta["symbolic_ms"] == 0.0


@pytest.mark.spmm_csr
def test_spmm_csr_op_validation_errors():
    device = accelerator_device()
    Asp = _random_csr_mk(8, 12, torch.float32, golden_device())
    data, indices, indptr = _csr_arrays(Asp, device)
    B_non = torch.randn(12, 4, dtype=torch.float32, device=golden_device()).to(device)
    B_trans = torch.randn(8, 4, dtype=torch.float32, device=golden_device()).to(device)
    with pytest.raises(ValueError, match="op must be one of"):
        flagsparse_spmm_csr(data, indices, indptr, B_non, (8, 12), op="bad")
    with pytest.raises(ValueError, match="transpose conflicts with op"):
        flagsparse_spmm_csr(
            data, indices, indptr, B_non, (8, 12), op="non", transpose=True
        )
    with pytest.raises(ValueError, match="transpose conflicts with op"):
        flagsparse_spmm_csr(
            data, indices, indptr, B_trans, (8, 12), op="trans", transpose=False
        )
    with pytest.raises(ValueError, match="B.shape\\[0\\] must be n_cols=8"):
        flagsparse_spmm_csr(data, indices, indptr, B_non, (8, 12), op="trans")
    with pytest.raises(ValueError, match="out shape/dtype must match result"):
        flagsparse_spmm_csr(
            data,
            indices,
            indptr,
            B_trans,
            (8, 12),
            op="trans",
            out=torch.empty((8, 4), dtype=torch.float32, device=device),
        )


@pytest.mark.spmm_csr_opt_alg1
@pytest.mark.spmm_csr_opt
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_OPT_DTYPES, ids=SPMM_OPT_DTYPE_IDS)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spmm_csr_opt_matches_torch(M, N, K, dtype, index_dtype):
    device = accelerator_device()
    golden = golden_device()
    Asp = _random_csr_mk(M, K, dtype, golden)
    B = torch.randn(K, N, dtype=dtype, device=golden)
    if common._use_scipy_accuracy_reference():
        ref_dtype = _reference_dtype(dtype)
        matrix = reference_utils.scipy_csr(
            _plain_sparse_values(Asp), Asp.col_indices(), Asp.crow_indices(), (M, K), ref_dtype
        )
        ref = reference_utils.as_torch(
            reference_utils.spmm(matrix, B, ref_dtype), ref_dtype, golden
        ).to(dtype)
    elif dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=Asp.crow_indices(),
            col_indices=Asp.col_indices(),
            values=_plain_sparse_values(Asp).double(),
            size=(M, K),
            dtype=torch.float64,
            device=golden,
        )
        ref = torch.sparse.mm(Asp64, B.double()).float()
    else:
        ref = torch.sparse.mm(Asp, B)
    data, indices, indptr = _csr_arrays(Asp, device, index_dtype)
    prepared = prepare_spmm_csr_opt(data, indices, indptr, (M, K))
    out = flagsparse_spmm_csr_opt(B=B.to(device), prepared=prepared)
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


# Explicit migrated algorithms share this operator's marker, golden and tolerance.
_MIGRATED_ROUTES = (('csr_row_tile', 'non'), ('csr_row_tile', 'trans'), ('csr_row_tile', 'conj'), ('csr_row_kparallel', 'non'), ('csr_row_kparallel', 'trans'), ('csr_row_kparallel', 'conj'), ('csr_row_panel', 'non'), ('csr_row_panel', 'trans'), ('csr_row_panel', 'conj'), ('csr_split_nnz_reduce', 'non'), ('csr_split_nnz_reduce', 'trans'), ('csr_split_nnz_reduce', 'conj'), ('csr_adaptive_tile_split', 'non'), ('csr_adaptive_tile_split', 'trans'), ('csr_adaptive_tile_split', 'conj'))


@pytest.mark.spmm_csr
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES, ids=[f"{a}-{o}" for a, o in _MIGRATED_ROUTES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128], ids=["float16", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_csr_migrated_routes(alg, op, dtype, index_dtype, layout):
    import importlib
    from tests.pytest.conftest import QUICK_MODE

    module = importlib.import_module("flagsparse.sparse_operations.spmm_csr")
    device = accelerator_device()
    # Repeated coordinates, unsorted inner indices, an empty row/column and tile tails.
    rows = torch.tensor([0, 4, 0, 6, 4, 1, 6, 0, 4], dtype=torch.int64)
    cols = torch.tensor([2, 0, 2, 3, 1, 4, 0, 1, 1], dtype=torch.int64)
    real = torch.tensor([1, -2, 3, 0, -1, 2, 1, -3, 4], dtype=torch.float64) / 8
    values = (torch.complex(real, real.flip(0)) if dtype.is_complex else real).to(dtype)
    shape = (9, 7)
    reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
    dense = torch.zeros(shape, dtype=reference_dtype)
    dense.index_put_((rows, cols), values.to(reference_dtype), accumulate=True)
    order = torch.argsort(rows, stable=True)
    values = values[order]
    first = cols[order].to(device=device, dtype=index_dtype)
    counts = torch.bincount(rows, minlength=shape[0])
    ptr_dtype = torch.int64 if index_dtype == torch.int32 else torch.int32
    second = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).to(device=device, dtype=ptr_dtype)
    data = values.to(device)
    input_rows = shape[1] if op == "non" else shape[0]
    n = 7 if QUICK_MODE else 33
    operand = (torch.arange(input_rows * n, dtype=torch.float64).reshape(input_rows, n) % 11 - 5) / 8
    if dtype.is_complex:
        operand = torch.complex(operand, operand.flip(0) / 2)
    operand = operand.to(dtype)
    expected_matrix = dense if op == "non" else dense.T.conj() if op == "conj" else dense.T
    expected = (expected_matrix @ operand.to(reference_dtype)).to(dtype)
    operand = operand.to(device)
    if layout == "col":
        operand = operand.T.contiguous().T
    try:
        prepared = module.prepare_spmm_csr_route(data, first, second, shape, op=op, alg=alg)
        backing = torch.full((expected.shape[0], expected.shape[1] * 2), float("nan"), dtype=dtype, device=device)
        out = backing[:, ::2]
        result = module.flagsparse_spmm_csr_run(prepared, operand, out=out)
        direct = module.flagsparse_spmm_csr(data, first, second, operand.T, shape, op=op, op_b="trans", out_dtype=dtype, alg=alg)
    except NotImplementedError as exc:
        pytest.skip(f"backend capability unavailable: {exc}")
    assert result is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(direct.cpu(), expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="op"):
        module.flagsparse_spmm_csr_run(prepared, operand, op="trans" if op == "non" else "non")


@pytest.mark.spmm_csr
@pytest.mark.parametrize("dtype", [torch.float16, torch.complex128])
def test_spmm_csr_split_bounded_workspace(dtype):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmm_csr")
    device = accelerator_device()
    counts = torch.tensor([0, 1, 73, 8, 0], dtype=torch.int64)
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    index = torch.arange(int(ptr[-1])) % 19
    data = torch.ones(index.numel(), dtype=dtype)
    if dtype.is_complex:
        data = data * (1 + 0.5j)
    outer = torch.repeat_interleave(torch.arange(5), counts)
    golden_dtype = torch.complex128 if dtype.is_complex else torch.float64
    matrix = torch.zeros((5, 19), dtype=golden_dtype)
    matrix.index_put_((outer, index), data.to(golden_dtype), accumulate=True)
    operand = torch.ones((19, 7), dtype=golden_dtype)
    operand = operand.to(dtype)
    expected = (matrix @ operand.to(golden_dtype)).to(dtype)
    cfg = dict(segment_nnz=8, reduce_block_size=2, workspace_bytes=256, block_k=8, block_n=4)
    try:
        prepared = module.prepare_spmm_csr_route(data.to(device), index.to(device), ptr.to(device), (5, 19),
            op="non", alg="csr_split_nnz_reduce", config=cfg)
        actual, meta = module.flagsparse_spmm_csr_run(prepared, operand.to(device), return_meta=True, timing=True)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
    assert meta["workspace_peak_bytes"] <= cfg["workspace_bytes"]
    assert meta["segment_batches"] > 1
    assert meta["reduction_levels"] >= 0
    wider = dict(cfg, workspace_bytes=65536)
    again, wider_meta = module.flagsparse_spmm_csr_run(prepared, operand.to(device), config=wider, return_meta=True)
    torch.testing.assert_close(again.cpu(), expected, rtol=rtol, atol=atol)
    assert wider_meta["reduction_levels"] > 1
    assert meta["process_cpu_ms"] == 0
    assert meta["operator_ms"] == meta["gpu_ms"]


@pytest.mark.spmm_csr
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5), (3, 5)])
def test_spmm_csr_migrated_empty(alg, op, shape):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmm_csr")
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.float32, device=device)
    first = torch.empty(0, dtype=torch.int64, device=device)
    second = torch.zeros(shape[0] + 1, dtype=torch.int32, device=device)
    input_rows, output_rows = shape[1], shape[0]
    if op != "non":
        input_rows, output_rows = output_rows, input_rows
    operand = torch.ones((input_rows, 7), device=device)
    out = torch.full((output_rows, 7), float("nan"), device=device)
    try:
        prepared = module.prepare_spmm_csr_route(data, first, second, shape, op=op, alg=alg)
        result = module.flagsparse_spmm_csr_run(prepared, operand, out=out)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert result is out
    assert torch.equal(result.cpu(), torch.zeros_like(result.cpu()))
