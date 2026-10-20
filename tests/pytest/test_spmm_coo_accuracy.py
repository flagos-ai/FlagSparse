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

from flagsparse import flagsparse_spmm_coo
from flagsparse.sparse_operations import _common as common
from tests import reference_utils

from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import MNK_SHAPES

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)


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
    """Sparse COO matrix built on ``device``.

    Callers that need a reference pass ``golden_device()``: ``_reference`` runs
    ``torch.sparse.mm``, which has no working implementation on MUSA, and complex
    ``randn``/masking has no muDNN kernel either.  See ``golden_device()``.
    """
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_coo().coalesce()


def _tol(dtype):
    return close_tolerances(dtype)


def _reference(Asp, B, op):
    if op == "non":
        return torch.sparse.mm(Asp, B)
    if op == "trans":
        return torch.sparse.mm(Asp.transpose(0, 1), B)
    if op == "conj":
        return torch.sparse.mm(Asp.conj().transpose(0, 1), B)
    raise ValueError(op)


def _scipy_reference(Asp, B, op, dtype):
    indices = Asp.indices()
    matrix = reference_utils.scipy_coo(
        Asp.values(), indices[0], indices[1], tuple(Asp.shape), dtype
    )
    return reference_utils.as_torch(
        reference_utils.spmm(matrix, B, dtype, op=op), dtype, golden_device()
    )


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
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_coo_matches_dense_reference(M, N, K, dtype_name, dtype, index_dtype, op):
    device = accelerator_device()
    golden = golden_device()
    Asp = _random_coo_mk(M, K, dtype, golden)
    indices = Asp.indices()
    data = Asp.values()
    row = indices[0].to(index_dtype).contiguous()
    col = indices[1].to(index_dtype).contiguous()
    b_rows = M if op in ("trans", "conj") else K
    B = _random_dense((b_rows, N), dtype, golden)
    ref_dtype = _reference_dtype(dtype)
    ref = (
        _scipy_reference(Asp, B, op, ref_dtype)
        if common._use_scipy_accuracy_reference()
        else _reference(Asp.to(ref_dtype), B.to(ref_dtype), op)
    ).to(dtype)
    out = flagsparse_spmm_coo(
        data.to(device),
        row.to(device),
        col.to(device),
        B.to(device),
        (M, K),
        op=op,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(
        out.to(device=ref.device, dtype=ref_dtype), ref.to(ref_dtype), rtol=rtol, atol=atol
    )


@pytest.mark.spmm_coo
def test_spmm_coo_return_meta_times_transpose_path():
    device = accelerator_device()
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    col = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
    B = torch.randn(3, 4, dtype=torch.float32, device=device)
    out, elapsed_ms, meta = flagsparse_spmm_coo(
        data,
        row,
        col,
        B,
        (3, 3),
        op="trans",
        return_time=True,
        return_meta=True,
    )
    assert out.shape == (3, 4)
    assert meta["op"] == "trans"
    assert meta["symbolic_ms"] >= 0.0
    assert meta["compute_ms"] >= 0.0
    assert elapsed_ms == pytest.approx(meta["op_total_ms"])
    assert meta["op_total_ms"] == pytest.approx(
        meta["symbolic_ms"] + meta["compute_ms"]
    )


@pytest.mark.spmm_coo
def test_spmm_coo_non_meta_has_zero_symbolic_time():
    device = accelerator_device()
    data = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    B = torch.randn(2, 3, dtype=torch.float32, device=device)
    out, meta = flagsparse_spmm_coo(data, row, col, B, (2, 2), return_meta=True)
    assert out.shape == (2, 3)
    assert meta["op"] == "non"
    assert meta["symbolic_ms"] == 0.0


@pytest.mark.spmm_coo
def test_spmm_coo_rejects_invalid_ops_and_shapes():
    device = accelerator_device()
    data = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    B_non = torch.randn(4, 3, dtype=torch.float32, device=device)
    B_bad_trans = torch.randn(4, 3, dtype=torch.float32, device=device)

    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_non, (2, 4), op="bad")
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_non, (2, 4), op="non", transpose=True)
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(
            data, row, col, B_bad_trans, (2, 4), op="trans", transpose=False
        )
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_bad_trans, (2, 4), op="trans")
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(
            data,
            row,
            col,
            B_non,
            (2, 4),
            op="trans",
            out=torch.empty((2, 3), dtype=torch.float32, device=device),
        )


# Explicit migrated algorithms share this operator's marker, golden and tolerance.
_MIGRATED_ROUTES = (('coo_segmented_panel_atomic', 'non'), ('coo_segmented_panel_atomic', 'trans'), ('coo_segmented_panel_atomic', 'conj'), ('coo_rowrun_panel', 'non'), ('coo_rowrun_panel', 'trans'), ('coo_rowrun_panel', 'conj'), ('coo_segmented_stream_panel_atomic', 'non'), ('coo_segmented_stream_panel_atomic', 'trans'), ('coo_segmented_stream_panel_atomic', 'conj'))


@pytest.mark.spmm_coo
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES, ids=[f"{a}-{o}" for a, o in _MIGRATED_ROUTES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128], ids=["float16", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_coo_migrated_routes(alg, op, dtype, index_dtype, layout):
    import importlib
    from tests.pytest.conftest import QUICK_MODE

    module = importlib.import_module("flagsparse.sparse_operations.spmm_coo")
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
    first, second = rows.to(device=device, dtype=index_dtype), cols.to(device=device, dtype=index_dtype)
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
        prepared = module.prepare_spmm_coo_route(data, first, second, shape, op=op, alg=alg)
        backing = torch.full((expected.shape[0], expected.shape[1] * 2), float("nan"), dtype=dtype, device=device)
        out = backing[:, ::2]
        result = module.flagsparse_spmm_coo_run(prepared, operand, out=out)
        direct = module.flagsparse_spmm_coo(data, first, second, operand, shape, op=op, alg=alg)
    except NotImplementedError as exc:
        pytest.skip(f"backend capability unavailable: {exc}")
    assert result is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(direct.cpu(), expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="op"):
        module.flagsparse_spmm_coo_run(prepared, operand, op="trans" if op == "non" else "non")


    with pytest.raises(ValueError, match="shape"):
        module.flagsparse_spmm_coo(data, first, second, operand, (shape[0] + 1, shape[1]),
                                   prepared=prepared)

@pytest.mark.spmm_coo
@pytest.mark.parametrize("blocks,reduce", [(1, "segment"), (2, "segment"), (4, "segment"), (8, "segment"), (2, "none")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("n", [1, 7])
def test_spmm_coo_stream_carry_boundaries(blocks, reduce, dtype, op, n):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmm_coo")
    device = accelerator_device()
    count = 521
    # A long adjacent run crosses both block and program boundaries; remaining
    # coordinates revisit the same output non-contiguously, including a tail.
    rows = torch.cat((torch.zeros(401, dtype=torch.int64), torch.arange(120) % 5))
    cols = torch.arange(count) % 7
    data = (torch.arange(count, dtype=torch.float64) % 9 - 4) / 16
    if dtype.is_complex:
        data = torch.complex(data, data.flip(0) / 2)
    data = data.to(dtype)
    shape = (5, 7)
    golden_dtype = torch.complex128 if dtype.is_complex else torch.float64
    matrix = torch.zeros(shape, dtype=golden_dtype)
    matrix.index_put_((rows, cols), data.to(golden_dtype), accumulate=True)
    matrix = matrix if op == "non" else matrix.T.conj() if op == "conj" else matrix.T
    operand = torch.ones((matrix.shape[1], n), dtype=golden_dtype)
    operand = operand.to(dtype)
    expected = (matrix @ operand.to(golden_dtype)).to(dtype)
    cfg = dict(block_nnz=32, blocks_per_program=blocks, local_reduce=reduce)
    try:
        for mode in ("materialize", "load"):
            cfg["conversion_mode"] = mode
            prepared = module.prepare_spmm_coo_route(data.to(device), rows.to(device), cols.to(device), shape,
                op=op, alg="coo_segmented_stream_panel_atomic", config=cfg)
            actual, meta = module.flagsparse_spmm_coo_run(prepared, operand.to(device), return_meta=True, timing=True)
            rtol, atol = close_tolerances(dtype)
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            assert meta["operator_ms"] == meta["gpu_ms"]
            assert meta["conversion_mode"] == mode
            assert meta["compute_dtype"] == str(torch.complex128 if dtype.is_complex else torch.float64)
    except NotImplementedError as exc:
        pytest.skip(str(exc))


@pytest.mark.spmm_coo
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5), (3, 5)])
def test_spmm_coo_migrated_empty(alg, op, shape):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmm_coo")
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.float32, device=device)
    first = torch.empty(0, dtype=torch.int64, device=device)
    second = first
    input_rows, output_rows = shape[1], shape[0]
    if op != "non":
        input_rows, output_rows = output_rows, input_rows
    operand = torch.ones((input_rows, 7), device=device)
    out = torch.full((output_rows, 7), float("nan"), device=device)
    try:
        prepared = module.prepare_spmm_coo_route(data, first, second, shape, op=op, alg=alg)
        result = module.flagsparse_spmm_coo_run(prepared, operand, out=out)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert result is out
    assert torch.equal(result.cpu(), torch.zeros_like(result.cpu()))
