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
from tests.pytest.conftest import QUICK_MODE

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)


def _value_dtype_cases():
    cases = [
        ("float16", torch.float16),
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
    if dtype in (torch.float16, torch.float32):
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
@pytest.mark.parametrize("alg", [None, "coo_segmented_panel_atomic", "coo_rowrun_panel"], ids=["legacy", "segmented_panel", "rowrun_panel"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_coo_matches_dense_reference(M, N, K, dtype_name, dtype, index_dtype, op, alg, layout):
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
        op=op, alg=alg, dense_layout=layout,
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


@pytest.mark.spmm_coo
@pytest.mark.parametrize("alg", ["coo_segmented_panel_atomic", "coo_rowrun_panel"])
@pytest.mark.parametrize("width", [1, 7, 33] if QUICK_MODE else [1, 7, 8, 16, 31, 32, 33, 64, 65, 128])
def test_spmm_coo_registered_panel_boundaries(alg, width, monkeypatch):
    from flagsparse import prepare_spmm_coo_route, flagsparse_spmm_coo_run
    from flagsparse.sparse_operations import spmv_coo as native
    device = accelerator_device()
    # Unsorted duplicate input; original column 2 becomes a long output row.
    row = torch.arange(263, dtype=torch.int64) % 5
    col = torch.full((263,), 2, dtype=torch.int32)
    values = torch.complex((torch.arange(263) % 3 - 1).double(), torch.ones(263).double() / 8)
    dense = torch.zeros((6, 7), dtype=values.dtype)
    dense.index_put_((row, col.long()), values, accumulate=True)
    B = torch.complex(torch.ones((6, width)).double(), torch.ones((6, width)).double() / 2)
    expected = dense.conj().T @ B
    prepared = prepare_spmm_coo_route(values.to(device), row.to(device), col.to(device),
                                      (6, 7), op="conj", alg=alg)
    assert prepared.shape == (6, 7)
    original = native._coo_sorted_runs
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(native, "_coo_sorted_runs", counted)
    output = torch.full((width, 7), float("nan"), dtype=B.dtype, device=device).T
    for timing in (False, True):
        actual, meta = flagsparse_spmm_coo_run(prepared, B.to(device), out=output,
                                              timing=timing, return_meta=True, dense_layout="col")
        assert actual is output
        rtol, atol = _tol(B.dtype)
        assert torch.allclose(actual.cpu(), expected, rtol=rtol, atol=atol)
        assert meta["operator_ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert len(calls) == (0 if "atomic" in alg else 3)
    assert torch.equal(prepared.data.cpu(), values)
    with pytest.raises(ValueError, match="op conflicts"):
        flagsparse_spmm_coo_run(prepared, B.to(device), op="non")


@pytest.mark.spmm_coo
@pytest.mark.parametrize("alg", ["coo_segmented_panel_atomic", "coo_rowrun_panel"])
@pytest.mark.parametrize("shape,width", [((0, 0), 0), ((0, 7), 3), ((5, 0), 7), ((5, 7), 0), ((5, 7), 3)])
def test_spmm_coo_registered_empty(alg, shape, width):
    device = accelerator_device()
    a = torch.empty(0, device=device)
    index = torch.empty(0, dtype=torch.int64, device=device)
    result = flagsparse_spmm_coo(a, index, index, torch.ones((shape[1], width), device=device), shape, alg=alg)
    assert result.shape == (shape[0], width)
    assert torch.count_nonzero(result) == 0


@pytest.fixture(autouse=True)
def _coo_extension_capability_guard(request):
    from flagsparse.sparse_operations import spmv_coo as native
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    alg = parameters.get("alg")
    if alg not in native._COO_NEW_ALGORITHMS:
        return
    dtype = parameters.get("dtype", torch.complex128)
    from flagsparse.sparse_operations.spmm_coo import _spmm_coo_compute_dtype
    dtype = _spmm_coo_compute_dtype(dtype)
    try:
        native._resolve_coo_config(alg, dtype, accelerator_device())
    except NotImplementedError as exc:
        pytest.skip(str(exc))


@pytest.mark.spmm_coo
@pytest.mark.parametrize("alg", ["coo_segmented_panel_atomic"])
def test_spmm_coo_registered_config_isolation(alg):
    from flagsparse import prepare_spmm_coo_route, flagsparse_spmm_coo_run
    device = accelerator_device()
    a = torch.tensor([1., -2.], device=device)
    r = torch.tensor([0, 1], dtype=torch.int32, device=device)
    c = torch.tensor([0, 1], dtype=torch.int64, device=device)
    B = torch.ones((2, 7), device=device)
    prepared = prepare_spmm_coo_route(a, r, c, (2, 2), alg=alg, config={"local_reduce": "none"})
    _, meta = flagsparse_spmm_coo_run(prepared, B, return_meta=True)
    assert meta["config"]["local_reduce"] == "none"
    _, switched = flagsparse_spmm_coo_run(prepared, B, alg="coo_rowrun_panel", return_meta=True)
    assert "local_reduce" not in switched["config"]
    with pytest.raises(ValueError, match="unsupported"):
        flagsparse_spmm_coo_run(prepared, B, alg="coo_rowrun_panel", config={"local_reduce": "none"})
    with pytest.raises(ValueError, match="overlap"):
        flagsparse_spmm_coo_run(prepared, B, out=B)
