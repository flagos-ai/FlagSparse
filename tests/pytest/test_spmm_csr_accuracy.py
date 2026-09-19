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
from flagsparse import prepare_spmm_csr_route, flagsparse_spmm_csr_run
from flagsparse.sparse_operations._spmm_csr_config import NEW_ALGORITHMS

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


# New routes remain unverified until these cases run on each accelerator backend.
def _new_route_inputs(dtype, index_dtype, ptr_dtype, lengths=(0, 1, 7, 8, 16, 17, 65, 2051, 3)):
    device = accelerator_device()
    ptr = torch.tensor([0] + list(__import__("itertools").accumulate(lengths)), dtype=ptr_dtype)
    idx = (torch.arange(sum(lengths), dtype=torch.int64) * 17 + 5) % 41
    t = torch.arange(sum(lengths), dtype=torch.float64)
    values = ((t % 9) - 4) / 32
    if dtype.is_complex:
        values = torch.complex(values, ((t % 7) - 3) / 64)
    values = values.to(dtype)
    shape = (len(lengths), 41)
    dense = torch.zeros(shape, dtype=_reference_dtype(dtype))
    for row in range(shape[0]):
        dense[row].index_add_(0, idx[int(ptr[row]):int(ptr[row + 1])],
                             values[int(ptr[row]):int(ptr[row + 1])].to(dense.dtype))
    return values.to(device), idx.to(device=device, dtype=index_dtype), ptr.to(device), shape, dense


def _run_new_or_skip(prepared, B, **kwargs):
    from flagsparse import SpmmCsrAlgorithmUnavailable
    try:
        return flagsparse_spmm_csr_run(prepared, B, **kwargs)
    except SpmmCsrAlgorithmUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmm_csr
@pytest.mark.parametrize("algorithm", NEW_ALGORITHMS)
@pytest.mark.parametrize("dtype", SPMM_OP_DTYPES)
@pytest.mark.parametrize("op", ("non", "trans", "conj"))
@pytest.mark.parametrize("index_dtype,ptr_dtype", [(torch.int32, torch.int32), (torch.int32, torch.int64),
                                                   (torch.int64, torch.int32), (torch.int64, torch.int64)])
@pytest.mark.parametrize("layout", ("row", "col"))
def test_spmm_csr_new_routes(algorithm, dtype, op, index_dtype, ptr_dtype, layout):
    data, idx, ptr, shape, dense = _new_route_inputs(dtype, index_dtype, ptr_dtype)
    config = dict(segment_nnz=16, short_row_threshold=8, split_row_threshold=32,
                  reduce_block_size=4, workspace_bytes=1 << 20)
    prepared = prepare_spmm_csr_route(data, idx, ptr, shape, op=op, alg=algorithm, config=config)
    left = _apply_dense_op(dense, op)
    Bcpu = _random_dense((left.shape[1], 19), dtype, "cpu")
    B = Bcpu.to(data.device)
    if layout == "col":
        B = B.T.contiguous().T
    out, meta = _run_new_or_skip(prepared, B, dense_layout=layout, return_meta=True)
    assert out.dtype == dtype
    ref = left @ Bcpu.to(left.dtype)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out.cpu().to(ref.dtype), ref, rtol=rtol, atol=atol)
    assert prepared.shape == shape
    assert prepared.materialized_route is None
    assert meta["alg_resolved"] == algorithm
    assert meta["compute_dtype"] == ("float64" if dtype in (torch.float64, torch.complex128) else "float32")
    assert meta["workspace_peak_bytes"] <= config["workspace_bytes"]


@pytest.mark.spmm_csr
@pytest.mark.parametrize("algorithm", NEW_ALGORITHMS)
@pytest.mark.parametrize("shape,n", [((0, 0), 0), ((0, 7), 3), ((5, 0), 3), ((5, 7), 0), ((5, 7), 3)])
@pytest.mark.parametrize("op", ("non", "trans", "conj"))
def test_spmm_csr_new_empty(algorithm, shape, n, op):
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.complex64, device=device)
    idx = torch.empty(0, dtype=torch.int32, device=device)
    ptr = torch.zeros(shape[0] + 1, dtype=torch.int64, device=device)
    prepared = prepare_spmm_csr_route(data, idx, ptr, shape, op=op, alg=algorithm)
    B = torch.zeros((shape[1] if op == "non" else shape[0], n), dtype=data.dtype, device=device)
    out = _run_new_or_skip(prepared, B)
    assert out.shape == (shape[0] if op == "non" else shape[1], n)
    assert torch.count_nonzero(out).item() == 0


@pytest.mark.spmm_csr
@pytest.mark.parametrize("algorithm", ("csr_split_nnz_reduce", "csr_adaptive_tile_split"))
@pytest.mark.parametrize("budget", (192, 1 << 20))
def test_spmm_csr_split_workspace_and_multiple_levels(algorithm, budget):
    data, idx, ptr, shape, dense = _new_route_inputs(torch.complex128, torch.int64, torch.int64,
                                                   lengths=(0, 2051, 1))
    config = dict(segment_nnz=8, short_row_threshold=4, split_row_threshold=16,
                  reduce_block_size=2, workspace_bytes=budget)
    p = prepare_spmm_csr_route(data, idx, ptr, shape, alg=algorithm, config=config)
    B = torch.ones((shape[1], 3), device=data.device, dtype=data.dtype) * (1 + 2j)
    out, meta = _run_new_or_skip(p, B, return_meta=True)
    rtol, atol = _tol(data.dtype)
    torch.testing.assert_close(out.cpu(), dense @ B.cpu(), rtol=rtol, atol=atol)
    assert meta["workspace_peak_bytes"] <= budget
    if budget == 192:
        assert meta["segment_batches"] > 1
    else:
        assert meta["reduction_levels"] > 1


@pytest.mark.spmm_csr
def test_spmm_csr_per_run_transpose_and_timing(monkeypatch):
    from flagsparse.sparse_operations import spmm_csr as module
    data, idx, ptr, shape, _ = _new_route_inputs(torch.complex64, torch.int64, torch.int32)
    p = prepare_spmm_csr_route(data, idx, ptr, shape, op="conj", alg="csr_row_tile", config={"tile_rows": 2})
    original = module._transpose_csr_for_spmm
    calls = []
    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_transpose_csr_for_spmm", tracked)
    B = _random_dense((shape[0], 7), data.dtype, data.device)
    first, meta = _run_new_or_skip(p, B, return_meta=True)
    assert len(calls) == 1
    timed, measured = _run_new_or_skip(p, B, timing=True, return_meta=True)
    assert len(calls) == 3  # full run plus independent phase diagnostic
    assert measured["operator_ms"] == measured["process_cpu_ms"] + measured["gpu_ms"]
    assert measured["process_gpu_ms"] >= 0 and measured["compute_ms"] >= 0
    torch.testing.assert_close(first, timed, rtol=0, atol=0)
    assert p.materialized_route is None
    # Algorithm changes do not inherit csr_row_tile's tile_rows override.
    _, other = _run_new_or_skip(p, B, alg="csr_row_kparallel", return_meta=True)
    assert other["config"]["tile_rows"] == 4
    with pytest.raises(ValueError, match="op conflicts"):
        flagsparse_spmm_csr_run(p, B, op="non")
    out = torch.empty_like(first)
    returned = _run_new_or_skip(p, B, out=out)
    assert returned is out
    with pytest.raises(ValueError, match="out"):
        flagsparse_spmm_csr_run(p, B, out=B)


@pytest.mark.spmm_csr
def test_spmm_csr_transpose_classifies_execution_rows():
    device = accelerator_device()
    m = 513
    # Original rows have one entry; transposed row 1 has 513 entries.
    data = torch.full((m,), 0.125, dtype=torch.float32, device=device)
    idx = torch.ones(m, dtype=torch.int32, device=device)
    ptr = torch.arange(m + 1, dtype=torch.int64, device=device)
    p = prepare_spmm_csr_route(data, idx, ptr, (m, 3), op="trans",
        alg="csr_adaptive_tile_split", config=dict(segment_nnz=16,
        short_row_threshold=8, split_row_threshold=32, reduce_block_size=4))
    B = torch.ones((m, 5), dtype=data.dtype, device=device)
    out, meta = _run_new_or_skip(p, B, return_meta=True)
    expected = torch.zeros((3, 5), dtype=data.dtype)
    expected[1] = m * 0.125
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert meta["split_rows"] == 1 and meta["segment_count"] == 33
    assert meta["tile_rows"] == 2


@pytest.mark.spmm_csr
def test_spmm_csr_new_out_overlap_and_high_level():
    device = accelerator_device()
    data = torch.ones(3, device=device)
    idx = torch.arange(3, dtype=torch.int32, device=device)
    ptr = torch.arange(4, dtype=torch.int64, device=device)
    p = prepare_spmm_csr_route(data, idx, ptr, (3, 3), alg="csr_row_kparallel")
    B = torch.ones((3, 5), device=device)
    expected = _run_new_or_skip(p, B)
    with pytest.raises(ValueError, match="overlap"):
        flagsparse_spmm_csr_run(p, B, out=B)
    out = flagsparse_spmm_csr(None, None, None, B, None, prepared=p)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="legacy block"):
        flagsparse_spmm_csr(data, idx, ptr, B, (3, 3), alg="csr_row_tile", block_n=32)


@pytest.mark.spmm_csr
def test_spmm_csr_out_rejects_original_index_storage_after_conversion():
    device = accelerator_device()
    values = torch.ones(3, dtype=torch.float64, device=device)
    columns = torch.arange(3, dtype=torch.int64, device=device)
    pointers = torch.arange(4, dtype=torch.int32, device=device)
    prepared = prepare_spmm_csr_route(values, columns, pointers, (3, 3), alg="csr_row_tile")
    B = torch.ones((3, 1), dtype=values.dtype, device=device)
    alias = columns.view(torch.float64).reshape(3, 1)
    with pytest.raises(ValueError, match="overlap"):
        flagsparse_spmm_csr_run(prepared, B, out=alias)
