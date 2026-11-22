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

import importlib
import json
import os
from pathlib import Path

import itertools

import pytest
import torch

from flagsparse import flagsparse_spmv_csr
from flagsparse.sparse_operations import _common as common
from flagsparse.sparse_operations._spmv_csr_benchmark import golden_csr
from tests import reference_utils
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import SPMV_MN_SHAPES

spmv_mod = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
pytestmark = pytest.mark.skipif(
    not accelerator_available(), reason=ACCELERATOR_REQUIRED
)


def _value_dtype_cases():
    cases = [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
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


def _tol(dtype):
    # fp32 / fp16 / bf16 and complex64 accumulate in fp32-precision components and
    # carry standard fp32 SpMV error (order-dependent), so they use a realistic
    # fp32 tolerance rather than the fp64-then-cast accuracy of the former
    # baseline. fp64 and complex128 accumulate in fp64 and keep the strict shared
    # tolerance.
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.complex64):
        return 1e-3, 1e-3
    return close_tolerances(dtype)


def _random_csr_mn(M, N, dtype, index_dtype, device):
    """Random CSR matrix: kernel inputs on ``device``, dense oracle copy on CPU.

    Construction runs on CPU because ``torch.where`` has no muDNN kernel for float64
    or complex on Moore Threads, so building the matrix on the accelerator failed
    before the operator under test was ever reached.  See ``golden_device()``.
    """
    golden = golden_device()
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=golden) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_dense((M, N), dtype, golden),
        torch.zeros((), dtype=dtype, device=golden),
    )
    rows, cols = torch.nonzero(mask, as_tuple=True)
    data = dense[rows, cols].contiguous()
    row_counts = torch.bincount(rows, minlength=M)
    indptr = torch.zeros(M + 1, dtype=torch.int64, device=golden)
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return (
        data.to(device),
        cols.to(index_dtype).contiguous().to(device),
        indptr.to(index_dtype).to(device),
        dense,
    )


def _make_x(length, dtype, device):
    return _random_dense((length,), dtype, device)


def _op_transposes(op):
    return op in ("trans", "conj")


def _apply_dense_op(dense, op):
    if op == "non":
        return dense
    if op == "trans":
        return dense.t()
    if op == "conj":
        return dense.conj().t()
    raise ValueError(f"unsupported op: {op}")


def _assert_close(actual, expected, dtype):
    """Compare on the golden device; ``expected`` never leaves CPU."""
    rtol, atol = _tol(dtype)
    ref_dtype = _reference_dtype(dtype)
    golden = golden_device()
    assert torch.allclose(
        actual.to(device=golden, dtype=ref_dtype),
        expected.to(device=golden, dtype=ref_dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.spmv_csr
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize(
    "name,dtype", _value_dtype_cases(), ids=[c[0] for c in _value_dtype_cases()]
)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmv_csr_matches_dense_reference(M, N, name, dtype, index_dtype, op):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csr_mn(M, N, dtype, index_dtype, device)
    transpose = _op_transposes(op)
    x_len = M if transpose else N
    x = _make_x(x_len, dtype, golden_device())
    ref_dtype = _reference_dtype(dtype)
    if common._use_scipy_accuracy_reference():
        matrix = reference_utils.scipy_csr(data, indices, indptr, (M, N), ref_dtype)
        ref = reference_utils.as_torch(
            reference_utils.spmv(matrix, x, ref_dtype, op=op),
            ref_dtype,
            golden_device(),
        ).to(dtype)
    else:
        ref_mat = _apply_dense_op(dense, op)
        ref = (ref_mat.to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
    out = flagsparse_spmv_csr(
        data,
        indices,
        indptr,
        x.to(device),
        shape=(M, N),
        op=op,
        index_fallback_policy="auto",
    )
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_csr
def test_spmv_csr_prepared_transpose_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csr_mn(
        8, 10, torch.float32, torch.int32, device
    )
    prepared = spmv_mod.prepare_spmv_csr(data, indices, indptr, (8, 10), transpose=True)
    x = torch.randn(8, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="does not match prepared.transpose"):
        flagsparse_spmv_csr(x=x, prepared=prepared, transpose=False)


@pytest.mark.spmv_csr
def test_spmv_csr_prepared_op_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csr_mn(
        8, 10, torch.complex64, torch.int32, device
    )
    prepared = spmv_mod.prepare_spmv_csr(data, indices, indptr, (8, 10), op="conj")
    x = _make_x(8, torch.complex64, device)
    with pytest.raises(ValueError, match="does not match prepared.op"):
        flagsparse_spmv_csr(x=x, prepared=prepared, op="trans")


@pytest.mark.spmv_csr
def test_spmv_csr_int64_auto_fallback_to_int32(monkeypatch):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csr_mn(
        12, 9, torch.float32, torch.int64, device
    )
    x = torch.randn(9, dtype=torch.float32, device=golden_device())
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    state = {"forced_once": False}
    original = spmv_mod._execute_spmv_route

    def fail_int64_once(prepared, x_in, out=None, timing=False):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("unsupported int64 kernel indices")
        return original(prepared, x_in, out=out, timing=timing)

    monkeypatch.setattr(spmv_mod, "_execute_spmv_route", fail_int64_once)
    out = flagsparse_spmv_csr(
        data,
        indices,
        indptr,
        x.to(device),
        shape=(12, 9),
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(
        out.to(device=golden_device(), dtype=torch.float64), ref, rtol=rtol, atol=atol
    )


@pytest.mark.spmv_csr
def test_spmv_csr_int64_strict_no_fallback(monkeypatch):
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csr_mn(
        12, 9, torch.float32, torch.int64, device
    )
    x = torch.randn(9, dtype=torch.float32, device=device)

    original = spmv_mod._execute_spmv_route

    def fail_int64(prepared, x_in, out=None, timing=False):
        if prepared.kernel_indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 kernel indices")
        return original(prepared, x_in, out=out, timing=timing)

    monkeypatch.setattr(spmv_mod, "_execute_spmv_route", fail_int64)
    with pytest.raises(RuntimeError, match="unsupported int64 kernel indices"):
        flagsparse_spmv_csr(
            data,
            indices,
            indptr,
            x,
            shape=(12, 9),
            index_fallback_policy="strict",
        )


@pytest.mark.spmv_csr
def test_spmv_csr_int64_auto_does_not_fallback_when_index_exceeds_int32(monkeypatch):
    device = accelerator_device()
    limit = spmv_mod._INDEX_LIMIT_INT32
    data = torch.ones(1, dtype=torch.float32, device=device)
    prepared = spmv_mod.PreparedCsrSpmv(
        data=data,
        kernel_indices=torch.tensor([limit + 1], dtype=torch.int64, device=device),
        kernel_indptr=torch.tensor([0, 1], dtype=torch.int64, device=device),
        shape=(1, limit + 2),
        n_rows=1,
        n_cols=limit + 2,
        block_nnz=256,
        max_segments=1,
        max_row_nnz=1,
        opt_buckets=[],
        transpose=False,
        index_fallback_policy="auto",
    )

    def fail_launch(_prepared, _x, use_opt=False, opt_buckets=None):
        raise RuntimeError("unsupported native int64 kernel indices")

    monkeypatch.setattr(spmv_mod, "_run_spmv_prepared", fail_launch)
    x = torch.empty(0, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="int32 fallback is unsafe"):
        spmv_mod._run_spmv_prepared_with_fallback(prepared, x, use_opt=False)


NEW_ALGORITHMS = spmv_mod.SPMV_CSR_NEW_ALGORITHMS
# The new CSR routes have launch profiles only for CUDA and ROCm.  Keep their
# coverage on those backends, but do not collect them as skipped XPU tests.
TESTED_ALGORITHMS = spmv_mod.list_spmv_csr_algorithms(
    backend=spmv_mod._backend_name(), dtype=torch.float32
)
TESTED_NEW_ALGORITHMS = tuple(
    alg for alg in NEW_ALGORITHMS if alg in TESTED_ALGORITHMS
    and "non" in spmv_mod.get_spmv_csr_algorithm_spec(alg)["ops"]
)
TESTED_ROUTES = tuple((alg, op) for alg in TESTED_ALGORITHMS
                      for op in spmv_mod.get_spmv_csr_algorithm_spec(alg)["ops"])
TARGETED_ROUTES = tuple((alg, op) for alg in
                       ("csr_row_refill_reduce", "csr_transpose_sharded_atomic")
                       if alg in TESTED_ALGORITHMS
                       for op in spmv_mod.get_spmv_csr_algorithm_spec(alg)["ops"])
REGRESSIONS = json.loads(
    (Path(__file__).resolve().parents[1] / "data/spmv_csr_regressions.json").read_text()
)


def _native_case(lengths, dtype, col_dtype, ptr_dtype, n=4099):
    device = accelerator_device()
    counts = torch.tensor(lengths, dtype=torch.int64)
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    nnz = int(ptr[-1])
    common._ACCEL.manual_seed(71)
    data = torch.randn(nnz, dtype=dtype, device=device)
    col = (torch.arange(nnz, dtype=torch.int64) % n).to(device=device, dtype=col_dtype)
    ptr = ptr.to(device=device, dtype=ptr_dtype)
    x = torch.randn(n, dtype=dtype, device=device)
    return data, col, ptr, x, (len(lengths), n)


def _new_prepared(data, col, ptr, shape, alg, **kwargs):
    return spmv_mod.prepare_spmv_csr(data, col, ptr, shape, alg=alg, **kwargs)


def _record_short_row_numerics(data, col, ptr, x, actual, expected, rtol, atol, record_property):
    """Replay only a failing half->FP32 row on CPU; never change the oracle.

    This is a failure diagnostic, outside all operator/performance timing. A
    match with serial FP32 is evidence about rounding order, not a reason to
    accept a result that failed the shared tolerance.
    """
    if data.dtype not in (torch.float16, torch.bfloat16) or actual.dtype != torch.float32:
        return
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape or not actual.numel():
        record_property("short_row_numerics", dict(reason="shape mismatch or empty output"))
        return
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        record_property("short_row_numerics", dict(reason="nonfinite output/reference"))
        return
    error = (actual.double() - expected.double()).abs()
    normalized = error / (atol + rtol * expected.double().abs())
    row = int(normalized.argmax())
    pointers = ptr.detach().cpu().to(torch.int64)
    start, stop = int(pointers[row]), int(pointers[row + 1])
    values = data[start:stop].detach().cpu().float()
    columns = col[start:stop].detach().cpu().to(torch.int64)
    rhs = x.detach().cpu().float()[columns]
    products = values * rhs
    serial = torch.zeros((), dtype=torch.float32, device="cpu")
    for product in products:
        serial.add_(product)
    # Explicit pairwise FP32 additions, rather than an unspecified sum kernel.
    tree = products.clone()
    while tree.numel() > 1:
        if tree.numel() % 2:
            tree = torch.cat((tree, torch.zeros(1, dtype=torch.float32, device="cpu")))
        tree = tree[::2] + tree[1::2]
    pairwise = float(tree[0]) if tree.numel() else 0.0
    exact_products = values.double() * rhs.double()
    reference64 = float(exact_products.sum())
    scale = float(atol + rtol * expected[row].double().abs())
    record_property("short_row_numerics", dict(
        input_seed=71, row=row, row_nnz=stop - start,
        failed_elements=int((normalized > 1).sum()),
        rtol=rtol, atol=atol, gpu_result=float(actual[row]),
        golden_output=float(expected[row]), reference_fp64=reference64,
        sum_abs_products=float(exact_products.abs().sum()),
        serial_fp32=float(serial), pairwise_fp32=pairwise,
        gpu_matches_serial_fp32=bool(actual[row] == serial),
        gpu_normalized_error=float(normalized[row]),
        serial_normalized_error=abs(float(serial) - float(expected[row])) / scale,
        pairwise_normalized_error=abs(pairwise - float(expected[row])) / scale,
    ))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", tuple(a for a in TESTED_NEW_ALGORITHMS
                                      if a not in spmv_mod._csr_config.SHORT_ROW_ALGORITHMS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spmv_csr_default_segment_multilevel(alg, dtype):
    data, col, ptr, x, shape = _native_case(
        [1024 * 257 + 1], dtype, torch.int64, torch.int32
    )
    prepared = _new_prepared(data, col, ptr, shape, alg)
    result = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), golden_csr(data, col, ptr, x, shape), rtol=rtol, atol=atol
    )


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", TESTED_NEW_ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "col_dtype,ptr_dtype",
    [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "lengths",
    [
        [],
        [0] * 19,
        [0, 1, 7, 8, 9, 31, 32] * 3,
        [0, 33, 1024, 1, 1025, 32, 2048, 2049, 7],
    ],
)
def test_spmv_csr_new_boundaries(alg, dtype, col_dtype, ptr_dtype, lengths):
    data, col, ptr, x, shape = _native_case(lengths, dtype, col_dtype, ptr_dtype)
    prepared = _new_prepared(data, col, ptr, shape, alg)
    out = torch.full((shape[0],), float("nan"), device=data.device, dtype=dtype)
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(
        prepared, x, out=out, return_meta=True, timing=True
    )
    assert actual is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape), rtol=rtol, atol=atol
    )
    assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert meta["compute_dtype"] == "float64"
    assert meta["alg_resolved"] == alg
    assert not prepared.opt_buckets


@pytest.mark.spmv_csr
@pytest.mark.parametrize(
    "alg",
    tuple(
        alg
        for alg in ("row_split_reduce", "row_adaptive_split")
        if alg in TESTED_NEW_ALGORITHMS
    ),
)
@pytest.mark.parametrize("dtype", spmv_mod.SUPPORTED_SPMV_VALUE_DTYPES)
def test_spmv_csr_multilevel_and_rebuild(alg, dtype, monkeypatch):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels

    config = {
        "short_row_threshold": 4,
        "split_row_threshold": 8,
        "row_split_reduce": {"segment_nnz": 8, "reduce_block_size": 2},
    }
    data, col, ptr, x, shape = _native_case(
        [0, 1, 4, 8, 9, 8 * 17 + 1, 2], dtype, torch.int64, torch.int32
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, config=config)
    plans = []
    original = kernels.build_plan

    def record_plan(*args):
        plan = original(*args)
        plans.append(plan)
        return plan

    monkeypatch.setattr(kernels, "build_plan", record_plan)
    for vector in (x, x * -0.75):
        result = spmv_mod.flagsparse_spmv_csr_run(prepared, vector)
        rtol, atol = close_tolerances(dtype)
        torch.testing.assert_close(
            result.cpu(),
            golden_csr(data, col, ptr, vector, shape),
            rtol=rtol,
            atol=atol,
        )
    assert len(plans) == 2 and plans[0] is not plans[1]
    assert len(plans[0]["levels"]) == 5


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", TESTED_NEW_ALGORITHMS)
def test_spmv_csr_cancellation(alg):
    data, col, ptr, x, shape = _native_case(
        [1025, 0, 1031], torch.float32, torch.int64, torch.int64
    )
    data.copy_(
        torch.tensor(
            [1e6, 1, -1e6, -1] * (data.numel() // 4) + [1] * (data.numel() % 4),
            device=data.device,
        )
    )
    x.fill_(1)
    prepared = _new_prepared(data, col, ptr, shape, alg)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(data.dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape), rtol=rtol, atol=atol
    )


@pytest.mark.spmv_csr
def test_spmv_csr_new_out_and_prepared_validation():
    data, col, ptr, x, shape = _native_case(
        [1, 2, 3, 0], torch.float32, torch.int32, torch.int64, n=4
    )
    prepared = _new_prepared(data, col, ptr, shape, "row_vector")
    for out in (
        x,
        data[:4],
        torch.empty(3, device=x.device),
        torch.empty(4, device=x.device, dtype=torch.float64),
    ):
        with pytest.raises((ValueError, TypeError)):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out)
    for args in (
        {"op": "trans"},
        {"alg": "row_tile"},
        {"config": {"row_vector": {"block_nnz": 64}}},
    ):
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, **args)
    transposed = spmv_mod.prepare_spmv_csr(
        data, col, ptr, shape, alg="row_vector", op="trans"
    )
    assert transposed.shape == shape
    assert transposed.kernel_indptr is ptr


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", tuple((a, op) for a in TESTED_NEW_ALGORITHMS
                                        for op in spmv_mod.get_spmv_csr_algorithm_spec(a)["ops"]))
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_spmv_csr_new_fallback_preserves_selection(alg, dtype, op, monkeypatch):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels

    data, col, ptr, x, shape = _native_case([0, 3, 65], dtype, torch.int64, torch.int64)
    if op != "non":
        x = _make_x(shape[0], dtype, data.device)
    config = ({"window_rows": 128} if alg == "csr_row_refill_reduce"
              else {"block_nnz": 64} if alg == "csr_rowgroup_stream_reduce"
              else {"rows_per_program": 32} if alg in ("csr_row_scalar_strip", "csr_row_scalar_bound", "csr_row_scalar_tail_reduce")
              else {"long_row_threshold": 32, "segment_nnz": 32, "block_nnz": 32} if alg in spmv_mod._csr_config.LONGROW_ALGORITHMS
              else {"short_row_threshold": 8, "block_nnz": 64} if alg in spmv_mod._csr_config.ROWQUEUE_ALGORITHMS
              else {"long_row_threshold": 32, "cooperative_parts": 4, "block_nnz": 32} if alg in spmv_mod._csr_config.COOPERATIVE_ALGORITHMS
              else {"short_row_threshold": 8, "long_row_threshold": 32, "block_nnz": 64} if alg in spmv_mod._csr_config.THREE_TIER_ALGORITHMS
              else {"block_nnz": 128, "atomic_mode": "all"} if alg == "csr_nnz_local_reduce_atomic"
              else {"block_nnz": 16} if alg == "csr_row_static_tile"
              else {"row_vector": {"block_nnz": 64}})
    prepared = _new_prepared(data, col, ptr, shape, alg, config=config, op=op)
    if alg in spmv_mod._csr_config.LONGROW_ALGORITHMS + spmv_mod._csr_config.ROWQUEUE_ALGORITHMS + spmv_mod._csr_config.COOPERATIVE_ALGORITHMS + spmv_mod._csr_config.THREE_TIER_ALGORITHMS + ("csr_nnz_local_reduce_atomic",):
        name = {"csr_longrow_queue_atomic": "run_longrow_queue", "csr_longrow_grid_atomic": "run_longrow_grid",
                "csr_nnz_local_reduce_atomic": "run_nnz_local_reduce", "csr_short_rowqueue_vector": "run_short_rowqueue",
                "csr_short_rowqueue_strided_vector": "run_short_rowqueue_strided",
                "csr_longrow_cooperative_reduce": "run_longrow_cooperative",
                "csr_three_tier_queue_reduce": "run_three_tier_queue"}[alg]
        original = getattr(kernels, name)
        def fail_queue_i64(route, vector, output, timing, phase):
            if route.kernel_indices.dtype == torch.int64:
                raise RuntimeError("unsupported int64 kernel indices")
            assert route.alg == alg and route.config == prepared.config
            return original(route, vector, output, timing, phase)
        monkeypatch.setattr(kernels, name, fail_queue_i64)
    else:
        original = kernels.compute
        def fail_i64(route, vector, output, algorithm, cfg, plan):
            if route.kernel_indices.dtype == torch.int64:
                raise RuntimeError("unsupported int64 kernel indices")
            assert algorithm == alg and cfg == prepared.config
            return original(route, vector, output, algorithm, cfg, plan)
        monkeypatch.setattr(kernels, "compute", fail_i64)

    actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, return_meta=True)
    rtol, atol = close_tolerances(data.dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape, op), rtol=rtol, atol=atol
    )
    assert meta["index_fallback_applied"] and meta["alg_resolved"] == alg
    assert meta["indices_dtype"] == "int32" and meta["input_indices_dtype"] == "int64"
    assert prepared.op == spmv_mod._normalize_spmv_op(op)
    prepared.index_fallback_policy = "strict"
    with pytest.raises(RuntimeError, match="unsupported int64"):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x)


if not TESTED_NEW_ALGORITHMS:
    # Do not leave empty parametrizations behind: an empty parametrization is
    # reported by pytest as a skipped case, which obscures XPU delivery rows.
    test_spmv_csr_default_segment_multilevel = None
    test_spmv_csr_new_boundaries = None
    test_spmv_csr_multilevel_and_rebuild = None
    test_spmv_csr_cancellation = None
    test_spmv_csr_new_out_and_prepared_validation = None
    test_spmv_csr_new_fallback_preserves_selection = None


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", TESTED_ROUTES)
@pytest.mark.parametrize(
    "dtype",
    spmv_mod.SUPPORTED_SPMV_VALUE_DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
@pytest.mark.parametrize("matrix_name", REGRESSIONS["matrices"])
def test_spmv_csr_external_matrix_regressions(alg, dtype, op, matrix_name):
    """Set FLAGSPARSE_SPMV_CSR_MTX_DIR to run every supplied matrix, including FP32 failures."""
    directory = os.environ.get("FLAGSPARSE_SPMV_CSR_MTX_DIR")
    if not directory:
        pytest.skip("external matrix regression directory not configured")
    from tests.mtx_fast import load_csr

    paths = list(Path(directory).rglob(matrix_name))
    assert (
        len(paths) == 1
    ), f"expected exactly one regression input {matrix_name} under {directory}"
    path = paths[0]
    data, col, ptr, shape = load_csr(path, dtype=dtype, device=accelerator_device())
    common._ACCEL.manual_seed(2026)
    assert not col.numel() or int(col.max()) <= 2147483647
    col, ptr = col.to(torch.int32), ptr.to(torch.int64)
    x = torch.randn(
        shape[1] if op == "non" else shape[0], dtype=dtype, device=data.device
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(),
        golden_csr(data, col, ptr, x, shape, op),
        rtol=rtol,
        atol=atol,
        msg=str(path),
    )


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", TESTED_ROUTES)
@pytest.mark.parametrize("dtype", spmv_mod.SUPPORTED_SPMV_VALUE_DTYPES)
@pytest.mark.parametrize(
    "col_dtype,ptr_dtype",
    [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    ],
)
def test_spmv_csr_full_dtype_op_surface(alg, dtype, op, col_dtype, ptr_dtype):
    if alg == "legacy_bucket_vector" and col_dtype == torch.int64:
        pytest.skip("legacy bucket requires int32 column indices")
    data, col, ptr, _, shape = _native_case(
        [3, 0, 1, 5], dtype, col_dtype, ptr_dtype, n=7
    )
    # Unsorted, duplicate column IDs and an explicit zero are retained by transpose.
    col.copy_(
        torch.tensor([5, 1, 5, 2, 6, 0, 3, 0, 1], device=col.device, dtype=col_dtype)
    )
    data[0] = 0
    x = _make_x(shape[1] if op == "non" else shape[0], dtype, data.device)
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    assert prepared.shape == shape and prepared.kernel_indptr is ptr
    output_size = shape[0] if op == "non" else shape[1]
    out = torch.full((output_size,), float("nan"), dtype=dtype, device=data.device)
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(
        prepared, x, out=out, return_meta=True
    )
    assert actual is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape, op), rtol=rtol, atol=atol
    )
    expected_compute = str(dtype).removeprefix("torch.")
    if dtype in (torch.float16, torch.bfloat16):
        expected_compute = "float32"
    elif dtype == torch.float32 and alg in (*NEW_ALGORITHMS, "legacy_rowpar"):
        expected_compute = "float64"
    assert meta["compute_dtype"] == expected_compute
    assert meta["transpose_strategy"] == (
        "none" if op == "non" else spmv_mod.get_spmv_csr_algorithm_spec(alg)["transpose_strategy"]
    )
    assert meta["alg_resolved"] == alg
    with pytest.raises(ValueError):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out[:-1])
    with pytest.raises(ValueError):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x[:-1])


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", tuple(a for a in TESTED_ALGORITHMS if spmv_mod.get_spmv_csr_algorithm_spec(a)["transpose_strategy"] == "per_run_csr_rebuild"))
@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmv_csr_transpose_rebuilt_inside_each_run(alg, op, monkeypatch):
    data, col, ptr, _, shape = _native_case(
        [1] * 2051, torch.complex64, torch.int32, torch.int64, n=3
    )
    col.zero_()  # Short original rows become one long output row and two empty rows.
    data.fill_(1 + 0.5j)
    x = torch.full((shape[0],), 1 + 0.25j, dtype=data.dtype, device=data.device)
    rebuilds = []
    original = spmv_mod._transpose_csr_for_spmv

    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        rebuilds.append(value)
        return value

    monkeypatch.setattr(spmv_mod, "_transpose_csr_for_spmv", record)
    config = (
        {"row_split_reduce": {"segment_nnz": 8, "reduce_block_size": 2}}
        if alg in ("row_split_reduce", "row_adaptive_split")
        else None
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op, config=config)
    assert not rebuilds
    saved_data = data.clone()
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(
        prepared, x, timing=True, return_meta=True
    )
    assert len(rebuilds) == 2  # Complete run and separate phase diagnostic.
    second, plain = spmv_mod.flagsparse_spmv_csr_run(prepared, -x, return_meta=True)
    assert len(rebuilds) == 3
    assert rebuilds[0][2].data_ptr() != rebuilds[1][2].data_ptr()
    assert prepared.shape == shape and prepared.data is data
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)
    rtol, atol = close_tolerances(data.dtype)
    ref = golden_csr(data, col, ptr, x, shape, op)
    torch.testing.assert_close(actual.cpu(), ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(second.cpu(), -ref, rtol=rtol, atol=atol)
    for timing_meta in (plain, meta):
        assert (
            timing_meta["ms"] == timing_meta["gpu_ms"] + timing_meta["process_cpu_ms"]
        )
    assert meta["process_gpu_ms"] >= 0 and meta["compute_ms"] >= 0


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", TESTED_ROUTES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("shape", [(0, 7), (5, 0), (0, 0), (5, 7)])
def test_spmv_csr_empty_dtype_ops(alg, dtype, op, shape):
    device = accelerator_device()
    data = torch.empty(0, dtype=dtype, device=device)
    col = torch.empty(0, dtype=torch.int32, device=device)
    ptr = torch.zeros(shape[0] + 1, dtype=torch.int64, device=device)
    x = torch.ones(shape[1] if op == "non" else shape[0], dtype=dtype, device=device)
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    assert actual.shape == (shape[0] if op == "non" else shape[1],)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", TESTED_ROUTES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_spmv_csr_complex_cancellation(alg, dtype, op):
    data, col, ptr, _, shape = _native_case(
        [1025, 0, 1031], dtype, torch.int32, torch.int64, n=3
    )
    col.zero_()
    pattern = torch.tensor(
        [10000 + 5000j, 1 - 2j, -10000 - 5000j, -1 + 2j],
        dtype=dtype,
        device=data.device,
    )
    data.copy_(pattern.repeat((data.numel() + 3) // 4)[: data.numel()])
    x = torch.full(
        (shape[1] if op == "non" else shape[0],),
        1 + 0.5j,
        dtype=dtype,
        device=data.device,
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape, op), rtol=rtol, atol=atol
    )


# Explicit migrated algorithms share this operator's marker, golden and tolerance.

@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("lengths", [[], [0, 0, 0], [0, 1, 7, 8, 9, 0, 513, 2, 0]])
@pytest.mark.parametrize("plan_mode", ["compact", "fixed"])
@pytest.mark.parametrize("workspace", [512, 16384])
def test_spmv_csr_nnz_tiles_batched_boundaries(plan_mode, workspace, dtype, wide, lengths):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    if "csr_nnz_tile_reduce" not in spmv_mod.list_spmv_csr_algorithms(dtype=dtype, backend=common._backend_name()):
        pytest.skip("NNZ tile capability is unavailable on this backend")
    output_dtype = torch.float32 if wide else dtype
    device = accelerator_device()
    ptr_cpu = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int64)
    count = int(ptr_cpu[-1])
    cols_cpu = (torch.arange(count) * 13) % 37
    values_cpu = torch.where(torch.arange(count) % 3 == 0, -0.125, 0.25).to(dtype)
    values, cols, ptr = values_cpu.to(device), cols_cpu.to(device), ptr_cpu.to(device)
    x = torch.full((37,), 0.5, dtype=dtype, device=device)
    cfg = dict(block_nnz=8, reduce_block_size=4, workspace_bytes=workspace, plan_mode=plan_mode)
    try:
        prepared = spmv_mod.prepare_spmv_csr(values, cols, ptr, (len(lengths), 37),
                                            alg="csr_nnz_tile_reduce", config=cfg)
        backing = torch.full((2 * len(lengths),), float("nan"), dtype=output_dtype, device=device)
        out = backing[::2]
        for _ in range(2):
            values.mul_(0.5)
            out.fill_(float("nan"))
            result, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out, return_meta=True)
            expected = torch.zeros(len(lengths), dtype=torch.float64)
            owners = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths, dtype=torch.int64))
            expected.index_add_(0, owners, values.cpu().double() * 0.5)
            assert result is out
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(result.cpu(), expected.to(output_dtype), rtol=rtol, atol=atol)
            assert meta["workspace_peak_bytes"] <= cfg["workspace_bytes"]
            assert meta["plan_builds"] == 1
            assert meta["plan_mode"] == plan_mode
            if plan_mode == "fixed":
                assert "boundary_record_count" not in meta
                assert meta["boundary_slot_capacity"] >= 0
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))

@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("plan_mode", ["compact", "fixed"])
@pytest.mark.parametrize("return_time,return_meta,timing", list(itertools.product([False, True], repeat=3)))
def test_spmv_csr_nnz_return_contract(dtype, wide, plan_mode, return_time, return_meta, timing):
    """Benchmark requests time+metadata; tensor-only success is not sufficient."""
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    if "csr_nnz_tile_reduce" not in spmv_mod.list_spmv_csr_algorithms(dtype=dtype, backend=common._backend_name()):
        pytest.skip("NNZ tile capability is unavailable on this backend")
    device = accelerator_device()
    output_dtype = torch.float32 if wide else dtype
    values = torch.tensor([1, 0.00390625, -0.125, 0.5, 0.25], dtype=dtype, device=device)
    indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=device)
    x = torch.full((4,), 1.0078125, dtype=dtype, device=device)
    expected = torch.tensor([0.87890625, 0, 0.75], dtype=torch.float64) * float(x[0].cpu())
    out = torch.full((6,), float("nan"), dtype=output_dtype, device=device)[::2]
    cfg = dict(plan_mode=plan_mode, block_nnz=2, reduce_block_size=4, workspace_bytes=256)
    try:
        prepared = spmv_mod.prepare_spmv_csr(values, indices, indptr, (3, 4),
                                            alg="csr_nnz_tile_reduce", config=cfg)
        result = spmv_mod.flagsparse_spmv_csr_run(
            prepared, x, out=out, out_dtype=output_dtype,
            return_time=return_time, return_meta=return_meta, timing=timing,
        )
        if return_meta:
            if return_time:
                actual, elapsed, meta = result
                assert elapsed == meta["ms"]
            else:
                actual, meta = result
            assert meta["plan_mode"] == plan_mode
            assert meta["ms"] == meta["process_cpu_ms"] + meta["gpu_ms"]
            assert meta["plan_builds"] == 1
            assert meta["workspace_peak_bytes"] <= cfg["workspace_bytes"]
            assert ("process_gpu_ms" in meta) == timing
            assert ("compute_ms" in meta) == timing
            expected_key = "boundary_slot_capacity" if plan_mode == "fixed" else "boundary_record_count"
            other_key = "boundary_record_count" if plan_mode == "fixed" else "boundary_slot_capacity"
            assert meta[expected_key] >= 0
            assert other_key not in meta
        elif return_time:
            actual, elapsed = result
            assert elapsed >= 0
        else:
            actual = result
        assert actual is out
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(actual.cpu(), expected.to(output_dtype), rtol=rtol, atol=atol)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


_MIGRATED_ROUTES = (('row_subgroup_stream', 'non'), ('row_subgroup_stream', 'trans'), ('row_subgroup_stream', 'conj'))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES, ids=[f"{a}-{o}" for a, o in _MIGRATED_ROUTES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128], ids=["float16", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])

def test_spmv_csr_migrated_routes(alg, op, dtype, index_dtype):
    import importlib
    from tests.pytest.conftest import QUICK_MODE

    module = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
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
    operand = (torch.arange(input_rows, dtype=torch.float64) % 7 - 3) / 8
    if dtype.is_complex:
        operand = torch.complex(operand, operand.flip(0) / 2)
    operand = operand.to(dtype)
    expected_matrix = dense if op == "non" else dense.T.conj() if op == "conj" else dense.T
    expected = (expected_matrix @ operand.to(reference_dtype)).to(dtype)
    operand = operand.to(device)

    try:
        prepared = module.prepare_spmv_csr(data, first, second, shape, op=op, alg=alg)
        out = torch.full(expected.shape, float("nan"), dtype=dtype, device=device)
        result = module.flagsparse_spmv_csr_run(prepared, operand, out=out)
        direct = module.flagsparse_spmv_csr(data=data, indices=first, indptr=second, x=operand, shape=shape, op=op, alg=alg)
    except NotImplementedError as exc:
        pytest.skip(f"backend capability unavailable: {exc}")
    assert result is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(direct.cpu(), expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="op"):
        module.flagsparse_spmv_csr_run(prepared, operand, op="trans" if op == "non" else "non")


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5), (3, 5)])
def test_spmv_csr_migrated_empty(alg, op, shape):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.float32, device=device)
    first = torch.empty(0, dtype=torch.int64, device=device)
    second = torch.zeros(shape[0] + 1, dtype=torch.int32, device=device)
    input_rows, output_rows = shape[1], shape[0]
    if op != "non":
        input_rows, output_rows = output_rows, input_rows
    operand = torch.ones((input_rows,), device=device)
    out = torch.full((output_rows,), float("nan"), device=device)
    try:
        prepared = module.prepare_spmv_csr(data, first, second, shape, op=op, alg=alg)
        result = module.flagsparse_spmv_csr_run(prepared, operand, out=out)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert result is out
    assert torch.equal(result.cpu(), torch.zeros_like(result.cpu()))


from tests.pytest.conftest import QUICK_MODE as _MIXED_QUICK_MODE


@pytest.mark.parametrize("index_pair", [(torch.int32, torch.int64)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
], ids=lambda pair: "/".join(str(dtype).removeprefix("torch.") for dtype in pair))
@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,config", [(name, None) for name in [
    "csr_rowgroup_stream_reduce",
    "row_tile", "row_vector", "row_subgroup_stream", "csr_subgroup_pair_mixed",
    "csr_nnz_tile_reduce",
    "row_split_reduce", "row_adaptive_split",
    "legacy_rowpar", "legacy_segbin", "legacy_bucket_vector",
]] + [
    pytest.param("csr_nnz_tile_reduce", {"plan_mode": "fixed"}, id="nnz-fixed"),
    pytest.param("row_subgroup_stream", {"row_subgroup_stream": {"row_tiles_per_program": 1, "lanes_per_row": 2}}, id="subgroup-small"),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("wide", [False, True], ids=["same_out", "f32_out"])
def test_spmv_csr_registered_wide_output(alg, config, dtype, op, wide, index_pair):
    """Wide output must retain bits that a half-output intermediate would lose."""
    spec = spmv_mod.get_spmv_csr_algorithm_spec(alg)
    if str(index_pair[0]).removeprefix("torch.") not in spec["index_dtypes"]:
        pytest.skip("algorithm does not support this column index dtype")
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        output_dtype = torch.float32 if wide else dtype
        device = accelerator_device()
        # Duplicate entries also exercise accumulation without coalescing.
        ptr = torch.tensor([0, 3, 3, 5], dtype=index_pair[1], device=device)
        cols = torch.tensor([0, 0, 1, 1, 2], dtype=index_pair[0], device=device)
        values_cpu = torch.tensor([1, 0.00390625, -0.125, 0.5, 0.25], dtype=dtype)
        values = values_cpu.to(device)
        matrix = torch.zeros((3, 4), dtype=torch.float64)
        matrix.index_put_(
            (torch.tensor([0, 0, 0, 2, 2]), cols.cpu().long()),
            values_cpu.double(), accumulate=True,
        )
        inner = 4 if op == "non" else 3
        x = torch.ones(inner, dtype=dtype, device=device) * 1.0078125
        expected = ((matrix if op == "non" else matrix.T) @ x.cpu().double()).to(output_dtype)
        from flagsparse.sparse_operations._common import _backend_name
        if alg not in spmv_mod.list_spmv_csr_algorithms(
            op=op, dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = spmv_mod.prepare_spmv_csr(values, cols, ptr, (3, 4), op=op, alg=alg, config=config)
        out = torch.full(expected.shape, float("nan"), dtype=output_dtype, device=device)
        result = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out, out_dtype=output_dtype)
        assert result is out
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
        direct = flagsparse_spmv_csr(
            values, cols, ptr, x, (3, 4), alg=alg, config=config, op=op, out_dtype=output_dtype
        )
        torch.testing.assert_close(direct, result, rtol=rtol, atol=atol)
        with pytest.raises(ValueError, match="conflicts"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out, out_dtype=dtype if wide else torch.float32)
        with pytest.raises(ValueError, match="overlap"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=values)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("config", [None, {'csr_subgroup_pair_mixed': {'prefetch_chunks': 1, 'lanes_per_row': 2, 'row_tiles_per_program': 1}}], ids=lambda c: "default" if c is None else str(c))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wide", [False, True], ids=["same_out", "f32_out"])
@pytest.mark.parametrize("case", ["empty", "tails", "unordered", "zero_axis"])
def test_spmv_csr_independent_mixed_boundaries(config, dtype, wide, case):
    from importlib import import_module
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    module = import_module("flagsparse.sparse_operations.spmv_csr")
    device = accelerator_device()
    lengths = [0] * 13 if case == "empty" else [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 129, 257]
    if case == "zero_axis":
        lengths = []
    axis = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths, dtype=torch.int64))
    other = (torch.arange(axis.numel()) * 7) % 19
    vals = ((torch.arange(axis.numel()) % 7 - 3).float() / 16).to(dtype)
    rows, cols = (axis, other)
    shape = (len(lengths), 19)
    if case == "unordered":
        # Reverse within each compressed object; COO additionally disperses owners.
        order = torch.arange(axis.numel() - 1, -1, -1)
        order = order[torch.argsort(axis[order], stable=True)]
        rows, cols, vals = rows[order], cols[order], vals[order]
    matrix = torch.zeros(shape, dtype=torch.float64)
    matrix.index_put_((rows.long(), cols.long()), vals.double(), accumulate=True)
    values = vals.to(device)
    first = cols.to(device=device, dtype=torch.int32)
    second = torch.tensor([0] + torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist(), dtype=torch.int64, device=device)
    op = "non"
    effective = matrix.T if op == "trans" else matrix
    output_dtype = torch.float32 if wide else dtype
    rhs = ((torch.arange(effective.shape[1]) % 5 - 2).float() / 8).to(dtype)
    operand = torch.empty(rhs.numel() * 2, dtype=dtype, device=device)[::2]
    operand.copy_(rhs.to(device))
    expected = (effective @ rhs.double()).to(output_dtype)
    backing = torch.full((*expected.shape[:-1], expected.shape[-1] * 2), float("nan"), dtype=output_dtype, device=device)
    out = backing[..., ::2]
    originals = [v.clone() for v in (values, first, second, operand)]
    try:
        prepared = module.prepare_spmv_csr(values, first, second, shape, op=op, alg="csr_subgroup_pair_mixed", config=config)
        result, meta = module.flagsparse_spmv_csr_run(prepared, operand, out=out, out_dtype=output_dtype, return_meta=True)
        assert result is out
        if config is None and case == "tails":
            measured, elapsed, diagnostic = module.flagsparse_spmv_csr_run(
                prepared, operand, out=out, out_dtype=output_dtype,
                return_meta=True, return_time=True, timing=True)
            assert measured is out
            assert elapsed == pytest.approx(diagnostic["gpu_ms"] + diagnostic.get("process_cpu_ms", 0.0))
            assert diagnostic["compute_ms"] >= 0
            assert diagnostic["process_gpu_ms"] >= 0
            with pytest.raises(ValueError, match="op|prepared"):
                module.flagsparse_spmv_csr_run(prepared, operand, op="non" if op == "trans" else "trans")
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
        for actual, original in zip((values, first, second, operand), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert meta.get("alg_resolved", meta.get("algorithm")) == "csr_subgroup_pair_mixed"
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("case", ["empty", "boundaries"])
@pytest.mark.parametrize("index_pair", [(torch.int32, torch.int64)] if _MIXED_QUICK_MODE else list(itertools.product((torch.int32, torch.int64), repeat=2)))
def test_csr_rowgroup_stream_contract(dtype, op, case, monkeypatch, record_property, index_pair):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    def forbidden(*args, **kwargs):
        raise AssertionError("rowgroup must not construct a runtime partition plan")
    monkeypatch.setattr(kernels, "build_plan", forbidden)
    device = accelerator_device()
    lengths = torch.tensor([0] * 19 if case == "empty" else
                           [0, 1, 63, 64, 65, 0, 129, 3, 0, 0, 513, 2, 1, 0, 7, 0, 9, 1, 0])
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    ids = torch.arange(int(ptr[-1]))
    cols = (ids * 13 % 23).to(torch.int32)
    raw = (ids % 7 - 3).double() / 32
    if dtype.is_complex:
        raw = raw + 1j * (ids % 5 - 2).double() / 16
    values = raw.to(dtype)
    golden_dtype = torch.complex128 if dtype.is_complex else torch.float64
    matrix = torch.zeros((19, 23), dtype=golden_dtype)
    matrix.index_put_((torch.repeat_interleave(torch.arange(19), lengths), cols.long()),
                      values.to(golden_dtype), accumulate=True)
    effective = matrix if op == "non" else matrix.T if op == "trans" else matrix.mH
    x_cpu = (torch.arange(effective.shape[1]).double() % 5 - 2) / 8
    if dtype.is_complex:
        x_cpu = x_cpu + 0.125j
    x_cpu = x_cpu.to(dtype)
    a, idx, p = values.to(device), cols.to(device=device, dtype=index_pair[0]), ptr.to(device=device, dtype=index_pair[1])
    x_storage = torch.zeros(x_cpu.numel() * 2, dtype=dtype, device=device)
    x = x_storage[::2]
    x.copy_(x_cpu)
    before = a.clone()
    record_property("algorithm", "csr_rowgroup_stream_reduce")
    record_property("a_dtype", str(dtype))
    record_property("rhs_dtype", str(dtype))
    record_property("output_dtypes", [str(dtype), "torch.float32"] if dtype in (torch.float16, torch.bfloat16) else [str(dtype)])
    record_property("compute_dtype", "float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32")
    record_property("opA", op)
    try:
        prepared = spmv_mod.prepare_spmv_csr(a, idx, p, (19, 23), op=op,
            alg="csr_rowgroup_stream_reduce", config={"rows_per_program": 8, "block_nnz": 64})
        output_types = [dtype, torch.float32] if dtype in (torch.float16, torch.bfloat16) else [dtype]
        for output_dtype in output_types:
            out = torch.full((effective.shape[0] * 2,), float("nan"), dtype=output_dtype, device=device)[::2]
            y, ms, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out,
                out_dtype=output_dtype, return_time=True, return_meta=True, timing=True)
            assert y is out
            assert meta["workspace_peak_bytes"] == 0
            assert meta["process_cpu_ms"] == 0
            assert ms == meta["gpu_ms"] + meta["process_cpu_ms"]
            expected = (effective @ x_cpu.to(golden_dtype)).to(output_dtype)
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(y.cpu(), expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(a, before, rtol=0, atol=0)
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, op="trans" if op == "non" else "non")
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,op", TARGETED_ROUTES)
@pytest.mark.parametrize("dtype,wide", [(dt, False) for dt in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)]
    + [(torch.float16, True), (torch.bfloat16, True)])
@pytest.mark.parametrize("case", ["empty", "refill_tail", "hotspot"])
@pytest.mark.parametrize("index_pair", [(torch.int32, torch.int64)] if _MIXED_QUICK_MODE else
                         list(itertools.product((torch.int32, torch.int64), repeat=2)))
@pytest.mark.parametrize("stride", [1, 2])
def test_spmv_csr_targeted_contract(alg, op, dtype, wide, case, index_pair, stride, monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    def forbidden(*args, **kwargs):
        raise AssertionError("targeted algorithms must not build a segment plan")
    monkeypatch.setattr(kernels, "build_plan", forbidden)
    if alg == "csr_transpose_sharded_atomic":
        monkeypatch.setattr(spmv_mod, "_transpose_csr_for_spmv", forbidden)
    lengths = ([0] * 19 if case == "empty" else
               ([1, 1, 1, 64, 64, 1, 1, 1, 0] * 3 + [513, 0, 2]))
    data, col, ptr, _, shape = _native_case(lengths, dtype, *index_pair, n=23)
    ids = torch.arange(data.numel(), device=data.device)
    data.copy_(((ids % 7 - 3).float() / 32).to(dtype))
    if dtype.is_complex:
        data.add_(1j * ((ids % 5 - 2).float() / 16))
    if case == "hotspot":
        col.zero_()
    x_cpu = ((torch.arange(shape[1] if op == "non" else shape[0]) % 5 - 2).double() / 8).to(dtype)
    if dtype.is_complex:
        x_cpu += 0.125j
    x = torch.zeros(stride * x_cpu.numel(), dtype=dtype, device=data.device)[::stride]
    x.copy_(x_cpu)
    output_dtype = torch.float32 if wide else dtype
    output_size = shape[0] if op == "non" else shape[1]
    saved = [t.clone() for t in (data, col, ptr, x)]
    configs = ([{"rows_per_program": 4, "lanes_per_row": 8, "window_rows": w} for w in (4, 16, 128)]
               if alg == "csr_row_refill_reduce" else [{"output_shards": h} for h in (1, 2, 4, 8)])
    # CPU reference retains FP64/complex128 until the requested final output.
    cpu_a, cpu_col, cpu_ptr = data.cpu(), col.cpu().long(), ptr.cpu().long()
    ref_type = torch.complex128 if dtype.is_complex else torch.float64
    mat = torch.zeros(shape, dtype=ref_type)
    owners = torch.repeat_interleave(torch.arange(shape[0]), cpu_ptr[1:] - cpu_ptr[:-1])
    mat.index_put_((owners, cpu_col), cpu_a.to(ref_type), accumulate=True)
    effective = mat if op == "non" else mat.T if op == "trans" else mat.mH
    expected = (effective @ x_cpu.to(ref_type)).to(output_dtype)
    record_property("algorithm", alg)
    record_property("type_combo", f"{dtype}:{dtype}:{output_dtype}")
    record_property("compute_dtype", "float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32")
    record_property("opA", op)
    for config in configs:
        try:
            prepared = spmv_mod.prepare_spmv_csr(data, col, ptr, shape, alg=alg, op=op, config=config)
        except BackendCapabilityUnavailable as exc:
            pytest.skip(str(exc))
        out = torch.full((stride * output_size,), float("nan"), dtype=output_dtype, device=data.device)[::stride]
        actual, ms, meta = spmv_mod.flagsparse_spmv_csr_run(
            prepared, x, out=out, out_dtype=output_dtype, return_time=True, return_meta=True, timing=True)
        assert actual is out and meta["alg_resolved"] == alg
        if alg == "csr_transpose_sharded_atomic":
            assert all(meta[key] >= 0 for key in ("init_ms", "scatter_ms", "reduce_ms"))
            assert ms == meta["gpu_ms"] + meta["process_cpu_ms"]
        assert ms == meta["gpu_ms"] + meta["process_cpu_ms"]
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        high = spmv_mod.flagsparse_spmv_csr(data, col, ptr, x, shape, alg=alg, op=op,
                                           config=config, out_dtype=output_dtype)
        torch.testing.assert_close(high.cpu(), expected, rtol=rtol, atol=atol)
        if alg == "csr_transpose_sharded_atomic":
            assert meta["transpose_strategy"] == "native_csr_sharded_scatter"
            assert meta["process_gpu_ms"] == 0
        else:
            assert meta["workspace_peak_bytes"] == 0
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, op="non" if op != "non" else "trans")
        with pytest.raises(ValueError, match="overlap"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=x)
        with pytest.raises((ValueError, TypeError)):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out,
                out_dtype=torch.float64 if output_dtype != torch.float64 else torch.float32)
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, alg="row_vector")
    for tensor, before in zip((data, col, ptr, x), saved):
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)


@pytest.mark.spmv_csr
def test_spmv_sharded_workspace_and_fallback(monkeypatch):
    from flagsparse.sparse_operations import _spmv_csr_transpose as transpose
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    data, col, ptr, _, shape = _native_case([0, 3, 5], torch.float16, torch.int64, torch.int64, n=7)
    x = torch.ones(shape[0], dtype=data.dtype, device=data.device)
    try:
        prepared = spmv_mod.prepare_spmv_csr(data, col, ptr, shape, op="trans",
            alg="csr_transpose_sharded_atomic", config={"output_shards": 2})
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    original = transpose.compute_sharded
    def inject(route, vector, out=None):
        if route.kernel_indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 kernel indices")
        assert route.config == prepared.config and route.op == prepared.op
        return original(route, vector, out)
    monkeypatch.setattr(transpose, "compute_sharded", inject)
    out = torch.empty(shape[1], dtype=torch.float32, device=data.device)
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out, return_meta=True)
    assert actual is out and meta["index_fallback_applied"]
    assert meta["compute_dtype"] == "float32" and meta["output_dtype"] == "float32"
    prepared.index_fallback_policy = "strict"
    with pytest.raises(RuntimeError, match="unsupported int64"):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out)
    monkeypatch.setattr(transpose, "compute_sharded", original)
    small = spmv_mod.prepare_spmv_csr(data, col, ptr, shape, op="trans",
        alg="csr_transpose_sharded_atomic", config={"workspace_limit_bytes": 1})
    with pytest.raises(MemoryError, match="workspace"):
        spmv_mod.flagsparse_spmv_csr_run(small, x)


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", ["csr_row_refill_reduce", "csr_transpose_sharded_atomic",
                                  "csr_row_scalar_strip", "csr_row_static_tile", "csr_row_scalar_bound", "csr_row_scalar_tail_reduce", "csr_longrow_queue_atomic",
                                  "csr_nnz_local_reduce_atomic", "csr_longrow_grid_atomic", "csr_short_rowqueue_vector",
                                  "csr_short_rowqueue_strided_vector", "csr_longrow_cooperative_reduce", "csr_three_tier_queue_reduce"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_spmv_targeted_wide_output_not_roundtrip(alg, dtype):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    device = accelerator_device()
    delta = 2 ** (-11 if dtype == torch.float16 else -9)
    a = torch.tensor([1.0, delta], dtype=dtype, device=device)
    if alg == "csr_row_scalar_tail_reduce":
        # Place both terms in the parallel tail, beyond the default scalar prefix.
        a = torch.cat((torch.zeros(8, dtype=dtype, device=device), a))
    if alg in ("csr_nnz_local_reduce_atomic", "csr_longrow_grid_atomic", *spmv_mod._csr_config.ROWQUEUE_ALGORITHMS, *spmv_mod._csr_config.COOPERATIVE_ALGORITHMS):
        # Force the two nonzero contributions into distinct tile/segment jobs.
        a = torch.cat((a[:1], torch.zeros(2047, dtype=dtype, device=device), a[1:]))
    if alg != "csr_transpose_sharded_atomic":
        shape, op, offsets, columns = (1, 2), "non", [0, a.numel()], [0] * (a.numel() - 1) + [1]
    else:
        shape, op, offsets, columns = (2, 1), "trans", [0, 1, 2], [0, 0]
    ptr = torch.tensor(offsets, dtype=torch.int64, device=device)
    col = torch.tensor(columns, dtype=torch.int32, device=device)
    x = torch.ones(2, dtype=dtype, device=device)
    try:
        prepared = spmv_mod.prepare_spmv_csr(a, col, ptr, shape, alg=alg, op=op)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    y = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out_dtype=torch.float32)
    expected = torch.tensor([1 + delta], dtype=torch.float32, device=device)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    assert not torch.equal(y, y.to(dtype).float())


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg,config", [
    ("csr_row_scalar_tail_reduce", None),
    ("csr_row_scalar_tail_reduce", {"scalar_steps": 0, "block_nnz": 8, "rows_per_program": 32}),
    ("csr_row_scalar_tail_reduce", {"scalar_steps": 4, "block_nnz": 16}),
    ("csr_row_scalar_tail_reduce", {"scalar_steps": 32, "block_nnz": 64, "rows_per_program": 128}),
    ("csr_row_scalar_bound", None), ("csr_row_scalar_bound", {"rows_per_program": 32}),
    ("csr_row_scalar_strip", None), ("csr_row_scalar_strip", {"rows_per_program": 32}),
    ("csr_row_static_tile", None),
    ("csr_row_static_tile", {"rows_per_program": 4, "block_nnz": 8}),
])
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32), (torch.float64, torch.float64),
    (torch.complex64, torch.complex64), (torch.complex128, torch.complex128),
], ids=lambda dtype: str(dtype).removeprefix("torch."))
@pytest.mark.parametrize("index_pair", [(torch.int64, torch.int32)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
])
def test_spmv_csr_short_rows_native(alg, config, dtype, output_dtype, index_pair, monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype),
                           output_dtype=str(output_dtype), opA="non", opB="non",
                           config=config, input_seed=71,
                           compute_dtype="float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32").items():
        record_property(key, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("short rows must not build a runtime plan")

    monkeypatch.setattr(kernels, "build_plan", forbidden)
    # Empty groups, a full fast group, both K boundaries, a long-row group and R tail.
    lengths = [0] * 128 + [1, 2, 3, 4, 5, 7] * 2 + [8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257, 0, 2, 1]
    if not _MIXED_QUICK_MODE:
        lengths += [8193, 0, 1, 0, 1025]
    data, col, ptr, x, shape = _native_case(lengths, dtype, *index_pair, n=19)
    x_storage = torch.empty(38, dtype=dtype, device=x.device)
    x_storage[::2] = x
    x = x_storage[::2]
    originals = [t.clone() for t in (data, col, ptr, x)]
    backing = torch.full((shape[0] * 2,), float("nan"), dtype=output_dtype, device=x.device)
    out = backing[::2]
    try:
        prepared = _new_prepared(data, col, ptr, shape, alg, config=config)
        result, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out, return_meta=True, timing=True)
        direct = spmv_mod.flagsparse_spmv_csr(data, col, ptr, x, shape,
                                            alg=alg, config=config, out_dtype=output_dtype)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    # Golden from stored input values, before any low-precision output conversion.
    reference_type = torch.complex128 if dtype.is_complex else torch.float64
    expected = golden_csr(data.to(reference_type), col, ptr, x.to(reference_type), shape).to(output_dtype)
    rtol, atol = close_tolerances(output_dtype)
    actual_cpu = result.cpu()
    try:
        torch.testing.assert_close(actual_cpu, expected, rtol=rtol, atol=atol)
    except AssertionError:
        _record_short_row_numerics(data, col, ptr, x, actual_cpu, expected, rtol, atol, record_property)
        raise
    torch.testing.assert_close(direct, result, rtol=rtol, atol=atol)
    assert result is out and meta["alg_resolved"] == alg
    if alg == "csr_row_scalar_tail_reduce":
        assert meta["plan_strategy"] == "none" and meta["workspace_peak_bytes"] == 0
        assert meta["scalar_steps"] == prepared.config["scalar_steps"]
    assert meta["process_cpu_ms"] == 0 and meta["process_gpu_ms"] == 0
    assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert torch.isnan(backing[1::2]).all()
    for tensor, original in zip((data, col, ptr, x), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="op"):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x, op="trans")
    with pytest.raises(ValueError, match="overlap"):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=data[:shape[0]])


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
], ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("threshold", [0, 8, 128, 256])
@pytest.mark.parametrize("alg", spmv_mod._csr_config.ROWQUEUE_ALGORITHMS)
def test_spmv_csr_short_rowqueue_thresholds(dtype, output_dtype, threshold, alg, record_property):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    lengths = [0, 1, 8, 9, 32, 33, 128, 129, 256, 257, 602, 713, 0]
    a, i, p, x, shape = _native_case(lengths, dtype, torch.int64, torch.int32, n=19)
    a.copy_(((torch.arange(a.numel(), device=a.device) % 7 - 3).float() / 16).to(dtype))
    x_storage = torch.ones(x.numel() * 3, dtype=dtype, device=x.device)
    x = x_storage[::3]
    out_storage = torch.full((shape[0] * 3,), float("nan"), dtype=output_dtype, device=x.device)
    out = out_storage[::3]
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype),
                           compute_dtype="float32", output_dtype=str(output_dtype), opA="non",
                           short_row_threshold=threshold).items():
        record_property(key, value)
    try:
        prepared = _new_prepared(a, i, p, shape, alg,
                                config=dict(short_row_threshold=threshold, block_nnz=64))
        for sign in (1, -1):
            x.fill_(sign)
            out.fill_(float("nan"))
            result, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out, return_meta=True)
            expected = golden_csr(a.double(), i, p, x.double(), shape).to(output_dtype)
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
            assert result is out and meta["row_queue_capacity"] == min(shape[0], a.numel() // (threshold + 1))
            assert meta["x_stride"] == (3,) and meta["y_stride"] == (3,)
            assert meta["queue_count_readback"] is False
        assert torch.isnan(out_storage[1::3]).all() and torch.isnan(out_storage[2::3]).all()
        assert torch.equal(x_storage[1::3], torch.ones_like(x_storage[1::3]))
        assert torch.equal(x_storage[2::3], torch.ones_like(x_storage[2::3]))
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32), (torch.float64, torch.float64),
    (torch.complex64, torch.complex64), (torch.complex128, torch.complex128),
], ids=lambda dtype: str(dtype).removeprefix("torch."))
@pytest.mark.parametrize("case", ["zero_rows", "empty", "ordinary", "mixed", "single_long", "queue_full", "medium_mix"])
@pytest.mark.parametrize("alg", ["csr_longrow_queue_atomic", "csr_longrow_grid_atomic", *spmv_mod._csr_config.ROWQUEUE_ALGORITHMS])
@pytest.mark.parametrize("index_pair", [(torch.int64, torch.int32)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
])
def test_spmv_csr_longrow_queue_native(dtype, output_dtype, case, index_pair, monkeypatch, alg, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    def forbidden(*args, **kwargs):
        raise AssertionError("long-row queue must not use a global row/split plan")
    monkeypatch.setattr(kernels, "build_plan", forbidden)
    lengths = {"zero_rows": [], "empty": [0] * 19, "ordinary": [0, 1, 31, 32, 0] * 5,
               "mixed": [0, 32, 33, 64, 1, 65, 257, 0, 31, 513, 7],
               "single_long": [4097 if _MIXED_QUICK_MODE else 65537],
               "queue_full": [33] * 129, "medium_mix": [0, 4, 128, 602, 713, 1024] * 23}[case]
    a, i, p, x, shape = _native_case(lengths, dtype, *index_pair, n=19)
    # Binary values keep the queue test focused on coverage, boundaries and races.
    values = (torch.arange(a.numel(), device=a.device) % 7 - 3).float() / 16
    if dtype.is_complex:
        values = torch.complex(values, values.flip(0) / 2)
    a.copy_(values.to(dtype))
    x.fill_(1)
    backing = torch.full((shape[0] * 2,), float("nan"), device=a.device, dtype=output_dtype)
    out = backing[::2]
    originals = [t.clone() for t in (a, i, p, x)]
    config = dict(long_row_threshold=32, segment_nnz=64, block_nnz=32, worker_programs=32)
    if alg == "csr_longrow_grid_atomic":
        config.pop("worker_programs")
    if alg in spmv_mod._csr_config.ROWQUEUE_ALGORITHMS:
        config = dict(short_row_threshold=32, rows_per_program=32, block_nnz=128)
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype),
                           output_dtype=str(output_dtype), opA="non", opB="non",
                           compute_dtype="float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32").items():
        record_property(key, value)
    try:
        prepared = _new_prepared(a, i, p, shape, alg, config=config)
        for sign in (1, -1):
            out.fill_(float("nan"))
            operand = x * sign
            result, meta = spmv_mod.flagsparse_spmv_csr_run(
                prepared, operand, out=out, return_meta=True, timing=True)
            ref_type = torch.complex128 if dtype.is_complex else torch.float64
            expected = golden_csr(a.to(ref_type), i, p, operand.to(ref_type), shape).to(output_dtype)
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
            assert result is out and meta["alg_resolved"] == alg
            assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
            assert meta["queue_count_readback"] is False
            assert meta["fused_plan_compute_ms"] >= 0
            assert meta["workspace_peak_bytes"] <= prepared.config["workspace_limit_bytes"]
            assert all(meta[k] >= 0 for k in ("init_ms", "producer_ms", "consumer_ms", "finish_ms"))
            if alg in spmv_mod._csr_config.ROWQUEUE_ALGORITHMS:
                assert meta["row_queue_capacity"] == min(shape[0], a.numel() // 33)
                assert meta["floating_point_atomic"] is False and meta["numeric_workspace_bytes"] == 0
                assert meta["plan_strategy"] == "per_run_row_ids_only"
                assert meta["finish_ms"] == 0 and meta["output_conversion"] == "final_store"
            if alg == "csr_longrow_grid_atomic":
                assert meta["task_acquisition"] == "static_program_id"
                assert meta["grid_capacity"] == meta["job_capacity"]
                assert all(meta[k] >= 0 for k in ("init_ms", "producer_ms", "consumer_ms", "finish_ms"))
        direct = spmv_mod.flagsparse_spmv_csr(a, i, p, -x, shape,
                    alg=alg, config=config, out_dtype=output_dtype)
        torch.testing.assert_close(direct, out, rtol=rtol, atol=atol)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    assert torch.isnan(backing[1::2]).all()
    for value, original in zip((a, i, p, x), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="op"):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x, op="trans")
    small = _new_prepared(a, i, p, shape, alg,
                          config=dict(config, workspace_limit_bytes=1))
    with pytest.raises(ValueError, match="workspace"):
        spmv_mod.flagsparse_spmv_csr_run(small, x, out=out)


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32), (torch.float64, torch.float64),
    (torch.complex64, torch.complex64), (torch.complex128, torch.complex128),
], ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("config", [None,
    dict(block_nnz=128, owner_search="full", single_row_fastpath=False, atomic_mode="all"),
    dict(block_nnz=512, owner_search="bounded", single_row_fastpath=False),
    dict(block_nnz=1024, atomic_mode="all"),
    dict(block_nnz=128, owner_search="full", single_row_fastpath=True),
], ids=["default", "full-scan-all", "bounded-scan", "fast-all", "full-fast-boundary"])
@pytest.mark.parametrize("index_pair", [(torch.int64, torch.int32)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
])
def test_spmv_csr_nnz_local_native(dtype, output_dtype, op, config, index_pair, monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    alg = "csr_nnz_local_reduce_atomic"
    def forbidden(*args, **kwargs):
        raise AssertionError("NNZ local reduction must not build a split/bucket plan")
    monkeypatch.setattr(kernels, "build_plan", forbidden)
    lengths = [0, 0, 128, 0, 127, 1, 129, 256, 257, 0, 1, 4097, 0, 0]
    if not _MIXED_QUICK_MODE:
        lengths += [65537] + [0] * 129 + [1, 1023, 1024, 1025]
    a, i, p, _, shape = _native_case(lengths, dtype, *index_pair, n=19)
    # Exact binary fractions isolate partition/atomic coverage from random cancellation.
    values = (torch.arange(a.numel(), device=a.device) % 7 - 3).float() / 16
    if dtype.is_complex:
        values = torch.complex(values, values.flip(0) / 2)
    a.copy_(values.to(dtype))
    inner, outer = (shape[1], shape[0]) if op == "non" else (shape[0], shape[1])
    x_store = torch.empty(inner * 2, dtype=dtype, device=a.device)
    x = x_store[::2]
    x.copy_(((torch.arange(inner, device=a.device) % 5 - 2).float() / 8).to(dtype))
    if dtype.is_complex:
        x.add_(0.25j)
    saved = [t.clone() for t in (a, i, p, x)]
    backing = torch.full((outer * 2,), float("nan"), dtype=output_dtype, device=a.device)
    out = backing[::2]
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype),
                           output_dtype=str(output_dtype), opA=op, opB="non",
                           compute_dtype="float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32").items():
        record_property(key, value)
    try:
        prepared = _new_prepared(a, i, p, shape, alg, op=op, config=config)
        for sign in (1, -1):
            out.fill_(float("nan"))
            actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x * sign, out=out, return_meta=True, timing=True)
            ref_type = torch.complex128 if dtype.is_complex else torch.float64
            expected = golden_csr(a.to(ref_type), i, p, (x * sign).to(ref_type), shape, op).to(output_dtype)
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            assert actual is out and meta["alg_resolved"] == alg
            assert meta["plan_strategy"] == "none" and meta["descriptor_peak_bytes"] == 0
            assert meta["ms"] == meta["process_cpu_ms"] + meta["gpu_ms"]
            assert meta["process_cpu_ms"] == 0
        direct = spmv_mod.flagsparse_spmv_csr(a, i, p, -x, shape, op=op, alg=alg, config=config, out_dtype=output_dtype)
        torch.testing.assert_close(direct, out, rtol=rtol, atol=atol)
        with pytest.raises(ValueError, match="op"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, op="trans" if op == "non" else "non")
        with pytest.raises(ValueError, match="overlap"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=a[:outer])
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    assert torch.isnan(backing[1::2]).all()
    for value, original in zip((a, i, p, x), saved):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.bfloat16, torch.float32), (torch.complex128, torch.complex128),
], ids=["bf16-wide", "complex128"])
@pytest.mark.parametrize("owner_search", ["bounded", "full"])
@pytest.mark.parametrize("atomic_mode", ["boundary", "all"])
def test_spmv_csr_nnz_local_branch_regression(dtype, output_dtype, owner_search, atomic_mode, record_property):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    # T=128: full single-owner tile; mixed 1+127 tile; one row spanning three
    # tiles; padded final single-owner tile. Empty rows exercise upper_bound.
    lengths = [0, 128, 0, 1, 127, 0, 384, 0, 17, 0]
    a, i, p, x, shape = _native_case(lengths, dtype, torch.int64, torch.int32, n=19)
    values = (torch.arange(a.numel(), device=a.device) % 5 + 1).float() / 16
    a.copy_((torch.complex(values, values / 2) if dtype.is_complex else values).to(dtype))
    x.fill_(0.5 + 0.25j if dtype.is_complex else 0.5)
    alg = "csr_nnz_local_reduce_atomic"
    config = dict(block_nnz=128, single_row_fastpath=True,
                  owner_search=owner_search, atomic_mode=atomic_mode)
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype),
                           output_dtype=str(output_dtype), opA="non", opB="non",
                           compute_dtype="float64" if dtype == torch.complex128 else "float32",
                           config=config, implementation_version=2).items():
        record_property(key, value)
    backing = torch.full((2 * shape[0],), float("nan"), dtype=output_dtype, device=a.device)
    out = backing[::2]
    ref_dtype = torch.complex128 if dtype.is_complex else torch.float64
    expected = golden_csr(a.to(ref_dtype), i, p, x.to(ref_dtype), shape).to(output_dtype)
    try:
        prepared = _new_prepared(a, i, p, shape, alg, config=config)
        for sign in (1, -1):
            out.fill_(float("nan"))
            actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x * sign, out=out, return_meta=True)
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(actual.cpu(), expected * sign, rtol=rtol, atol=atol)
            assert actual is out and meta["alg_resolved"] == alg
            assert meta["implementation_version"] == 2
            assert meta["config"]["single_row_fastpath"] is True
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    assert torch.isnan(backing[1::2]).all()


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32), (torch.float64, torch.float64),
    (torch.complex64, torch.complex64), (torch.complex128, torch.complex128),
], ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("grid_blocks", [1, 8, 256] if _MIXED_QUICK_MODE else [1, 8, 256, 1024])
@pytest.mark.parametrize("case", ["zero", "one", "before", "exact", "after", "rounds"])
def test_spmv_csr_rowqueue_strided_grid_boundaries(dtype, output_dtype, grid_blocks, case,
                                                monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable

    alg = "csr_short_rowqueue_strided_vector"
    queued = {"zero": 0, "one": 1, "before": grid_blocks - 1, "exact": grid_blocks,
              "after": grid_blocks + 1, "rounds": 2 * grid_blocks + 3}[case]
    # Even zero queued rows have positive capacity greater than the grid limit.
    # Varied row lengths/signs expose accumulator leakage across consumer rounds.
    lengths = [0, 1, 7, 32] * (grid_blocks + 2)
    lengths += [33 + 64 * (j % 3) for j in range(queued)]
    a, i, p, x, shape = _native_case(lengths, dtype, torch.int64, torch.int32, n=19)
    values = (torch.arange(a.numel(), device=a.device) % 7 - 3).float() / 16
    a.copy_((torch.complex(values, values / 2) if dtype.is_complex else values).to(dtype))
    x_storage = torch.ones(x.numel() * 3, dtype=dtype, device=x.device)
    vector = x_storage[::3]
    vector.fill_(0.5 + 0.25j if dtype.is_complex else 0.5)
    originals = [v.clone() for v in (a, i, p)]
    backing = torch.full((shape[0] * 3,), float("nan"), dtype=output_dtype, device=a.device)
    out = backing[::3]
    config = dict(grid_blocks=grid_blocks, short_row_threshold=32, block_nnz=64)
    captures = []
    consumer = kernels.short_rowqueue_strided_vector_kernel
    class CaptureLaunch:
        def __getitem__(self, grid):
            launch = consumer[grid]
            def call(*args, **kwargs):
                captures.append((grid, args[5], args[6]))
                return launch(*args, **kwargs)
            return call
    monkeypatch.setattr(kernels, "short_rowqueue_strided_vector_kernel", CaptureLaunch())
    def forbidden(*args, **kwargs):
        raise AssertionError("bounded row queue must not enter split/global-plan paths")
    monkeypatch.setattr(kernels, "build_plan", forbidden)
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype), output_dtype=str(output_dtype),
                           compute_dtype="float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32",
                           opA="non", config=config, expected_queued_rows=queued).items():
        record_property(key, value)
    try:
        prepared = _new_prepared(a, i, p, shape, alg, config=config)
        for diagnostic in (False, True):
            launches_before = len(captures)
            out.fill_(float("nan"))
            vector.neg_()
            operand_before = vector.clone()
            actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, vector, out=out,
                                                          timing=diagnostic, return_meta=True)
            ref_dtype = torch.complex128 if dtype.is_complex else torch.float64
            expected = golden_csr(a.to(ref_dtype), i, p, vector.to(ref_dtype), shape).to(output_dtype)
            rtol, atol = close_tolerances(output_dtype)
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            assert actual is out and meta["alg_resolved"] == alg
            assert len(captures) - launches_before == 1 + int(diagnostic)
            assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
            assert meta["consumer_grid_blocks"] == min(meta["row_queue_capacity"], grid_blocks)
            assert meta["consumer_grid_blocks"] == grid_blocks
            assert meta["consumer_grid_limit"] == grid_blocks
            assert meta["consumer_rounds_capacity"] == (meta["row_queue_capacity"] + grid_blocks - 1) // grid_blocks
            assert meta["task_acquisition"] == "static_grid_stride"
            assert meta["queue_count_readback"] is False and meta["floating_point_atomic"] is False
            assert meta["numeric_workspace_bytes"] == 0
            assert meta["x_stride"] == (3,) and meta["y_stride"] == (3,)
            torch.testing.assert_close(vector, operand_before, rtol=0, atol=0)
            # Test-only readback, after run; never part of the runtime or benchmark.
            grid, count, row_ids = captures[-1]
            assert grid == (grid_blocks,) and int(count.item()) == queued
            actual_ids = row_ids[:queued].cpu().sort().values
            expected_ids = torch.tensor([r for r, length in enumerate(lengths) if length > 32], dtype=torch.int64)
            torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        assert len(captures) == 3  # One full run, then full run + separate diagnostic.
        assert len({id(v[1]) for v in captures}) == 3 and len({id(v[2]) for v in captures}) == 3
        with pytest.raises(ValueError, match="op"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, vector, op="trans")
        with pytest.raises(ValueError, match="overlap"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, vector, out=a[:shape[0]], out_dtype=dtype)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    for value, original in zip((a, i, p), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert torch.isnan(backing[1::3]).all() and torch.isnan(backing[2::3]).all()
    assert torch.equal(x_storage[1::3], torch.ones_like(x_storage[1::3]))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32), (torch.float64, torch.float64),
    (torch.complex64, torch.complex64), (torch.complex128, torch.complex128),
], ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("parts", [1, 4, 8] if _MIXED_QUICK_MODE else [1, 2, 4, 8, 16])
@pytest.mark.parametrize("case", ["empty", "zero_cols", "no_queue", "single", "many", "partition_edges"])
@pytest.mark.parametrize("index_pair", [(torch.int64, torch.int32)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64), (torch.int64, torch.int32), (torch.int64, torch.int64)])
def test_spmv_csr_longrow_cooperative_native(dtype, output_dtype, parts, case, index_pair,
                                           monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable, cooperative_queue_capacity

    alg = "csr_longrow_cooperative_reduce"
    lengths = {
        "empty": [], "zero_cols": [0] * 11, "no_queue": [0, 1, 7, 31, 32] * 3,
        "single": [0, 1, 32 * (parts * 4 + 1) + 1, 0, 7],
        "many": [0, 32, 33, 1, 32 * parts + 1] * 7,
        "partition_edges": [0, 31, 32, 33, 63, 64, 65, 32 * parts - 1, 32 * parts, 32 * parts + 1],
    }[case]
    config = dict(long_row_threshold=32, cooperative_parts=parts, block_nnz=32,
                  grid_blocks=3, rows_per_program=4, lanes_per_row=8)
    a, i, p, x, shape = _native_case(lengths, dtype, *index_pair, n=0 if case == "zero_cols" else 19)
    # Binary-exact signed products permit inspection of every independent part,
    # including empty parts. Existing cancellation tests retain the shared oracle.
    values = ((torch.arange(a.numel(), device=a.device) % 7) - 3).float() / 16
    a.copy_((torch.complex(values, -values / 2) if dtype.is_complex else values).to(dtype))
    storage = torch.ones(x.numel() * 3, dtype=dtype, device=x.device)
    vector = storage[::3]
    vector.fill_(0.5 + 0.25j if dtype.is_complex else 0.5)
    saved = [v.clone() for v in (a, i, p, vector)]
    backing = torch.full((shape[0] * 3,), float("nan"), device=x.device, dtype=output_dtype)
    out = backing[::3]
    captures = []
    consumer = kernels.cooperative_row_consumer_kernel
    class CaptureLaunch:
        def __getitem__(self, grid):
            launch = consumer[grid]
            def call(*args, **kwargs):
                # Test-only poisoning: all consumed slots must be explicitly
                # written by their owner, including parts with no NNZ.
                args[6].fill_(float("nan"))
                captures.append((grid, args[4], args[5], args[6]))
                return launch(*args, **kwargs)
            return call
    monkeypatch.setattr(kernels, "cooperative_row_consumer_kernel", CaptureLaunch())
    def forbidden(*args, **kwargs):
        raise AssertionError("cooperative rows must not use segment descriptors or another algorithm")
    for name in ("build_plan", "run_longrow_queue", "run_longrow_grid", "run_short_rowqueue_strided"):
        monkeypatch.setattr(kernels, name, forbidden)
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype), output_dtype=str(output_dtype),
                           compute_dtype=spmv_mod._csr_config.compute_dtype(alg, dtype),
                           component_dtype="float64" if dtype in (torch.float32, torch.float64, torch.complex128) else "float32",
                           opA="non", config=config, implementation_version=1).items():
        record_property(key, value)
    try:
        prepared = _new_prepared(a, i, p, shape, alg, config=config)
        stats = cooperative_queue_capacity(shape[0], a.numel(), prepared.config, dtype)
        grid = min(stats["row_queue_capacity"], config["grid_blocks"])
        ref_dtype = torch.complex128 if dtype.is_complex else torch.float64
        expected = golden_csr(a.to(ref_dtype), i, p, vector.to(ref_dtype), shape).to(output_dtype)
        rtol, atol = close_tolerances(output_dtype)
        for diagnostic in (False, True):
            before = len(captures)
            out.fill_(float("nan"))
            actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, vector, out=out,
                                                          out_dtype=output_dtype, timing=diagnostic, return_meta=True)
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            assert actual is out and meta["alg_resolved"] == alg
            assert len(captures) - before == (1 + int(diagnostic)) * bool(grid)
            assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
            assert meta["cooperative_parts"] == parts and meta["row_grid_blocks"] == grid
            assert meta["consumer_grid_blocks"] == grid * parts
            assert meta["floating_point_atomic"] is False and meta["queue_count_readback"] is False
            assert meta["task_acquisition"] == "static_grid_stride_parts"
            assert meta["plan_strategy"] == "per_run_row_ids_fixed_parts"
            for key, value in stats.items():
                assert meta[key] == value
            if grid:
                launch_grid, count, row_ids, partial = captures[-1]
                assert launch_grid == (grid, parts)
                q = int(count.item())  # Accuracy-only reads after run.
                expected_ids = [r for r, length in enumerate(lengths) if length > 32]
                ids = row_ids[:q].cpu().tolist()
                assert sorted(ids) == expected_ids and q == len(expected_ids)
                used = partial[:q * parts * (2 if dtype.is_complex else 1)].cpu()
                assert torch.isfinite(used).all()
                if dtype.is_complex:
                    used = torch.view_as_complex(used.reshape(q, parts, 2))
                else:
                    used = used.reshape(q, parts)
                ptr_cpu, values_cpu, col_cpu, rhs_cpu = p.cpu(), a.to(ref_dtype).cpu(), i.long().cpu(), vector.to(ref_dtype).cpu()
                for slot, row in enumerate(ids):
                    start, stop = int(ptr_cpu[row]), int(ptr_cpu[row + 1])
                    chunks = (stop - start + 31) // 32
                    for part in range(parts):
                        begin = min(stop, start + (chunks * part // parts) * 32)
                        end = min(stop, start + (chunks * (part + 1) // parts) * 32)
                        reference = (values_cpu[begin:end] * rhs_cpu[col_cpu[begin:end]]).sum().to(used.dtype)
                        torch.testing.assert_close(used[slot, part], reference, rtol=0, atol=0)
        # High-level prepared call reaches the same implementation and output.
        actual = spmv_mod.flagsparse_spmv_csr(prepared=prepared, x=vector, out=out, out_dtype=output_dtype)
        assert actual is out
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        if grid:
            assert len(captures) == 4
            for position in (1, 2, 3):
                assert len({id(record[position]) for record in captures}) == 4
        with pytest.raises(ValueError, match="op"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, vector, op="trans")
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, vector, config={"cooperative_parts": 2 if parts != 2 else 4})
        if a.numel() >= shape[0] > 0:
            with pytest.raises(ValueError, match="overlap"):
                spmv_mod.flagsparse_spmv_csr_run(prepared, vector, out=a[:shape[0]], out_dtype=dtype)
        too_small = dict(config, workspace_limit_bytes=stats["workspace_peak_bytes"] - 1)
        small = _new_prepared(a, i, p, shape, alg, config=too_small)
        with pytest.raises(ValueError, match="workspace"):
            spmv_mod.flagsparse_spmv_csr_run(small, vector, out=out, out_dtype=output_dtype)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    for original, snapshot in zip((a, i, p, vector), saved):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
    assert torch.isnan(backing[1::3]).all() and torch.isnan(backing[2::3]).all()


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype,output_dtype", [
    pytest.param(torch.float32, torch.float32, id="regular_f32"),
    pytest.param(torch.float64, torch.float64, id="regular_f64"),
    pytest.param(torch.complex64, torch.complex64, id="regular_c32"),
    pytest.param(torch.complex128, torch.complex128, id="regular_c64"),
    pytest.param(torch.float16, torch.float16, id="mixed_f16_f16"),
    pytest.param(torch.float16, torch.float32, id="mixed_f16_f32"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="mixed_bf16_bf16"),
    pytest.param(torch.bfloat16, torch.float32, id="mixed_bf16_f32"),
])
@pytest.mark.parametrize("parts", [1, 4] if _MIXED_QUICK_MODE else [1, 4, 16])
@pytest.mark.parametrize("case", ["empty", "zero_cols", "short_only", "medium_only", "long_only", "mixed", "tail"])
@pytest.mark.parametrize("index_pair", [(torch.int64, torch.int32)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64), (torch.int64, torch.int32), (torch.int64, torch.int64)])
def test_spmv_csr_three_tier_native(dtype, output_dtype, parts, case, index_pair,
                                   monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable, three_tier_queue_capacity

    alg = "csr_three_tier_queue_reduce"
    lengths = {
        "empty": [], "zero_cols": [0] * 35, "short_only": [0, 1, 7, 8] * 9,
        "medium_only": [9, 63, 64] * 13, "long_only": [65, 129, 1025] * 3,
        "mixed": [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 513] * 3,
        "tail": [0] * 128 + [8, 64, 64 * (parts * 3 + 1) + 1],
    }[case]
    cfg = dict(short_row_threshold=8, long_row_threshold=64, rows_per_program=128,
               block_nnz=64, cooperative_parts=parts, long_grid_blocks=2)
    a, i, p, x, shape = _native_case(lengths, dtype, *index_pair, n=0 if case == "zero_cols" else 19)
    values = ((torch.arange(a.numel(), device=a.device) % 7) - 3).float() / 16
    a.copy_((torch.complex(values, values / 2) if dtype.is_complex else values).to(dtype))
    storage = torch.ones(x.numel() * 3, device=x.device, dtype=dtype)
    vector = storage[::3]
    values_x = (torch.arange(x.numel(), device=x.device) % 5).float() / 8
    vector.copy_((torch.complex(values_x, values_x / 2) if dtype.is_complex else values_x).to(dtype))
    originals = [v.clone() for v in (a, i, p, vector)]
    backing = torch.full((shape[0] * 3,), float("nan"), device=x.device, dtype=output_dtype)
    out = backing[::3]
    captures, partials = [], []
    producer, consumer = kernels.three_tier_producer_kernel, kernels.cooperative_row_consumer_kernel
    class CaptureProducer:
        def __getitem__(self, grid):
            launch = producer[grid]
            def call(*args, **kwargs):
                captures.append((args[5], args[6], args[7]))
                return launch(*args, **kwargs)
            return call
    class CaptureLong:
        def __getitem__(self, grid):
            launch = consumer[grid]
            def call(*args, **kwargs):
                args[6].fill_(float("nan"))  # Only accuracy tests poison partial storage.
                partials.append((args[4], args[6]))
                return launch(*args, **kwargs)
            return call
    monkeypatch.setattr(kernels, "three_tier_producer_kernel", CaptureProducer())
    monkeypatch.setattr(kernels, "cooperative_row_consumer_kernel", CaptureLong())
    def forbidden(*args, **kwargs):
        raise AssertionError("three-tier route must not call another algorithm or build segment plans")
    for name in ("build_plan", "run_short_rowqueue", "run_short_rowqueue_strided", "run_longrow_cooperative", "run_longrow_queue"):
        monkeypatch.setattr(kernels, name, forbidden)
    for key, value in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype), output_dtype=str(output_dtype),
                           compute_dtype=spmv_mod._csr_config.compute_dtype(alg, dtype), opA="non", config=cfg).items():
        record_property(key, value)
    try:
        prepared = _new_prepared(a, i, p, shape, alg, config=cfg)
        stats = three_tier_queue_capacity(shape[0], a.numel(), prepared.config, dtype)
        ref_dtype = torch.complex128 if dtype.is_complex else torch.float64
        expected = golden_csr(a.to(ref_dtype), i, p, vector.to(ref_dtype), shape).to(output_dtype)
        rtol, atol = close_tolerances(output_dtype)
        for diagnostic in (False, True):
            out.fill_(float("nan"))
            actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, vector, out=out, out_dtype=output_dtype,
                                                          timing=diagnostic, return_meta=True)
            assert actual is out and meta["alg_resolved"] == alg
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
            assert not meta["queue_count_readback"] and not meta["floating_point_atomic"]
            assert meta["plan_strategy"] == "per_run_three_tier_row_ids"
            for key, value in stats.items():
                assert meta[key] == value
            if diagnostic:
                assert meta["consumer_ms"] == meta["medium_consumer_ms"] + meta["long_consumer_ms"]
            if shape[0]:
                count, mids, longs = captures[-1]
                qm, ql = count.cpu().tolist()  # Test-only ownership audit.
                assert sorted(mids[:qm].cpu().tolist()) == [r for r, n in enumerate(lengths) if 8 < n <= 64]
                assert sorted(longs[:ql].cpu().tolist()) == [r for r, n in enumerate(lengths) if n > 64]
            if partials:
                count, partial = partials[-1]
                q = int(count.item())
                assert torch.isfinite(partial[:q * parts * (2 if dtype.is_complex else 1)]).all()
        actual = spmv_mod.flagsparse_spmv_csr(prepared=prepared, x=vector, out=out, out_dtype=output_dtype)
        assert actual is out
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        if shape[0]:
            assert len(captures) == 4  # full, full+diagnostic, high-level call
            for position in range(3):
                assert len({id(c[position]) for c in captures}) == 4
        with pytest.raises(ValueError, match="op"):
            spmv_mod.flagsparse_spmv_csr_run(prepared, vector, op="trans")
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, vector, config={"short_row_threshold": 16})
        if a.numel() >= shape[0] > 0:
            with pytest.raises(ValueError, match="overlap"):
                spmv_mod.flagsparse_spmv_csr_run(prepared, vector, out=a[:shape[0]], out_dtype=dtype)
        small = _new_prepared(a, i, p, shape, alg, config=dict(cfg, workspace_limit_bytes=stats["workspace_peak_bytes"] - 1))
        with pytest.raises(ValueError, match="workspace"):
            spmv_mod.flagsparse_spmv_csr_run(small, vector, out=out, out_dtype=output_dtype)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    for value, snapshot in zip((a, i, p, vector), originals):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    assert torch.isnan(backing[1::3]).all() and torch.isnan(backing[2::3]).all()
    assert torch.equal(storage[1::3], torch.ones_like(storage[1::3]))


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_spmv_csr_three_tier_mixed_wide_output(dtype, record_property):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    alg = "csr_three_tier_queue_reduce"
    a, i, p, x, shape = _native_case([2, 64, 2049], dtype, torch.int64, torch.int32, n=19)
    a.zero_()
    delta = 2 ** (-11 if dtype == torch.float16 else -9)
    for start, end in ((0, 2), (2, 66), (66, 2115)):
        a[start], a[end - 1] = 1, delta
    x.fill_(1)
    for k, v in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype), compute_dtype="float32",
                     output_dtype="float32", opA="non").items():
        record_property(k, v)
    try:
        route = _new_prepared(a, i, p, shape, alg)
        actual = spmv_mod.flagsparse_spmv_csr_run(route, x, out_dtype=torch.float32)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
    expected = torch.full((3,), 1 + delta, dtype=torch.float32, device=x.device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, actual.to(dtype).float())


@pytest.mark.spmv_csr
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wide", [False, True])
def test_spmv_csr_three_tier_mixed_index_fallback(dtype, wide, monkeypatch, record_property):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    alg = "csr_three_tier_queue_reduce"
    a, i, p, x, shape = _native_case([2, 64, 2049], dtype, torch.int64, torch.int64, n=19)
    a.fill_(0.125)
    x.fill_(0.5)
    output_dtype = torch.float32 if wide else dtype
    out = torch.full((shape[0],), float("nan"), dtype=output_dtype, device=x.device)
    implementation = kernels.run_three_tier_queue
    calls = []
    def inject(route, vector, target, timing, phase):
        calls.append(route.kernel_indices.dtype)
        assert route.alg == alg and target is out and target.dtype == output_dtype
        if route.kernel_indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 kernel indices")
        return implementation(route, vector, target, timing, phase)
    monkeypatch.setattr(kernels, "run_three_tier_queue", inject)
    for k, v in dict(algorithm=alg, a_dtype=str(dtype), rhs_dtype=str(dtype), output_dtype=str(output_dtype),
                     compute_dtype="float32", opA="non").items():
        record_property(k, v)
    try:
        route = _new_prepared(a, i, p, shape, alg)
        actual, meta = spmv_mod.flagsparse_spmv_csr_run(route, x, out=out, out_dtype=output_dtype, return_meta=True)
        assert actual is out and calls == [torch.int64, torch.int32]
        assert meta["index_fallback_applied"] and meta["alg_resolved"] == alg
        assert meta["config"] == route.config and meta["output_dtype"] == str(output_dtype).removeprefix("torch.")
        expected = golden_csr(a.double(), i, p, x.double(), shape).to(output_dtype)
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        route.index_fallback_policy = "strict"
        with pytest.raises(RuntimeError, match="unsupported int64"):
            spmv_mod.flagsparse_spmv_csr_run(route, x, out=out, out_dtype=output_dtype)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
