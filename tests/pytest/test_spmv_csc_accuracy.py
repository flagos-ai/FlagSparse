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

import pytest
import torch

from flagsparse import flagsparse_spmv_csc, prepare_spmv_csc
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import SPMV_MN_SHAPES

spmv_csc_mod = importlib.import_module("flagsparse.sparse_operations.spmv_csc")
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


def _random_values(shape, dtype, device):
    if dtype in (torch.float16, torch.float32, torch.float64):
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
    if dtype in (torch.float16, torch.float32):
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _random_csc_mn(M, N, dtype, index_dtype, device):
    golden = golden_device()
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=golden) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, N), dtype, golden),
        torch.zeros((), dtype=dtype, device=golden),
    )
    rows, cols = torch.nonzero(mask, as_tuple=True)
    order = torch.argsort(cols * max(1, M) + rows)
    rows = rows[order]
    cols = cols[order]
    data = dense[rows, cols].contiguous()
    col_counts = torch.bincount(cols, minlength=N)
    indptr = torch.zeros(N + 1, dtype=torch.int64, device=golden)
    indptr[1:] = torch.cumsum(col_counts, dim=0)
    return (
        data.to(device),
        rows.to(index_dtype).contiguous().to(device),
        indptr.to(index_dtype).to(device),
        dense,
    )


def _make_x(length, dtype, device):
    return _random_values((length,), dtype, device)


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


def _tol(dtype):
    # Same policy as test_spmv_csr_accuracy: fp32 and complex64 accumulate in
    # fp32-precision components and carry ordinary order-dependent fp32 SpMV error.
    # The golden reference now runs on CPU in fp64, so it no longer happens to share
    # the kernel's summation order -- a 1.3e-6 tolerance turned that into a flake at
    # 160x1024. fp64 and complex128 accumulate in fp64 and keep the strict tolerance.
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.complex64):
        return 1e-3, 1e-3
    return close_tolerances(dtype)


def _assert_close(actual, expected, dtype):
    rtol, atol = _tol(dtype)
    ref_dtype = _reference_dtype(dtype)
    golden = golden_device()
    assert torch.allclose(
        actual.to(device=golden, dtype=ref_dtype),
        expected.to(device=golden, dtype=ref_dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.spmv_csc
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize(
    "name,dtype", _value_dtype_cases(), ids=[c[0] for c in _value_dtype_cases()]
)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmv_csc_matches_dense_reference(M, N, name, dtype, index_dtype, op):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csc_mn(M, N, dtype, index_dtype, device)
    x_len = M if _op_transposes(op) else N
    x = _make_x(x_len, dtype, golden_device())
    ref_dtype = _reference_dtype(dtype)
    ref = (_apply_dense_op(dense, op).to(ref_dtype) @ x.to(ref_dtype)).to(dtype)

    out = flagsparse_spmv_csc(
        data,
        indices,
        indptr,
        x.to(device),
        shape=(M, N),
        op=op,
        index_fallback_policy="auto",
    )
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_csc
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmv_csc_prepared_path_matches_dense_reference(op):
    device = accelerator_device()
    M, N = 8, 10
    dtype = torch.complex64
    data, indices, indptr, dense = _random_csc_mn(M, N, dtype, torch.int32, device)
    prepared = prepare_spmv_csc(data, indices, indptr, (M, N), op=op)
    x_len = M if _op_transposes(op) else N
    x = _make_x(x_len, dtype, golden_device())
    ref_dtype = _reference_dtype(dtype)
    ref = (_apply_dense_op(dense, op).to(ref_dtype) @ x.to(ref_dtype)).to(dtype)

    out = flagsparse_spmv_csc(x=x.to(device), prepared=prepared)
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_csc
def test_spmv_csc_prepared_transpose_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csc_mn(
        8, 10, torch.float32, torch.int32, device
    )
    prepared = prepare_spmv_csc(data, indices, indptr, (8, 10), transpose=True)
    x = torch.randn(10, dtype=torch.float32, device=golden_device()).to(device)
    with pytest.raises(ValueError, match="does not match prepared.transpose"):
        flagsparse_spmv_csc(x=x, prepared=prepared, transpose=False)


@pytest.mark.spmv_csc
def test_spmv_csc_prepared_op_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csc_mn(
        8, 10, torch.complex64, torch.int32, device
    )
    prepared = prepare_spmv_csc(data, indices, indptr, (8, 10), op="conj")
    x = _make_x(8, torch.complex64, golden_device()).to(device)
    with pytest.raises(ValueError, match="does not match prepared.op"):
        flagsparse_spmv_csc(x=x, prepared=prepared, op="trans")


@pytest.mark.spmv_csc
def test_spmv_csc_int64_auto_fallback_to_int32(monkeypatch):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csc_mn(
        12, 9, torch.float32, torch.int64, device
    )
    x = torch.randn(9, dtype=torch.float32, device=golden_device())
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    x = x.to(device)
    state = {"forced_once": False}
    original = spmv_csc_mod._triton_spmv_csc_kernel

    def fail_int64_once(prepared, x_in, op_code):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, x_in, op_code)

    monkeypatch.setattr(spmv_csc_mod, "_triton_spmv_csc_kernel", fail_int64_once)
    out = flagsparse_spmv_csc(
        data,
        indices,
        indptr,
        x,
        shape=(12, 9),
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    _assert_close(out, ref.to(torch.float32), torch.float32)


@pytest.mark.spmv_csc
def test_spmv_csc_int64_strict_no_fallback(monkeypatch):
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csc_mn(
        12, 9, torch.float32, torch.int64, device
    )
    x = torch.randn(9, dtype=torch.float32, device=golden_device()).to(device)
    original = spmv_csc_mod._triton_spmv_csc_kernel

    def fail_int64(prepared, x_in, op_code):
        if prepared.kernel_indices.dtype == torch.int64:
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, x_in, op_code)

    monkeypatch.setattr(spmv_csc_mod, "_triton_spmv_csc_kernel", fail_int64)
    with pytest.raises(RuntimeError, match="forced int64 launch failure"):
        flagsparse_spmv_csc(
            data,
            indices,
            indptr,
            x,
            shape=(12, 9),
            index_fallback_policy="strict",
        )


from tests.pytest.conftest import QUICK_MODE


@pytest.mark.spmv_csc
@pytest.mark.parametrize("alg,op", [("csc_col_tile_atomic", "non"), ("csc_col_subgroup", "trans"), ("csc_col_subgroup", "conj")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("index_dtype,ptr_dtype", [(torch.int32, torch.int64)] if QUICK_MODE else
                         [(i, p) for i in (torch.int32, torch.int64) for p in (torch.int32, torch.int64)])

def test_spmv_csc_registered_columns(alg, op, dtype, index_dtype, ptr_dtype):
    from flagsparse.sparse_operations._spmm_csr_config import resolve_csc_config
    from flagsparse.sparse_operations._spmm_csr_runtime import backend_caps
    from flagsparse import prepare_spmv_csc, flagsparse_spmv_csc_run
    device = accelerator_device()
    try:
        resolve_csc_config(alg, str(dtype).removeprefix("torch."), 7, backend_caps(device), op=op)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    lengths = [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 0, 257 if QUICK_MODE else 2049]
    ptr = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    rows = (torch.arange(int(ptr[-1])) * 7 + 3) % 19
    torch.manual_seed(20261010)
    values = _random_values((rows.numel(),), dtype, golden_device())
    values[::17] = 0
    matrix = torch.zeros((19, len(lengths)), dtype=_reference_dtype(dtype), device=golden_device())
    cols = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    matrix.index_put_((rows, cols), values.to(matrix.dtype), accumulate=True)
    data, indices, indptr = values.to(device), rows.to(device=device, dtype=index_dtype), ptr.to(device=device, dtype=ptr_dtype)
    dense_rows = len(lengths) if op == "non" else 19
    dense_cpu = _random_values((dense_rows,), dtype, golden_device())
    dense = dense_cpu.to(device)
    effective = matrix if op == "non" else matrix.T if op == "trans" else matrix.conj().T
    expected = effective @ dense_cpu.to(matrix.dtype)
    prepared = prepare_spmv_csc(data, indices, indptr, (19, len(lengths)), op=op, alg=alg)
    before = data.clone()
    output = torch.full(expected.shape, float("nan"), dtype=dtype, device=device)
    result, meta = flagsparse_spmv_csc_run(prepared, dense, out=output, timing=True, return_meta=True)
    assert result is output
    _assert_close(result, expected, dtype)
    assert torch.equal(data, before)
    assert meta["op_total_ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert meta["alg_resolved"] == alg
    assert not hasattr(prepared, "col_ids")
    with pytest.raises(ValueError, match="op"):
        flagsparse_spmv_csc_run(prepared, dense, op="trans" if op == "non" else "non")
    with pytest.raises(ValueError):
        flagsparse_spmv_csc_run(prepared, dense, config={"num_warps": 3})


@pytest.mark.spmv_csc
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (7, 0), (7, 5)])
def test_spmv_csc_registered_empty(op, shape):
    from flagsparse import prepare_spmv_csc, flagsparse_spmv_csc_run
    from flagsparse.sparse_operations._spmm_csr_runtime import backend_caps
    from flagsparse.sparse_operations._spmm_csr_config import resolve_csc_config
    alg = 'csc_col_tile_atomic' if op == "non" else 'csc_col_subgroup'
    device = accelerator_device()
    try:
        resolve_csc_config(alg, "float32", 7, backend_caps(device), op=op)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    a = torch.empty(0, device=device)
    i = torch.empty(0, dtype=torch.int64, device=device)
    p = torch.zeros(shape[1] + 1, dtype=torch.int32, device=device)
    prepared = prepare_spmv_csc(a, i, p, shape, alg=alg, op=op)
    length = shape[1] if op == "non" else shape[0]
    dense = torch.empty((length,), device=device)
    result = flagsparse_spmv_csc_run(prepared, dense)
    assert result.shape[0] == (shape[0] if op == "non" else shape[1])
    assert torch.count_nonzero(result).item() == 0


@pytest.mark.spmv_csc
@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_csc_registered_index_fallback(monkeypatch, policy):
    from flagsparse import flagsparse_spmv_csc_run
    device = accelerator_device()
    a = torch.ones(2, device=device)
    i = torch.tensor([0, 1], dtype=torch.int64, device=device)
    p = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    prepared = prepare_spmv_csc(a, i, p, (2, 2), alg="csc_col_subgroup", op="trans", index_fallback_policy=policy)
    original = spmv_csc_mod._launch_csc_columns
    def inject(data, indices, indptr, *args):
        if indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 index type")
        return original(data, indices, indptr, *args)
    monkeypatch.setattr(spmv_csc_mod, "_launch_csc_columns", inject)
    x = torch.ones(2, device=device)
    if policy == "strict":
        with pytest.raises(RuntimeError, match="int64"):
            flagsparse_spmv_csc_run(prepared, x)
    else:
        result, meta = flagsparse_spmv_csc_run(prepared, x, return_meta=True)
        _assert_close(result, x, torch.float32)
        assert meta["index_fallback_applied"]
        assert meta["alg_resolved"] == "csc_col_subgroup"
        prepared.int32_safe = False
        with pytest.raises(RuntimeError, match="unsafe"):
            flagsparse_spmv_csc_run(prepared, x)


@pytest.mark.spmv_csc
def test_spmv_csc_direct_route_does_not_prepare_plan(monkeypatch):
    from flagsparse import flagsparse_spmv_csc_run
    device = accelerator_device()
    a = torch.ones(2, device=device)
    i = torch.tensor([0, 1], dtype=torch.int32, device=device)
    p = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    prepared = prepare_spmv_csc(a, i, p, (2, 2), alg="csc_col_subgroup", op="trans")
    def forbidden(*args, **kwargs):
        raise AssertionError("direct CSC must not prepare a legacy plan")
    monkeypatch.setattr(spmv_csc_mod, "prepare_spmv_csc", forbidden)
    for timing in (False, True):
        result = flagsparse_spmv_csc_run(prepared, torch.ones(2, device=device), timing=timing)
        assert torch.equal(result, torch.ones_like(result))


@pytest.fixture(autouse=True)
def _skip_unknown_csc_extension_capabilities(request):
    if request.node.name.startswith(("test_spmv_csc_registered_index", "test_spmv_csc_direct_route")):
        from flagsparse.sparse_operations._spmm_csr_config import resolve_csc_config
        from flagsparse.sparse_operations._spmm_csr_runtime import backend_caps
        try:
            resolve_csc_config("csc_col_subgroup", "float32", 1, backend_caps(accelerator_device()), op="trans")
        except NotImplementedError as exc:
            pytest.skip(str(exc))


@pytest.mark.spmv_csc
def test_spmv_csc_registered_base_rebuilds_each_run(monkeypatch):
    from flagsparse import flagsparse_spmv_csc_run
    device = accelerator_device()
    a = torch.ones(2, device=device)
    i = torch.tensor([0, 1], dtype=torch.int32, device=device)
    p = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    prepared = prepare_spmv_csc(a, i, p, (2, 2), alg="spmv_csc_base")
    original = spmv_csc_mod.prepare_spmv_csc
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(spmv_csc_mod, "prepare_spmv_csc", observed)
    x = torch.ones(2, device=device)
    flagsparse_spmv_csc_run(prepared, x)
    assert len(calls) == 1
    flagsparse_spmv_csc_run(prepared, x, timing=True)
    assert len(calls) == 3
    assert all(not call["_allow_delegate"] for call in calls)
