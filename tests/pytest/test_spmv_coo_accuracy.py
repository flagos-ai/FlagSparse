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
    flagsparse_spmv_coo,
    flagsparse_spmv_coo_tocsr,
    prepare_spmv_coo,
    prepare_spmv_coo_tocsr,
)
import flagsparse.sparse_operations.spmv_coo as spmv_coo_mod
from flagsparse.sparse_operations import _common as common
from tests import reference_utils

from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import (
    SPMV_COO_DTYPES,
    SPMV_COO_DTYPE_IDS,
    SPMV_MN_SHAPES,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)


_TOCSR_DTYPES = (torch.float32, torch.float64)
_TOCSR_DTYPE_IDS = ("float32", "float64")


def _random_dense(shape, dtype, device):
    if dtype in (torch.float16, torch.float32, torch.float64):
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
    if dtype in (torch.float16, torch.float32):
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _random_coo_mn(M, N, dtype, device):
    """COO arrays on ``device``, dense oracle copy on CPU.

    Built on the golden device with only the operator's inputs copied over:
    ``torch.where`` has no muDNN kernel for float64/complex on Moore Threads.  The
    sparse tensor stays on CPU too -- torch.sparse has no working matmul on MUSA, so
    a reference that needs it (``test_spmv_coo_tocsr_*``) must evaluate there.
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
    sp = dense.to_sparse_coo().coalesce()
    return sp.values().to(device), sp.indices().to(device), dense


def _tol(dtype):
    # Same policy as test_spmv_csr_accuracy: fp32 and complex64 accumulate in
    # fp32-precision components and carry ordinary order-dependent fp32 SpMV error.
    # The golden reference now runs on CPU in fp64, so it no longer happens to share
    # the kernel's summation order -- a 1.3e-6 tolerance turned that into a flake at
    # 160x1024. fp64 and complex128 accumulate in fp64 and keep the strict tolerance.
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.complex64):
        return 1e-3, 1e-3
    return close_tolerances(dtype)


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
    """Compare on the golden device; the reference never leaves CPU."""
    rtol, atol = _tol(dtype)
    ref_dtype = _reference_dtype(dtype)
    golden = golden_device()
    assert torch.allclose(
        actual.to(device=golden, dtype=ref_dtype),
        expected.to(device=golden, dtype=ref_dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.spmv_coo
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("dtype", SPMV_COO_DTYPES, ids=SPMV_COO_DTYPE_IDS)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
@pytest.mark.parametrize("alg", [None, "coo_segmented_atomic", "coo_rowrun_subgroup"], ids=["legacy", "segmented", "subgroup"])
def test_spmv_coo_matches_dense_reference(M, N, dtype, index_dtype, op, alg):
    device = accelerator_device()
    data, indices, dense = _random_coo_mn(M, N, dtype, device)
    row = indices[0].to(index_dtype).contiguous()
    col = indices[1].to(index_dtype).contiguous()
    x_len = M if _op_transposes(op) else N
    x = _make_x(x_len, dtype, golden_device())
    ref_dtype = _reference_dtype(dtype)
    if common._use_scipy_accuracy_reference():
        matrix = reference_utils.scipy_coo(
            data, indices[0], indices[1], (M, N), ref_dtype
        )
        ref = reference_utils.as_torch(
            reference_utils.spmv(matrix, x, ref_dtype, op=op), ref_dtype, golden_device()
        ).to(dtype)
    else:
        ref = (_apply_dense_op(dense, op).to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
    out = flagsparse_spmv_coo(data, row, col, x.to(device), shape=(M, N), op=op, alg=alg)
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_coo
def test_spmv_coo_prepared_reuses_structure_across_ops():
    device = accelerator_device()
    data, indices, dense = _random_coo_mn(8, 10, torch.complex64, device)
    row = indices[0].to(torch.int32).contiguous()
    col = indices[1].to(torch.int32).contiguous()
    prepared = prepare_spmv_coo(data, row, col, (8, 10))

    x_non = _make_x(10, torch.complex64, golden_device())
    ref_non = dense.to(torch.complex128) @ x_non.to(torch.complex128)
    out_non = flagsparse_spmv_coo(x=x_non.to(device), prepared=prepared, op="non")
    _assert_close(out_non, ref_non.to(torch.complex64), torch.complex64)

    x_trans = _make_x(8, torch.complex64, golden_device())
    ref_trans = dense.t().to(torch.complex128) @ x_trans.to(torch.complex128)
    out_trans = flagsparse_spmv_coo(x=x_trans.to(device), prepared=prepared, op="trans")
    _assert_close(out_trans, ref_trans.to(torch.complex64), torch.complex64)

    ref_conj = dense.conj().t().to(torch.complex128) @ x_trans.to(torch.complex128)
    out_conj = flagsparse_spmv_coo(x=x_trans.to(device), prepared=prepared, op="conj")
    _assert_close(out_conj, ref_conj.to(torch.complex64), torch.complex64)


@pytest.mark.spmv_coo
@pytest.mark.parametrize("op", ["trans", "conj"], ids=["trans", "conj"])
def test_spmv_coo_runtime_launch_matches_public_api_for_ops(op):
    device = accelerator_device()
    data, indices, _dense = _random_coo_mn(7, 9, torch.complex64, device)
    row = indices[0].to(torch.int32).contiguous()
    col = indices[1].to(torch.int32).contiguous()
    x = _make_x(7, torch.complex64, golden_device()).to(device)

    expected = flagsparse_spmv_coo(data, row, col, x, shape=(7, 9), op=op)
    launch = spmv_coo_mod._prepare_spmv_coo_launch_from_raw(
        data=data,
        row=row,
        col=col,
        shape=(7, 9),
        sort_by_row=True,
        op=op,
    )
    actual = spmv_coo_mod._run_spmv_coo_prepared_with_fallback(
        launch,
        x,
        block_size=256,
        num_warps=4,
        block_inner=128,
    )
    _assert_close(actual, expected, torch.complex64)


@pytest.mark.spmv_coo
def test_spmv_coo_prepared_explicit_transpose_conflict_rejected():
    device = accelerator_device()
    data, indices, _dense = _random_coo_mn(8, 10, torch.float32, device)
    row = indices[0].to(torch.int32).contiguous()
    col = indices[1].to(torch.int32).contiguous()
    prepared = prepare_spmv_coo(data, row, col, (8, 10), transpose=True)
    x = torch.randn(10, dtype=torch.float32, device=golden_device()).to(device)
    with pytest.raises(ValueError, match="transpose conflicts with op"):
        flagsparse_spmv_coo(x=x, prepared=prepared, op="non", transpose=True)


@pytest.mark.spmv_coo
def test_spmv_coo_int64_auto_fallback_to_int32(monkeypatch):
    device = accelerator_device()
    data, indices, dense = _random_coo_mn(12, 9, torch.float32, device)
    row = indices[0].to(torch.int64).contiguous()
    col = indices[1].to(torch.int64).contiguous()
    x = torch.randn(9, dtype=torch.float32, device=golden_device())
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    x = x.to(device)
    state = {"forced_once": False}
    original = spmv_coo_mod._triton_spmv_coo_kernel

    def fail_int64_once(prepared, x_in, block_size, num_warps, block_inner):
        if prepared.row.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("int64 index unsupported (injected compatibility failure)")
        return original(prepared, x_in, block_size, num_warps, block_inner)

    monkeypatch.setattr(spmv_coo_mod, "_triton_spmv_coo_kernel", fail_int64_once)
    out = flagsparse_spmv_coo(
        data,
        row,
        col,
        x,
        shape=(12, 9),
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(
        out.to(device=ref.device, dtype=torch.float64), ref, rtol=rtol, atol=atol
    )


@pytest.mark.spmv_coo_tocsr
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("dtype", _TOCSR_DTYPES, ids=_TOCSR_DTYPE_IDS)
def test_spmv_coo_tocsr_matches_torch(M, N, dtype):
    device = accelerator_device()
    data, indices, dense = _random_coo_mn(M, N, dtype, device)
    row = indices[0].contiguous()
    col = indices[1].contiguous()
    x = _random_dense((N,), dtype, golden_device())
    # torch.sparse stays the reference this test is named for, but evaluated on CPU:
    # it has no working matmul on MUSA (aten::addmm is unregistered for Sparsemusa).
    Asp = dense.to_sparse_coo().coalesce()
    ref = torch.sparse.mm(Asp, x.unsqueeze(1)).squeeze(1)
    out = flagsparse_spmv_coo_tocsr(data, row, col, x.to(device), shape=(M, N))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spmv_coo_tocsr
def test_spmv_coo_tocsr_prepared_path_matches_torch():
    device = accelerator_device()
    M, N = 8, 10
    dtype = torch.float32
    data, indices, dense = _random_coo_mn(M, N, dtype, device)
    row = indices[0].contiguous()
    col = indices[1].contiguous()
    x = _random_dense((N,), dtype, golden_device())
    prepared = prepare_spmv_coo_tocsr(data, row, col, (M, N))
    Asp = dense.to_sparse_coo().coalesce()
    ref = torch.sparse.mm(Asp, x.unsqueeze(1)).squeeze(1)
    out = flagsparse_spmv_coo_tocsr(x=x.to(device), prepared=prepared)
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spmv_coo
@pytest.mark.parametrize("alg", ["coo_segmented_atomic", "coo_rowrun_subgroup"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("ordered", [False, True], ids=["unsorted", "sorted"])
def test_spmv_coo_registered_boundaries(alg, op, ordered, monkeypatch):
    from flagsparse import flagsparse_spmv_coo_run
    device = accelerator_device()
    # A long row spans NNZ tiles; row 3 is empty; duplicate coordinates remain.
    row = torch.tensor([0] * 259 + [2, 1, 2, 4, 0], dtype=torch.int64)
    col = torch.arange(row.numel(), dtype=torch.int32) % 7
    values = (torch.arange(row.numel()) % 5 - 2).to(torch.float64)
    values = torch.complex(values, values.flip(0) / 4)
    if not ordered:
        order = torch.arange(row.numel() - 1, -1, -1)
        row, col, values = row[order], col[order], values[order]
    dense = torch.zeros((5, 7), dtype=values.dtype)
    dense.index_put_((row, col.long()), values, accumulate=True)
    x = torch.ones(7 if op == "non" else 5, dtype=values.dtype) * (1 + 2j)
    expected = _apply_dense_op(dense, op) @ x
    a, r, c, x = values.to(device), row.to(device), col.to(device), x.to(device)
    prepared = prepare_spmv_coo(a, r, c, (5, 7), alg=alg, op=op)
    original = spmv_coo_mod._coo_sorted_runs
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(spmv_coo_mod, "_coo_sorted_runs", counted)
    out = torch.full(expected.shape, float("nan"), dtype=values.dtype, device=device)
    for timing in (False, True):
        y, meta = flagsparse_spmv_coo_run(prepared, x, out=out, return_meta=True, timing=timing)
        assert y is out
        _assert_close(y, expected, values.dtype)
        assert meta["operator_ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert len(calls) == (0 if "atomic" in alg else 3)
    assert torch.equal(a.cpu(), values)
    with pytest.raises(ValueError, match="op conflicts"):
        flagsparse_spmv_coo_run(prepared, x, op="trans" if op == "non" else "non")
    with pytest.raises(ValueError):
        flagsparse_spmv_coo_run(prepared, x, config={"invalid": 1})


@pytest.mark.spmv_coo
@pytest.mark.parametrize("alg", ["coo_segmented_atomic", "coo_rowrun_subgroup"])
@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (5, 0), (5, 7)])
def test_spmv_coo_registered_empty(alg, shape):
    device = accelerator_device()
    values = torch.empty(0, device=device)
    indices = torch.empty(0, dtype=torch.int64, device=device)
    y = flagsparse_spmv_coo(values, indices, indices, torch.ones(shape[1], device=device),
                            shape, alg=alg)
    assert y.shape == (shape[0],)
    assert torch.count_nonzero(y) == 0


@pytest.fixture(autouse=True)
def _coo_extension_capability_guard(request):
    from flagsparse.sparse_operations import spmv_coo as native
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    alg = parameters.get("alg")
    if alg not in native._COO_NEW_ALGORITHMS:
        return
    dtype = parameters.get("dtype", torch.complex128)
    try:
        native._resolve_coo_config(alg, dtype, accelerator_device())
    except NotImplementedError as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_coo
@pytest.mark.parametrize("policy", ["auto", "strict"])
@pytest.mark.parametrize("alg", ["coo_segmented_atomic"])
def test_registered_coo_index_fallback_is_same_algorithm(monkeypatch, policy, alg):
    device = accelerator_device()
    a = torch.tensor([2., 3.], device=device)
    r = torch.tensor([0, 1], dtype=torch.int64, device=device)
    c = torch.tensor([1, 0], dtype=torch.int64, device=device)
    x = torch.ones(2, device=device)
    original = spmv_coo_mod._launch_coo_extension
    calls = []
    def injected(data, row, col, starts, B, shape, selected, config, conj=False):
        calls.append((selected, row.dtype))
        if row.dtype == torch.int64:
            raise RuntimeError("int64 index unsupported (injected)")
        return original(data, row, col, starts, B, shape, selected, config, conj)
    monkeypatch.setattr(spmv_coo_mod, "_launch_coo_extension", injected)
    if policy == "strict":
        with pytest.raises(RuntimeError, match="int64 index unsupported"):
            flagsparse_spmv_coo(a, r, c, x, (2, 2), alg=alg, index_fallback_policy=policy)
    else:
        result = flagsparse_spmv_coo(a, r, c, x, (2, 2), alg=alg, index_fallback_policy=policy)
        _assert_close(result, a, a.dtype)
        assert calls == [(alg, torch.int64), (alg, torch.int32)]
    unsafe = spmv_coo_mod._PreparedCooLaunch(a[:1], r[:1] + 2**31, c[:1], (2**31 + 1, 2))
    with pytest.raises(RuntimeError, match="unsafe"):
        spmv_coo_mod._spmv_coo_prepared_with_int32_indices(unsafe, "int64 unsupported")
