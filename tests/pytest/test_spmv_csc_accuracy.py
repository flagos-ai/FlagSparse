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
    if spmv_csc_mod._is_ascend_runtime():
        pytest.skip("Ascend uses the index_add path instead of the Triton fallback")
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
            raise RuntimeError("unsupported int64 kernel indices (injected)")
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
    if spmv_csc_mod._is_ascend_runtime():
        pytest.skip("Ascend uses the index_add path instead of the Triton fallback")
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


# Explicit migrated algorithms share this operator's marker, golden and tolerance.
_MIGRATED_ROUTES = (('csc_col_subgroup', 'trans'), ('csc_col_subgroup', 'conj'), ('csc_col_vector', 'trans'), ('csc_col_vector', 'conj'), ('csc_col_split_reduce', 'trans'), ('csc_col_split_reduce', 'conj'), ('csc_col_tile_atomic', 'non'))


@pytest.mark.spmv_csc
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES, ids=[f"{a}-{o}" for a, o in _MIGRATED_ROUTES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128], ids=["float16", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])

def test_spmv_csc_migrated_routes(alg, op, dtype, index_dtype):
    import importlib
    from tests.pytest.conftest import QUICK_MODE

    module = importlib.import_module("flagsparse.sparse_operations.spmv_csc")
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
    order = torch.argsort(cols, stable=True)
    values = values[order]
    first = rows[order].to(device=device, dtype=index_dtype)
    counts = torch.bincount(cols, minlength=shape[1])
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
        prepared = module.prepare_spmv_csc(data, first, second, shape, op=op, alg=alg)
        backing = torch.full((expected.numel() * 2,), float("nan"), dtype=dtype, device=device)
        out = backing[::2]
        result = module.flagsparse_spmv_csc_run(prepared, operand, out=out)
        direct = module.flagsparse_spmv_csc(data, first, second, operand, shape, op=op, alg=alg)
    except NotImplementedError as exc:
        pytest.skip(f"backend capability unavailable: {exc}")
    assert result is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(direct.cpu(), expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="op"):
        module.flagsparse_spmv_csc_run(prepared, operand, op="trans" if op == "non" else "non")


@pytest.mark.spmv_csc
@pytest.mark.parametrize("dtype,output_dtype", [(torch.complex128, torch.complex128), (torch.float16, torch.float16), (torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)])
def test_spmv_csc_split_bounded_workspace(dtype, output_dtype):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        import importlib

        module = importlib.import_module("flagsparse.sparse_operations.spmv_csc")
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
        operand = torch.ones(19, dtype=golden_dtype)
        operand = operand.to(dtype)
        expected = (matrix @ operand.to(golden_dtype)).to(output_dtype)
        cfg = dict(segment_nnz=8, reduce_block_size=2, workspace_bytes=128, block_nnz=8, block_n=1)
        from flagsparse.sparse_operations._common import _backend_name
        if 'csc_col_split_reduce' not in module.list_spmv_csc_algorithms(
            op='trans', dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = module.prepare_spmv_csc(data.to(device), index.to(device), ptr.to(device), (19, 5),
            op="trans", alg="csc_col_split_reduce", config=cfg)
        actual, meta = module.flagsparse_spmv_csc_run(prepared, operand.to(device), out_dtype=output_dtype, return_meta=True, timing=True)
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        assert meta["workspace_peak_bytes"] <= cfg["workspace_bytes"]
        assert meta["segment_batches"] > 1
        assert meta["reduction_levels"] >= 0
        wider = dict(cfg, workspace_bytes=65536)
        again, wider_meta = module.flagsparse_spmv_csc_run(prepared, operand.to(device), config=wider, out_dtype=output_dtype, return_meta=True)
        torch.testing.assert_close(again.cpu(), expected, rtol=rtol, atol=atol)
        assert wider_meta["reduction_levels"] > 1
        assert meta["process_cpu_ms"] == 0
        assert meta["operator_ms"] == meta["gpu_ms"]
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_csc
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5), (3, 5)])
def test_spmv_csc_migrated_empty(alg, op, shape):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmv_csc")
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.float32, device=device)
    first = torch.empty(0, dtype=torch.int64, device=device)
    second = torch.zeros(shape[1] + 1, dtype=torch.int32, device=device)
    input_rows, output_rows = shape[1], shape[0]
    if op != "non":
        input_rows, output_rows = output_rows, input_rows
    operand = torch.ones((input_rows,), device=device)
    out = torch.full((output_rows,), float("nan"), device=device)
    try:
        prepared = module.prepare_spmv_csc(data, first, second, shape, op=op, alg=alg)
        result = module.flagsparse_spmv_csc_run(prepared, operand, out=out)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert result is out
    assert torch.equal(result.cpu(), torch.zeros_like(result.cpu()))


from tests.pytest.conftest import QUICK_MODE as _MIXED_QUICK_MODE


@pytest.mark.parametrize("index_pair", [(torch.int32, torch.int64)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
], ids=lambda pair: "/".join(str(dtype).removeprefix("torch.") for dtype in pair))
@pytest.mark.spmv_csc
@pytest.mark.parametrize("alg,op", [('csc_col_tile_atomic', 'non'), ('csc_col_subgroup', 'trans'), ('csc_col_subgroup', 'conj'), ('csc_col_vector', 'trans'), ('csc_col_vector', 'conj'), ('csc_col_split_reduce', 'trans'), ('csc_col_split_reduce', 'conj'), ('spmv_csc_base', 'non'), ('spmv_csc_base', 'trans'), ('spmv_csc_base', 'conj')])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True], ids=["same_out", "f32_out"])
def test_spmv_csc_registered_half_combinations(alg, op, dtype, wide, index_pair):
    """Exercise the selected kernel, including a result not exactly representable in half."""
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        import importlib

        module = importlib.import_module("flagsparse.sparse_operations.spmv_csc")
        device = accelerator_device()
        values = torch.tensor([1, 0.00390625, -0.125, 0.5], dtype=dtype, device=device)
        first = torch.tensor([0, 0, 0, 2], dtype=index_pair[0], device=device)
        second = torch.tensor([0, 2, 3, 4, 4], dtype=index_pair[1], device=device)
        matrix = torch.zeros((3, 4), dtype=torch.float64)
        matrix.index_put_(
            (torch.tensor([0, 0, 0, 2]), torch.tensor([0, 0, 1, 2])),
            values.cpu().double(), accumulate=True,
        )
        inner = 4 if op == "non" else 3
        effective = torch.full((inner,), 1.0078125, dtype=dtype, device=device)
        operand = effective
        extra = {}
        output_dtype = torch.float32 if wide else dtype
        expected = ((matrix if op == "non" else matrix.T) @ effective.cpu().double()).to(output_dtype)
        backing = torch.full((*expected.shape[:-1], expected.shape[-1] * 2),
                             float("nan"), dtype=output_dtype, device=device)
        out = backing[..., ::2]
        originals = [tensor.clone() for tensor in (values, first, second, operand)]
        from flagsparse.sparse_operations._common import _backend_name
        if alg not in module.list_spmv_csc_algorithms(
            op=op, dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = module.prepare_spmv_csc(values, first, second, (3, 4), op=op, alg=alg)
        result = module.flagsparse_spmv_csc_run(
            prepared, operand, out=out, out_dtype=output_dtype, **extra
        )
        assert result is out
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
        direct = module.flagsparse_spmv_csc(
            values, first, second, operand, (3, 4), alg=alg, op=op,
            out_dtype=output_dtype, **extra,
        )
        torch.testing.assert_close(direct, result, rtol=rtol, atol=atol)
        for actual, original in zip((values, first, second, operand), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        conflicting = dtype if wide else torch.float32
        with pytest.raises(ValueError, match="conflicts"):
            module.flagsparse_spmv_csc_run(
                prepared, operand, out=out, out_dtype=conflicting, **extra
            )
        with pytest.raises(ValueError, match="overlap"):
            module.flagsparse_spmv_csc_run(prepared, operand, out=values)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
