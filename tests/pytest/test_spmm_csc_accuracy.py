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

from flagsparse import flagsparse_spmm_csc, prepare_spmm_csc_route
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)


spmm_csc_mod = importlib.import_module("flagsparse.sparse_operations.spmm_csc")
pytestmark = pytest.mark.skipif(
    not accelerator_available(),
    reason=ACCELERATOR_REQUIRED,
)

SPMM_CSC_MKN_SHAPES = ((7, 5, 4), (16, 32, 8), (64, 96, 16))


def _value_dtype_cases():
    return [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device) * 0.125
    if dtype == torch.complex64:
        return torch.complex(
            torch.randn(shape, dtype=torch.float32, device=device) * 0.125,
            torch.randn(shape, dtype=torch.float32, device=device) * 0.125,
        )
    if dtype == torch.complex128:
        return torch.complex(
            torch.randn(shape, dtype=torch.float64, device=device) * 0.125,
            torch.randn(shape, dtype=torch.float64, device=device) * 0.125,
        )
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _op_transposes(op):
    return op in ("trans", "conj")


def _logical_b_rows(M, K, op):
    return int(M) if _op_transposes(op) else int(K)


def _logical_out_rows(M, K, op):
    return int(K) if _op_transposes(op) else int(M)


def _dense_reference(dense, B, dtype, op):
    ref_dtype = _reference_dtype(dtype)
    dense_ref = dense.to(ref_dtype)
    B_ref = B.to(ref_dtype)
    if op == "non":
        return (dense_ref @ B_ref).to(dtype)
    if op == "trans":
        return (dense_ref.T @ B_ref).to(dtype)
    if op == "conj":
        return (dense_ref.conj().T @ B_ref).to(dtype)
    raise ValueError(f"unsupported op: {op}")


def _dense_to_csc(dense, index_dtype):
    device = dense.device
    M, K = dense.shape
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    order = torch.argsort(cols * max(1, M) + rows)
    rows = rows[order]
    cols = cols[order]
    data = dense[rows, cols].contiguous()
    col_counts = torch.bincount(cols, minlength=K)
    indptr = torch.zeros(K + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(col_counts, dim=0)
    return data, rows.to(index_dtype).contiguous(), indptr.to(index_dtype).contiguous()


def _random_csc_mk(M, K, dtype, index_dtype, device):
    """CSC arrays on ``device``, dense oracle copy on CPU.

    Built on the golden device: ``torch.where`` has no muDNN kernel for
    float64/complex on Moore Threads, so generating the matrix on the accelerator
    failed before the operator under test ran.  See ``golden_device()``.
    """
    golden = golden_device()
    p = min(0.25, max(0.06, 32.0 / max(M * K, 1)))
    mask = torch.rand(M, K, device=golden) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, K), dtype, golden),
        torch.zeros((), dtype=dtype, device=golden),
    )
    data, indices, indptr = _dense_to_csc(dense, index_dtype)
    return data.to(device), indices.to(device), indptr.to(device), dense


def _assert_close(actual, expected, dtype):
    rtol, atol = close_tolerances(dtype)
    ref_dtype = _reference_dtype(dtype)
    golden = golden_device()
    assert torch.allclose(
        actual.to(device=golden, dtype=ref_dtype),
        expected.to(device=golden, dtype=ref_dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.spmm_csc
@pytest.mark.parametrize("M, K, N", SPMM_CSC_MKN_SHAPES)
@pytest.mark.parametrize(
    "name,dtype", _value_dtype_cases(), ids=[case[0] for case in _value_dtype_cases()]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("layout", ["row", "col"], ids=["row", "col"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmm_csc_matches_dense_reference(M, K, N, name, dtype, index_dtype, layout, op):
    del name
    device = accelerator_device()
    data, indices, indptr, dense = _random_csc_mk(
        M, K, dtype, index_dtype, device
    )
    b_rows = _logical_b_rows(M, K, op)
    B = _random_values((b_rows, N), dtype, golden_device())
    ref = _dense_reference(dense, B, dtype, op)
    B_dev = B.to(device)
    if layout == "col":
        B_col = torch.empty_strided(
            (b_rows, N), (1, max(1, b_rows)), dtype=dtype, device=device
        )
        B_col.copy_(B_dev)
        B_dev = B_col
    out = flagsparse_spmm_csc(
        data,
        indices,
        indptr,
        B_dev,
        shape=(M, K),
        op=op,
        index_fallback_policy="auto",
    )
    assert out.shape == (_logical_out_rows(M, K, op), N)
    _assert_close(out, ref, dtype)


@pytest.mark.spmm_csc
def test_spmm_csc_prepared_path_and_meta():
    device = accelerator_device()
    M, K, N = 7, 5, 4
    dtype = torch.complex64
    data, indices, indptr, dense = _random_csc_mk(
        M, K, dtype, torch.int32, device
    )
    op = "conj"
    prepared = prepare_spmm_csc_route(data, indices, indptr, (M, K), op=op)
    B = _random_values((M, N), dtype, golden_device())
    ref = _dense_reference(dense, B, dtype, op)
    out, meta = spmm_csc_mod.flagsparse_spmm_csc_run(
        prepared,
        B.to(device),
        op=op,
        return_meta=True,
        timing=True,
    )
    assert out.shape == (K, N)
    assert meta["alg"] == "spmm_csc_base"
    assert meta["op"] == op
    assert meta["logical_shape"] == (M, K)
    assert meta["process_cpu_ms"] == 0.0
    assert meta["process_gpu_ms"] >= 0.0
    assert meta["compute_ms"] >= 0.0
    _assert_close(out, ref, dtype)


@pytest.mark.spmm_csc
@pytest.mark.parametrize("op", ["non", "trans"], ids=["non", "trans"])
def test_spmm_csc_B_length_mismatch_rejected(op):
    device = accelerator_device()
    M, K, N = 8, 12, 4
    data, indices, indptr, _dense = _random_csc_mk(
        M, K, torch.float32, torch.int32, device
    )
    good_rows = _logical_b_rows(M, K, op)
    B = torch.randn(
        (good_rows - 1, N), dtype=torch.float32, device=golden_device()
    ).to(device)
    with pytest.raises(ValueError, match="B.shape\\[0\\] must be"):
        flagsparse_spmm_csc(data, indices, indptr, B, shape=(M, K), op=op)


@pytest.mark.spmm_csc
def test_spmm_csc_prepared_op_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csc_mk(
        8, 10, torch.float32, torch.int32, device
    )
    prepared = prepare_spmm_csc_route(data, indices, indptr, (8, 10), op="non")
    B = torch.randn((8, 4), dtype=torch.float32, device=golden_device()).to(device)
    with pytest.raises(ValueError, match="does not match"):
        spmm_csc_mod.flagsparse_spmm_csc_run(prepared, B, op="trans")


@pytest.mark.spmm_csc
@pytest.mark.parametrize("op", ["trans", "conj"], ids=["trans", "conj"])
def test_spmm_csc_transpose_family_high_level(op):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csc_mk(
        8, 12, torch.float32, torch.int32, device
    )
    B = torch.randn((8, 4), dtype=torch.float32, device=golden_device())
    ref = _dense_reference(dense, B, torch.float32, op)
    out = flagsparse_spmm_csc(
        data, indices, indptr, B.to(device), shape=(8, 12), op=op
    )
    assert out.shape == (12, 4)
    _assert_close(out, ref, torch.float32)


@pytest.mark.spmm_csc
@pytest.mark.parametrize("op", ["non", "trans"], ids=["non", "trans"])
def test_spmm_csc_int64_auto_fallback_to_int32(monkeypatch, op):
    if spmm_csc_mod._is_ascend_runtime():
        pytest.skip("Ascend uses the index_add path instead of the Triton fallback")
    device = accelerator_device()
    M, K, N = 7, 5, 4
    dtype = torch.float32
    data, indices, indptr, dense = _random_csc_mk(
        M, K, dtype, torch.int64, device
    )
    B = _random_values((_logical_b_rows(M, K, op), N), dtype, golden_device())
    ref = _dense_reference(dense, B, dtype, op)
    B = B.to(device)
    state = {"forced_once": False}
    original = spmm_csc_mod._triton_spmm_csc_base_kernel

    def fail_int64_once(prepared, B_in, op_code=None):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("unsupported int64 kernel indices (injected)")
        return original(prepared, B_in, op_code)

    monkeypatch.setattr(spmm_csc_mod, "_triton_spmm_csc_base_kernel", fail_int64_once)
    out = flagsparse_spmm_csc(
        data,
        indices,
        indptr,
        B,
        shape=(M, K),
        op=op,
        index_fallback_policy="auto",
    )
    assert state["forced_once"] is True
    _assert_close(out, ref, dtype)


@pytest.mark.spmm_csc
def test_spmm_csc_int64_strict_no_fallback(monkeypatch):
    if spmm_csc_mod._is_ascend_runtime():
        pytest.skip("Ascend uses the index_add path instead of the Triton fallback")
    device = accelerator_device()
    M, K, N = 7, 5, 4
    data, indices, indptr, _dense = _random_csc_mk(
        M, K, torch.float32, torch.int64, device
    )
    B = _random_values((K, N), torch.float32, device)
    original = spmm_csc_mod._triton_spmm_csc_base_kernel

    def fail_int64(prepared, B_in, op_code=None):
        if prepared.kernel_indices.dtype == torch.int64:
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, B_in, op_code)

    monkeypatch.setattr(spmm_csc_mod, "_triton_spmm_csc_base_kernel", fail_int64)
    with pytest.raises(RuntimeError, match="forced int64 launch failure"):
        flagsparse_spmm_csc(
            data,
            indices,
            indptr,
            B,
            shape=(M, K),
            index_fallback_policy="strict",
        )


# Explicit migrated algorithms share this operator's marker, golden and tolerance.
_MIGRATED_ROUTES = (('csc_col_panel', 'trans'), ('csc_col_panel', 'conj'), ('csc_col_kparallel', 'trans'), ('csc_col_kparallel', 'conj'), ('csc_col_split_reduce', 'trans'), ('csc_col_split_reduce', 'conj'), ('csc_col_tile_panel_atomic', 'non'))


@pytest.mark.spmm_csc
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES, ids=[f"{a}-{o}" for a, o in _MIGRATED_ROUTES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128], ids=["float16", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_csc_migrated_routes(alg, op, dtype, index_dtype, layout):
    import importlib
    from tests.pytest.conftest import QUICK_MODE

    module = importlib.import_module("flagsparse.sparse_operations.spmm_csc")
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
        prepared = module.prepare_spmm_csc_route(data, first, second, shape, op=op, alg=alg)
        backing = torch.full((expected.shape[0], expected.shape[1] * 2), float("nan"), dtype=dtype, device=device)
        out = backing[:, ::2]
        result = module.flagsparse_spmm_csc_run(prepared, operand, out=out)
        direct = module.flagsparse_spmm_csc(data, first, second, operand, shape, op=op, alg=alg)
    except NotImplementedError as exc:
        pytest.skip(f"backend capability unavailable: {exc}")
    assert result is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(direct.cpu(), expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="op"):
        module.flagsparse_spmm_csc_run(prepared, operand, op="trans" if op == "non" else "non")


@pytest.mark.spmm_csc
@pytest.mark.parametrize("dtype,output_dtype", [(torch.complex128, torch.complex128), (torch.float16, torch.float16), (torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)])
def test_spmm_csc_split_bounded_workspace(dtype, output_dtype):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        import importlib

        module = importlib.import_module("flagsparse.sparse_operations.spmm_csc")
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
        expected = (matrix @ operand.to(golden_dtype)).to(output_dtype)
        cfg = dict(segment_nnz=8, reduce_block_size=2, workspace_bytes=256, block_nnz=8, block_n=4)
        from flagsparse.sparse_operations._common import _backend_name
        if 'csc_col_split_reduce' not in module.list_spmm_csc_algorithms(
            op='trans', dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = module.prepare_spmm_csc_route(data.to(device), index.to(device), ptr.to(device), (19, 5),
            op="trans", alg="csc_col_split_reduce", config=cfg)
        actual, meta = module.flagsparse_spmm_csc_run(prepared, operand.to(device), out_dtype=output_dtype, return_meta=True, timing=True)
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        assert meta["workspace_peak_bytes"] <= cfg["workspace_bytes"]
        assert meta["segment_batches"] > 1
        assert meta["reduction_levels"] >= 0
        wider = dict(cfg, workspace_bytes=65536)
        again, wider_meta = module.flagsparse_spmm_csc_run(prepared, operand.to(device), config=wider, out_dtype=output_dtype, return_meta=True)
        torch.testing.assert_close(again.cpu(), expected, rtol=rtol, atol=atol)
        assert wider_meta["reduction_levels"] > 1
        assert meta["process_cpu_ms"] == 0
        assert meta["operator_ms"] == meta["gpu_ms"]
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmm_csc
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5), (3, 5)])
def test_spmm_csc_migrated_empty(alg, op, shape):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmm_csc")
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.float32, device=device)
    first = torch.empty(0, dtype=torch.int64, device=device)
    second = torch.zeros(shape[1] + 1, dtype=torch.int32, device=device)
    input_rows, output_rows = shape[1], shape[0]
    if op != "non":
        input_rows, output_rows = output_rows, input_rows
    operand = torch.ones((input_rows, 7), device=device)
    out = torch.full((output_rows, 7), float("nan"), device=device)
    try:
        prepared = module.prepare_spmm_csc_route(data, first, second, shape, op=op, alg=alg)
        result = module.flagsparse_spmm_csc_run(prepared, operand, out=out)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert result is out
    assert torch.equal(result.cpu(), torch.zeros_like(result.cpu()))


from tests.pytest.conftest import QUICK_MODE as _MIXED_QUICK_MODE


@pytest.mark.parametrize("index_pair", [(torch.int32, torch.int64)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
], ids=lambda pair: "/".join(str(dtype).removeprefix("torch.") for dtype in pair))
@pytest.mark.spmm_csc
@pytest.mark.parametrize("alg,op", [('csc_col_tile_panel_atomic', 'non'), ('csc_col_panel', 'trans'), ('csc_col_panel', 'conj'), ('csc_col_kparallel', 'trans'), ('csc_col_kparallel', 'conj'), ('csc_col_split_reduce', 'trans'), ('csc_col_split_reduce', 'conj'), ('spmm_csc_base', 'non'), ('spmm_csc_base', 'trans'), ('spmm_csc_base', 'conj')])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True], ids=["same_out", "f32_out"])
@pytest.mark.parametrize("op_b", ["non", "trans", "conj"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_csc_registered_half_combinations(alg, op, dtype, wide, op_b, layout, index_pair):
    """Exercise the selected kernel, including a result not exactly representable in half."""
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        import importlib

        module = importlib.import_module("flagsparse.sparse_operations.spmm_csc")
        device = accelerator_device()
        values = torch.tensor([1, 0.00390625, -0.125, 0.5], dtype=dtype, device=device)
        first = torch.tensor([0, 0, 0, 2], dtype=index_pair[0], device=device)
        second = torch.tensor([0, 2, 3, 4, 4], dtype=index_pair[1], device=device)
        matrix = torch.zeros((3, 4), dtype=torch.float64)
        matrix.index_put_(
            (torch.tensor([0, 0, 0, 2]), torch.tensor([0, 0, 1, 2])),
            values.cpu().double(), accumulate=True,
        )
        n = 7 if _MIXED_QUICK_MODE else 33
        inner = 4 if op == "non" else 3
        effective = torch.full((inner, n), 1.0078125, dtype=dtype, device=device)
        if layout == "col":
            effective = effective.T.contiguous().T
        operand = effective if op_b == "non" else effective.T
        extra = {"op_b": op_b}
        output_dtype = torch.float32 if wide else dtype
        expected = ((matrix if op == "non" else matrix.T) @ effective.cpu().double()).to(output_dtype)
        backing = torch.full((*expected.shape[:-1], expected.shape[-1] * 2),
                             float("nan"), dtype=output_dtype, device=device)
        out = backing[..., ::2]
        originals = [tensor.clone() for tensor in (values, first, second, operand)]
        from flagsparse.sparse_operations._common import _backend_name
        if alg not in module.list_spmm_csc_algorithms(
            op=op, dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = module.prepare_spmm_csc_route(values, first, second, (3, 4), op=op, alg=alg)
        result = module.flagsparse_spmm_csc_run(
            prepared, operand, out=out, out_dtype=output_dtype, **extra
        )
        assert result is out
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
        direct = module.flagsparse_spmm_csc(
            values, first, second, operand, (3, 4), alg=alg, op=op,
            out_dtype=output_dtype, **extra,
        )
        torch.testing.assert_close(direct, result, rtol=rtol, atol=atol)
        for actual, original in zip((values, first, second, operand), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        conflicting = dtype if wide else torch.float32
        with pytest.raises(ValueError, match="conflicts"):
            module.flagsparse_spmm_csc_run(
                prepared, operand, out=out, out_dtype=conflicting, **extra
            )
        with pytest.raises(ValueError, match="overlap"):
            module.flagsparse_spmm_csc_run(prepared, operand, out=values, op_b=op_b)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))
