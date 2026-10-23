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
    if dtype in (torch.float32, torch.float64):
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
    if dtype == torch.float32:
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
def test_spmv_coo_matches_dense_reference(M, N, dtype, index_dtype, op):
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
    out = flagsparse_spmv_coo(data, row, col, x.to(device), shape=(M, N), op=op)
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
    if spmv_coo_mod._is_ascend_runtime():
        pytest.skip("Ascend uses the index_add path instead of the Triton fallback")
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
            raise RuntimeError("forced int64 launch failure")
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


# Explicit migrated algorithms share this operator's marker, golden and tolerance.
_MIGRATED_ROUTES = (('coo_segmented_atomic', 'non'), ('coo_segmented_atomic', 'trans'), ('coo_segmented_atomic', 'conj'), ('coo_rowrun_subgroup', 'non'), ('coo_rowrun_subgroup', 'trans'), ('coo_rowrun_subgroup', 'conj'), ('coo_segmented_stream_atomic', 'non'), ('coo_segmented_stream_atomic', 'trans'), ('coo_segmented_stream_atomic', 'conj'))


@pytest.mark.spmv_coo
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES, ids=[f"{a}-{o}" for a, o in _MIGRATED_ROUTES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128], ids=["float16", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])

def test_spmv_coo_migrated_routes(alg, op, dtype, index_dtype):
    import importlib
    from tests.pytest.conftest import QUICK_MODE

    module = importlib.import_module("flagsparse.sparse_operations.spmv_coo")
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
    operand = (torch.arange(input_rows, dtype=torch.float64) % 7 - 3) / 8
    if dtype.is_complex:
        operand = torch.complex(operand, operand.flip(0) / 2)
    operand = operand.to(dtype)
    expected_matrix = dense if op == "non" else dense.T.conj() if op == "conj" else dense.T
    expected = (expected_matrix @ operand.to(reference_dtype)).to(dtype)
    operand = operand.to(device)

    try:
        prepared = module.prepare_spmv_coo(data, first, second, shape, op=op, alg=alg)
        backing = torch.full((expected.numel() * 2,), float("nan"), dtype=dtype, device=device)
        out = backing[::2]
        result = module.flagsparse_spmv_coo_run(prepared, operand, out=out)
        direct = module.flagsparse_spmv_coo(data, first, second, operand, shape, op=op, alg=alg)
    except NotImplementedError as exc:
        pytest.skip(f"backend capability unavailable: {exc}")
    assert result is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(direct.cpu(), expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="op"):
        module.flagsparse_spmv_coo_run(prepared, operand, op="trans" if op == "non" else "non")


    with pytest.raises(ValueError, match="sort_by_row"):
        module.prepare_spmv_coo(data, first, second, shape, op=op, alg=alg,
                                sort_by_row=alg != "coo_rowrun_subgroup")

@pytest.mark.spmv_coo
@pytest.mark.parametrize("blocks,reduce", [(1, "segment"), (2, "segment"), (4, "segment"), (8, "segment"), (2, "none")])
@pytest.mark.parametrize("dtype,output_dtype", [(torch.float32, torch.float32), (torch.complex64, torch.complex64), (torch.float16, torch.float16), (torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_coo_stream_carry_boundaries(blocks, reduce, dtype, output_dtype, op):
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        import importlib

        module = importlib.import_module("flagsparse.sparse_operations.spmv_coo")
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
        operand = torch.ones(matrix.shape[1], dtype=golden_dtype)
        operand = operand.to(dtype)
        expected = (matrix @ operand.to(golden_dtype)).to(output_dtype)
        cfg = dict(block_nnz=32, blocks_per_program=blocks, local_reduce=reduce)
        from flagsparse.sparse_operations._common import _backend_name
        if 'coo_segmented_stream_atomic' not in module.list_spmv_coo_algorithms(
            op=op, dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = module.prepare_spmv_coo(data.to(device), rows.to(device), cols.to(device), shape,
            op=op, alg="coo_segmented_stream_atomic", config=cfg)
        actual, meta = module.flagsparse_spmv_coo_run(prepared, operand.to(device), out_dtype=output_dtype, return_meta=True, timing=True)
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        assert meta["operator_ms"] == meta["gpu_ms"]
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_coo
@pytest.mark.parametrize("alg,op", _MIGRATED_ROUTES)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5), (3, 5)])
def test_spmv_coo_migrated_empty(alg, op, shape):
    import importlib

    module = importlib.import_module("flagsparse.sparse_operations.spmv_coo")
    device = accelerator_device()
    data = torch.empty(0, dtype=torch.float32, device=device)
    first = torch.empty(0, dtype=torch.int64, device=device)
    second = first
    input_rows, output_rows = shape[1], shape[0]
    if op != "non":
        input_rows, output_rows = output_rows, input_rows
    operand = torch.ones((input_rows,), device=device)
    out = torch.full((output_rows,), float("nan"), device=device)
    try:
        prepared = module.prepare_spmv_coo(data, first, second, shape, op=op, alg=alg)
        result = module.flagsparse_spmv_coo_run(prepared, operand, out=out)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert result is out
    assert torch.equal(result.cpu(), torch.zeros_like(result.cpu()))


from tests.pytest.conftest import QUICK_MODE as _MIXED_QUICK_MODE


@pytest.mark.parametrize("index_pair", [(torch.int32, torch.int64)] if _MIXED_QUICK_MODE else [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64),
], ids=lambda pair: "/".join(str(dtype).removeprefix("torch.") for dtype in pair))
@pytest.mark.spmv_coo
@pytest.mark.parametrize("alg,op", [
    ('coo_segmented_atomic', 'non'),
    ('coo_segmented_atomic', 'trans'),
    ('coo_segmented_atomic', 'conj'),
    ('coo_rowrun_subgroup', 'non'),
    ('coo_rowrun_subgroup', 'trans'),
    ('coo_rowrun_subgroup', 'conj'),
    ('coo_segmented_stream_atomic', 'non'),
    ('coo_segmented_stream_atomic', 'trans'),
    ('coo_segmented_stream_atomic', 'conj'),
    ('coo_rowrun', 'non'),
    ('coo_rowrun', 'trans'),
    ('coo_rowrun', 'conj'),
    ('coo_atomic', 'non'),
    ('coo_atomic', 'trans'),
    ('coo_atomic', 'conj'),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True], ids=["same_out", "f32_out"])
def test_spmv_coo_registered_half_combinations(alg, op, dtype, wide, index_pair):
    """Exercise the selected kernel, including a result not exactly representable in half."""
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    try:
        import importlib

        module = importlib.import_module("flagsparse.sparse_operations.spmv_coo")
        device = accelerator_device()
        values = torch.tensor([1, 0.00390625, -0.125, 0.5], dtype=dtype, device=device)
        first = torch.tensor([0, 0, 0, 2], dtype=index_pair[0], device=device)
        second = torch.tensor([0, 0, 1, 2], dtype=index_pair[1], device=device)
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
        if alg not in module.list_spmv_coo_algorithms(
            op=op, dtype=dtype, rhs_dtype=dtype, out_dtype=output_dtype, backend=_backend_name()
        ):
            pytest.skip("algorithm does not declare this type/op/backend combination")
        prepared = module.prepare_spmv_coo(values, first, second, (3, 4), op=op, alg=alg)
        result = module.flagsparse_spmv_coo_run(
            prepared, operand, out=out, out_dtype=output_dtype, **extra
        )
        assert result is out
        rtol, atol = close_tolerances(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
        direct = module.flagsparse_spmv_coo(
            values, first, second, operand, (3, 4), alg=alg, op=op,
            out_dtype=output_dtype, **extra,
        )
        torch.testing.assert_close(direct, result, rtol=rtol, atol=atol)
        for actual, original in zip((values, first, second, operand), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        conflicting = dtype if wide else torch.float32
        with pytest.raises(ValueError, match="conflicts"):
            module.flagsparse_spmv_coo_run(
                prepared, operand, out=out, out_dtype=conflicting, **extra
            )
        with pytest.raises(ValueError, match="overlap"):
            module.flagsparse_spmv_coo_run(prepared, operand, out=values)
    except BackendCapabilityUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.spmv_coo
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("sort_by_row", [False, True])
def test_spmv_coo_legacy_prepared_wide_output(dtype, op, sort_by_row):
    device = accelerator_device()
    values = torch.tensor([1, 0.00390625], dtype=dtype, device=device)
    row = torch.tensor([0, 0], dtype=torch.int32, device=device)
    col = torch.tensor([0, 0], dtype=torch.int64, device=device)
    prepared = prepare_spmv_coo(values, row, col, (2, 3), sort_by_row=sort_by_row, op=op)
    vector = torch.full((3 if op == "non" else 2,), 1.0078125, dtype=dtype, device=device)
    expected = torch.zeros(2 if op == "non" else 3, dtype=torch.float32)
    expected[0] = (values.cpu().double().sum() * vector[0].cpu().double()).float()
    out = torch.full(expected.shape, float("nan"), device=device)
    result = spmv_coo_mod.flagsparse_spmv_coo_run(prepared, vector, out=out, out_dtype=torch.float32)
    assert result is out
    rtol, atol = close_tolerances(torch.float32)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
