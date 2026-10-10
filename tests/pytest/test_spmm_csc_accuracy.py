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
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]


def _random_values(shape, dtype, device):
    if dtype in (torch.float16, torch.float32, torch.float64):
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
    if dtype in (torch.float16, torch.float32):
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
    assert meta["process_gpu_ms"] == 0.0
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
            raise RuntimeError("forced int64 launch failure")
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


from tests.pytest.conftest import QUICK_MODE


@pytest.mark.spmm_csc
@pytest.mark.parametrize("alg,op", [("csc_col_tile_panel_atomic", "non"), ("csc_col_panel", "trans"), ("csc_col_panel", "conj")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("index_dtype,ptr_dtype", [(torch.int32, torch.int64)] if QUICK_MODE else
                         [(i, p) for i in (torch.int32, torch.int64) for p in (torch.int32, torch.int64)])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_csc_registered_columns(alg, op, dtype, index_dtype, ptr_dtype, layout):
    from flagsparse.sparse_operations._spmm_csr_config import resolve_csc_config
    from flagsparse.sparse_operations._spmm_csr_runtime import backend_caps
    from flagsparse import prepare_spmm_csc_route, flagsparse_spmm_csc_run
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
    dense_cpu = _random_values((dense_rows, 7), dtype, golden_device())
    dense = dense_cpu.to(device)
    if layout == "col":
        dense = dense.T.contiguous().T
    effective = matrix if op == "non" else matrix.T if op == "trans" else matrix.conj().T
    expected = effective @ dense_cpu.to(matrix.dtype)
    prepared = prepare_spmm_csc_route(data, indices, indptr, (19, len(lengths)), op=op, alg=alg)
    before = data.clone()
    output = torch.full(expected.shape, float("nan"), dtype=dtype, device=device)
    result, meta = flagsparse_spmm_csc_run(prepared, dense, out=output, timing=True, return_meta=True)
    assert result is output
    _assert_close(result, expected, dtype)
    assert torch.equal(data, before)
    assert meta["op_total_ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert meta["alg_resolved"] == alg
    assert not hasattr(prepared, "col_ids")
    with pytest.raises(ValueError, match="op"):
        flagsparse_spmm_csc_run(prepared, dense, op="trans" if op == "non" else "non")
    with pytest.raises(ValueError):
        flagsparse_spmm_csc_run(prepared, dense, config={"num_warps": 3})


@pytest.mark.spmm_csc
@pytest.mark.parametrize("alg,op", [("csc_col_panel", "trans"), ("csc_col_tile_panel_atomic", "non")])
@pytest.mark.parametrize("n", [1, 7, 8, 16, 31, 32, 33, 64, 65, 128] + ([] if QUICK_MODE else [256]))
def test_spmm_csc_panel_tail_and_empty_columns(alg, op, n):
    from flagsparse import flagsparse_spmm_csc_run
    from flagsparse.sparse_operations._spmm_csr_runtime import backend_caps
    from flagsparse.sparse_operations._spmm_csr_config import resolve_csc_config
    device = accelerator_device()
    try:
        resolve_csc_config(alg, "float32", n, backend_caps(device), op=op)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    matrix = torch.tensor([[0., 2., 0.], [0., -1., 0.]])
    a, i, p = _dense_to_csc(matrix, torch.int32)
    dense = torch.ones((3 if op == "non" else 2, n), device=device)
    prepared = prepare_spmm_csc_route(a.to(device), i.to(device), p.to(device), (2, 3), alg=alg, op=op)
    expected = (matrix if op == "non" else matrix.T) @ dense.cpu()
    out = torch.full(expected.shape, float("nan"), device=device)
    actual = flagsparse_spmm_csc_run(prepared, dense, out=out)
    assert actual is out
    _assert_close(actual, expected, torch.float32)
    zero = flagsparse_spmm_csc_run(prepared, dense[:, :0])
    assert zero.shape == (expected.shape[0], 0)


@pytest.mark.spmm_csc
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (7, 0), (7, 5)])
def test_spmm_csc_registered_empty(op, shape):
    from flagsparse import prepare_spmm_csc_route, flagsparse_spmm_csc_run
    from flagsparse.sparse_operations._spmm_csr_runtime import backend_caps
    from flagsparse.sparse_operations._spmm_csr_config import resolve_csc_config
    alg = 'csc_col_tile_panel_atomic' if op == "non" else 'csc_col_panel'
    device = accelerator_device()
    try:
        resolve_csc_config(alg, "float32", 7, backend_caps(device), op=op)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    a = torch.empty(0, device=device)
    i = torch.empty(0, dtype=torch.int64, device=device)
    p = torch.zeros(shape[1] + 1, dtype=torch.int32, device=device)
    prepared = prepare_spmm_csc_route(a, i, p, shape, alg=alg, op=op)
    length = shape[1] if op == "non" else shape[0]
    dense = torch.empty((length, 7), device=device)
    result = flagsparse_spmm_csc_run(prepared, dense)
    assert result.shape[0] == (shape[0] if op == "non" else shape[1])
    assert torch.count_nonzero(result).item() == 0
