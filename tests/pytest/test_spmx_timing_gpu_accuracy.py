"""GPU correctness regressions for live trans/conj inputs and timed mixed routes."""
import importlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from flagsparse.sparse_operations import _common as common

pytestmark = pytest.mark.skipif(not common._ACCEL.is_available(), reason="accelerator required")


def inputs(fmt, dtype):
    values = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    if dtype.is_complex:
        values = torch.complex(values, values / 4)
    values = values.to(dtype)
    device = torch.device(common._ACCEL_DEVICE_TYPE)
    if fmt == "csr":
        arrays = (values.to(device), torch.tensor([1, 0, 4, 3], dtype=torch.int32, device=device),
                  torch.tensor([0, 1, 3, 4], dtype=torch.int32, device=device))
    elif fmt == "coo":
        arrays = (values.to(device), torch.tensor([0, 1, 1, 2], dtype=torch.int32, device=device),
                  torch.tensor([1, 0, 4, 3], dtype=torch.int32, device=device))
    else:
        arrays = (values[[1, 0, 3, 2]].to(device), torch.tensor([1, 0, 2, 1], dtype=torch.int32, device=device),
                  torch.tensor([0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device))
    dense = torch.zeros((3, 5), dtype=dtype)
    dense[0, 1], dense[1, 0], dense[1, 4], dense[2, 3] = values
    return arrays, dense, device


def operand(shape, dtype):
    base = torch.arange(1, 1 + shape[0]*shape[1], dtype=torch.float32).reshape(shape) / 8
    return (torch.complex(base, base / 4) if dtype.is_complex else base).to(dtype)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("op_b", ["non", "conj"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_spmm_timed_operations_refresh_live_values(fmt, op, op_b, dtype):
    module = importlib.import_module("flagsparse.sparse_operations.spmm_" + fmt)
    arrays, dense, device = inputs(fmt, dtype)
    prepare = getattr(module, "prepare_spmm_" + fmt + "_route")
    run = getattr(module, "flagsparse_spmm_" + fmt + "_run")
    prepared = prepare(*arrays, (3, 5), op=op, alg="auto")
    inner = 5 if op == "non" else 3
    raw_rhs = operand((inner, 4) if op_b == "non" else (4, inner), dtype)
    rhs = raw_rhs.to(device)
    output_dtype = dtype if dtype.is_complex else torch.float32
    reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
    for diagnostics in (False, True):
        arrays[0].mul_(0.5)
        dense.mul_(0.5)
        # Ordinary non-op complex COO canonical preparation is an explicit
        # numerical snapshot; callers rebuild it when they update A values.
        if fmt == "coo" and op == "non" and dtype.is_complex:
            prepared = prepare(*arrays, (3, 5), op=op, alg="auto")
        result, elapsed, meta = run(prepared, rhs, op_b=op_b, out_dtype=output_dtype,
                                    return_time=True, return_meta=True, timing=diagnostics)
        a = dense.to(reference_dtype)
        if op != "non": a = a.T
        if op == "conj": a = a.conj()
        b = raw_rhs.to(reference_dtype)
        if op_b != "non": b = b.T.conj()
        expected = (a @ b).to(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.02)
        assert elapsed == pytest.approx(meta["gpu_ms"] + meta.get("process_cpu_ms", 0.0))
        assert meta.get("op_total_ms", elapsed) == pytest.approx(elapsed)
        assert result.dtype == output_dtype


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_spmv_timed_operations_refresh_live_values(fmt, op, dtype):
    module = importlib.import_module("flagsparse.sparse_operations.spmv_" + fmt)
    arrays, dense, device = inputs(fmt, dtype)
    prepare = getattr(module, "prepare_spmv_" + fmt)
    run = getattr(module, "flagsparse_spmv_" + fmt + "_run")
    prepared = prepare(*arrays, (3, 5), op=op, alg="auto")
    raw_rhs = operand((5 if op == "non" else 3, 1), dtype)[:, 0]
    rhs = raw_rhs.to(device)
    output_dtype = dtype if dtype.is_complex else torch.float32
    reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
    for diagnostics in (False, True):
        arrays[0].mul_(0.5)
        dense.mul_(0.5)
        result, elapsed, meta = run(prepared, rhs, out_dtype=output_dtype,
                                    return_time=True, return_meta=True, timing=diagnostics)
        a = dense.to(reference_dtype)
        if op != "non": a = a.T
        if op == "conj": a = a.conj()
        expected = (a @ raw_rhs.to(reference_dtype)).to(output_dtype)
        torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.02)
        assert elapsed == pytest.approx(meta["gpu_ms"] + meta.get("process_cpu_ms", 0.0))
        assert meta.get("op_total_ms", elapsed) == pytest.approx(elapsed)
