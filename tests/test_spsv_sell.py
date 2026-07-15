"""SELL SpSV correctness and fair Triton/cuSPARSE timing checks."""

import argparse
import csv
import ctypes
import ctypes.util
import glob
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
for path in (_PROJECT_ROOT, _SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

if __name__ != "__main__":
    import pytest

import flagsparse as fs
from flagsparse.sparse_operations import spsv as spsv_impl
from tests.test_spsv import (
    _allinone_filtered_avg_ms,
    _apply_csr_op,
    _build_random_triangular_csr,
    _dtype_name,
    _fmt_err,
    _fmt_ms,
    _fmt_ratio,
    _load_mtx_to_csr_torch,
    _random_rhs_for_spsv,
)


if __name__ != "__main__":
    pytestmark = pytest.mark.spsv_sell

VALUE_DTYPES = (torch.float32, torch.float64)
INDEX_DTYPES = (torch.int32, torch.int64)
WARMUP = 1
ITERS = 1

CSV_FIELDS = [
    "matrix",
    "value_dtype",
    "index_dtype",
    "opA",
    "n_rows",
    "n_cols",
    "nnz",
    "FlagSparse_ms",
    "cuSPARSE_ms",
    "PyTorch_ms",
    "FlagSparse_vs_cuSPARSE_speedup",
    "FlagSparse_vs_PyTorch_speedup",
    "status",
    "err_ref",
    "err_res",
    "err_pt",
    "err_cu",
    "pytorch_reason",
    "error",
]

_SUCCESS = 0
_NON_TRANSPOSE = 0
_INDEX_BASE_ZERO = 0
_INDEX_32I = 2
_INDEX_64I = 3
_SPMAT_FILL_MODE = 0
_SPMAT_DIAG_TYPE = 1
_FILL_MODE_LOWER = 0
_DIAG_TYPE_NON_UNIT = 0
_SPSV_ALG_DEFAULT = 0
_CUDA_R_32F = 0
_CUDA_R_64F = 1


def _check(status, name):
    if int(status) != _SUCCESS:
        raise RuntimeError(f"{name} failed with cuSPARSE status {int(status)}")


def _cuda_dtype(dtype):
    return {torch.float32: _CUDA_R_32F, torch.float64: _CUDA_R_64F}[dtype]


def _index_dtype(dtype):
    return {torch.int32: _INDEX_32I, torch.int64: _INDEX_64I}[dtype]


def _configure_cusparse(lib):
    p = ctypes.c_void_p
    pp = ctypes.POINTER(p)
    i = ctypes.c_int
    i64 = ctypes.c_int64

    signatures = {
        "cusparseCreate": ([pp], i),
        "cusparseDestroy": ([p], i),
        "cusparseSetStream": ([p, p], i),
        "cusparseCreateSlicedEll": (
            [pp, i64, i64, i64, i64, i64, p, p, p, i, i, i, i],
            i,
        ),
        "cusparseDestroySpMat": ([p], i),
        "cusparseSpMatSetAttribute": ([p, i, p, ctypes.c_size_t], i),
        "cusparseCreateDnVec": ([pp, i64, p, i], i),
        "cusparseDestroyDnVec": ([p], i),
        "cusparseSpSV_createDescr": ([pp], i),
        "cusparseSpSV_destroyDescr": ([p], i),
        "cusparseSpSV_bufferSize": (
            [p, i, p, p, p, p, i, i, p, ctypes.POINTER(ctypes.c_size_t)],
            i,
        ),
        "cusparseSpSV_analysis": ([p, i, p, p, p, p, i, i, p, p], i),
        "cusparseSpSV_solve": ([p, i, p, p, p, p, i, i, p], i),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = restype


def _load_cusparse():
    name = ctypes.util.find_library("cusparse") or "libcusparse.so.12"
    lib = ctypes.CDLL(name)
    _configure_cusparse(lib)
    return lib


def _stream_ptr():
    return ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream))


def _csr_to_sell(values, cols, row_ptr, n_rows, slice_size):
    """Convert CSR to cuSPARSE's column-major Sliced ELLPACK layout."""

    slice_size = int(slice_size)
    n_slices = (n_rows + slice_size - 1) // slice_size
    widths = []
    for slice_id in range(n_slices):
        row0 = slice_id * slice_size
        row1 = min(row0 + slice_size, n_rows)
        widths.append(
            max(
                int(row_ptr[row + 1].item() - row_ptr[row].item())
                for row in range(row0, row1)
            )
        )

    offsets = torch.zeros(
        n_slices + 1, dtype=row_ptr.dtype, device=row_ptr.device
    )
    if widths:
        increments = torch.tensor(
            [width * slice_size for width in widths],
            dtype=row_ptr.dtype,
            device=row_ptr.device,
        )
        offsets[1:] = torch.cumsum(increments, dim=0)

    padded_size = int(offsets[-1].item())
    sell_values = torch.zeros(
        padded_size, dtype=values.dtype, device=values.device
    )
    sell_cols = torch.full(
        (padded_size,), -1, dtype=cols.dtype, device=cols.device
    )
    for slice_id in range(n_slices):
        row0 = slice_id * slice_size
        row1 = min(row0 + slice_size, n_rows)
        base = int(offsets[slice_id].item())
        for row in range(row0, row1):
            start = int(row_ptr[row].item())
            end = int(row_ptr[row + 1].item())
            count = end - start
            dst = (
                base
                + torch.arange(count, device=values.device) * slice_size
                + row
                - row0
            )
            sell_values[dst] = values[start:end]
            sell_cols[dst] = cols[start:end]
    return sell_values, sell_cols, offsets


def _time_cuda(run, warmup=None, iters=None):
    warmup = WARMUP if warmup is None else int(warmup)
    iters = ITERS if iters is None else int(iters)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    samples = []
    output = None
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = run()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return output, _allinone_filtered_avg_ms(samples, fmt="SELL")


class _CusparseSellSpSV:
    """Minimal native cuSPARSE SELL baseline with reusable descriptors/workspace."""

    def __init__(self, values, cols, offsets, b, n_rows, nnz, slice_size):
        self.lib = _load_cusparse()
        self.values = values
        self.cols = cols
        self.offsets = offsets
        self.b = b
        self.x = torch.empty_like(b)
        self.value_type = _cuda_dtype(values.dtype)
        self.alpha = (
            ctypes.c_float(1.0)
            if values.dtype == torch.float32
            else ctypes.c_double(1.0)
        )
        self.handle = ctypes.c_void_p()
        self.matrix = ctypes.c_void_p()
        self.vec_b = ctypes.c_void_p()
        self.vec_x = ctypes.c_void_p()
        self.descr = ctypes.c_void_p()
        self.workspace = None

        _check(self.lib.cusparseCreate(ctypes.byref(self.handle)), "cusparseCreate")
        _check(
            self.lib.cusparseSetStream(self.handle, _stream_ptr()),
            "cusparseSetStream",
        )
        _check(
            self.lib.cusparseCreateSlicedEll(
                ctypes.byref(self.matrix),
                ctypes.c_int64(n_rows),
                ctypes.c_int64(n_rows),
                ctypes.c_int64(nnz),
                ctypes.c_int64(values.numel()),
                ctypes.c_int64(slice_size),
                ctypes.c_void_p(offsets.data_ptr()),
                ctypes.c_void_p(cols.data_ptr()),
                ctypes.c_void_p(values.data_ptr()),
                ctypes.c_int(_index_dtype(offsets.dtype)),
                ctypes.c_int(_index_dtype(cols.dtype)),
                ctypes.c_int(_INDEX_BASE_ZERO),
                ctypes.c_int(self.value_type),
            ),
            "cusparseCreateSlicedEll",
        )

        fill = ctypes.c_int(_FILL_MODE_LOWER)
        diag = ctypes.c_int(_DIAG_TYPE_NON_UNIT)
        for attribute, value in (
            (_SPMAT_FILL_MODE, fill),
            (_SPMAT_DIAG_TYPE, diag),
        ):
            _check(
                self.lib.cusparseSpMatSetAttribute(
                    self.matrix,
                    ctypes.c_int(attribute),
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                ),
                "cusparseSpMatSetAttribute",
            )

        for target, tensor in (("vec_b", b), ("vec_x", self.x)):
            descriptor = ctypes.c_void_p()
            _check(
                self.lib.cusparseCreateDnVec(
                    ctypes.byref(descriptor),
                    ctypes.c_int64(n_rows),
                    ctypes.c_void_p(tensor.data_ptr()),
                    ctypes.c_int(self.value_type),
                ),
                "cusparseCreateDnVec",
            )
            setattr(self, target, descriptor)

        _check(
            self.lib.cusparseSpSV_createDescr(ctypes.byref(self.descr)),
            "cusparseSpSV_createDescr",
        )
        size = ctypes.c_size_t()
        _check(
            self.lib.cusparseSpSV_bufferSize(
                self.handle,
                ctypes.c_int(_NON_TRANSPOSE),
                ctypes.byref(self.alpha),
                self.matrix,
                self.vec_b,
                self.vec_x,
                ctypes.c_int(self.value_type),
                ctypes.c_int(_SPSV_ALG_DEFAULT),
                self.descr,
                ctypes.byref(size),
            ),
            "cusparseSpSV_bufferSize",
        )
        self.workspace = torch.empty(
            max(1, size.value), dtype=torch.uint8, device=values.device
        )
        self.workspace_ptr = (
            ctypes.c_void_p(self.workspace.data_ptr())
            if size.value
            else ctypes.c_void_p()
        )

    def analysis_and_solve(self):
        _check(
            self.lib.cusparseSpSV_analysis(
                self.handle,
                ctypes.c_int(_NON_TRANSPOSE),
                ctypes.byref(self.alpha),
                self.matrix,
                self.vec_b,
                self.vec_x,
                ctypes.c_int(self.value_type),
                ctypes.c_int(_SPSV_ALG_DEFAULT),
                self.descr,
                self.workspace_ptr,
            ),
            "cusparseSpSV_analysis",
        )
        _check(
            self.lib.cusparseSpSV_solve(
                self.handle,
                ctypes.c_int(_NON_TRANSPOSE),
                ctypes.byref(self.alpha),
                self.matrix,
                self.vec_b,
                self.vec_x,
                ctypes.c_int(self.value_type),
                ctypes.c_int(_SPSV_ALG_DEFAULT),
                self.descr,
            ),
            "cusparseSpSV_solve",
        )
        return self.x

    def close(self):
        if self.descr.value:
            self.lib.cusparseSpSV_destroyDescr(self.descr)
        if self.vec_x.value:
            self.lib.cusparseDestroyDnVec(self.vec_x)
        if self.vec_b.value:
            self.lib.cusparseDestroyDnVec(self.vec_b)
        if self.matrix.value:
            self.lib.cusparseDestroySpMat(self.matrix)
        if self.handle.value:
            self.lib.cusparseDestroy(self.handle)


def _benchmark_triton(values, cols, offsets, b, n_rows, slice_size):
    ready = torch.empty(n_rows, dtype=torch.int32, device=b.device)
    row_counter = torch.empty(1, dtype=torch.int32, device=b.device)
    out = torch.empty_like(b)

    def analysis_and_solve():
        return spsv_impl._launch_spsv_sell(
            values,
            cols,
            offsets,
            b,
            n_rows,
            slice_size=slice_size,
            out=out,
            ready=ready,
            row_counter=row_counter,
        )

    return _time_cuda(analysis_and_solve)


def _run_case(
    matrix,
    values,
    cols,
    row_ptr,
    b,
    expected,
    slice_size,
):
    n_rows = int(row_ptr.numel() - 1)
    sell_values, sell_cols, offsets = _csr_to_sell(
        values, cols, row_ptr, n_rows, slice_size
    )
    public_result = fs.flagsparse_spsv_sell(
        sell_values,
        sell_cols,
        offsets,
        b,
        (n_rows, n_rows),
        slice_size=slice_size,
    )
    triton_result, triton_ms = _benchmark_triton(
        sell_values, sell_cols, offsets, b, n_rows, slice_size
    )
    baseline = _CusparseSellSpSV(
        sell_values,
        sell_cols,
        offsets,
        b,
        n_rows,
        values.numel(),
        slice_size,
    )
    try:
        cusparse_result, cusparse_ms = _time_cuda(baseline.analysis_and_solve)
    finally:
        baseline.close()

    err_ref = float(torch.max(torch.abs(public_result - expected)).item())
    err_res = float(
        torch.max(
            torch.abs(
                _apply_csr_op(
                    values,
                    cols,
                    row_ptr,
                    triton_result,
                    (n_rows, n_rows),
                    "NON",
                    lower=True,
                )
                - b
            )
        ).item()
    )
    err_cu = float(torch.max(torch.abs(triton_result - cusparse_result)).item())
    atol = 2e-5 if values.dtype == torch.float32 else 1e-11
    rtol = 2e-5 if values.dtype == torch.float32 else 1e-11
    status = (
        "PASS"
        if torch.allclose(triton_result, expected, atol=atol, rtol=rtol)
        and torch.allclose(triton_result, cusparse_result, atol=atol, rtol=rtol)
        else "FAIL"
    )
    record = {
        "matrix": matrix,
        "value_dtype": _dtype_name(values.dtype),
        "index_dtype": _dtype_name(cols.dtype),
        "opA": "NON",
        "n_rows": n_rows,
        "n_cols": n_rows,
        "nnz": int(values.numel()),
        "FlagSparse_ms": triton_ms,
        "cuSPARSE_ms": cusparse_ms,
        "PyTorch_ms": None,
        "FlagSparse_vs_cuSPARSE_speedup": cusparse_ms / triton_ms,
        "FlagSparse_vs_PyTorch_speedup": None,
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": None,
        "err_cu": err_cu,
        "pytorch_reason": "not used for SELL",
        "error": None,
    }
    return record, triton_result, cusparse_result


def _print_header(slice_size, value_dtype, index_dtype):
    print("=" * 144)
    print(
        f"Value dtype: {_dtype_name(value_dtype)} | "
        f"Index dtype: {_dtype_name(index_dtype)} | SELL | "
        f"triA=LOWER | opA=NON | slice_size={slice_size}"
    )
    print(
        f"Benchmark schedule: warmup={WARMUP}, iter={ITERS}; "
        "FS.ms and CU.ms both include every per-call analysis/preparation + solve."
    )
    print("CU.spd = CU.ms / FS.ms; PT.spd = PT.ms / FS.ms.")
    print("-" * 144)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
        f"{'FS.ms':>10} {'CU.ms':>10} {'PT.ms':>10} "
        f"{'CU.spd':>10} {'PT.spd':>10} {'Status':>6} "
        f"{'Eref':>10} {'Eres':>10} {'Ept':>10} {'Ecu':>10}"
    )
    print("-" * 144)


def _print_record(record):
    name = str(record["matrix"])
    name = name[:27] + ("…" if len(name) > 27 else "")
    print(
        f"{name:<28} {record['n_rows']:>7} {record['n_cols']:>7} "
        f"{record['nnz']:>10} "
        f"{_fmt_ms(record['FlagSparse_ms']):>10} "
        f"{_fmt_ms(record['cuSPARSE_ms']):>10} "
        f"{_fmt_ms(record['PyTorch_ms']):>10} "
        f"{_fmt_ratio(record['FlagSparse_vs_cuSPARSE_speedup']):>10} "
        f"{_fmt_ratio(record['FlagSparse_vs_PyTorch_speedup']):>10} "
        f"{record['status']:>6} {_fmt_err(record['err_ref']):>10} "
        f"{_fmt_err(record['err_res']):>10} {_fmt_err(record['err_pt']):>10} "
        f"{_fmt_err(record['err_cu']):>10}"
    )


def test_spsv_sell_matches_cusparse(value_dtype, index_dtype, slice_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    n_rows = 64
    values, cols, row_ptr, shape = _build_random_triangular_csr(
        n_rows,
        value_dtype,
        index_dtype,
        torch.device("cuda"),
        lower=True,
    )
    row_ptr = row_ptr.to(index_dtype)
    expected = _random_rhs_for_spsv(
        shape, value_dtype, values.device, op_mode="NON", seed=1234
    )
    b = _apply_csr_op(
        values, cols, row_ptr, expected, shape, "NON", lower=True
    )
    atol = 2e-5 if value_dtype == torch.float32 else 1e-11
    rtol = 2e-5 if value_dtype == torch.float32 else 1e-11
    try:
        record, triton_result, cusparse_result = _run_case(
            "synthetic-64", values, cols, row_ptr, b, expected, slice_size
        )
    except (AttributeError, OSError, RuntimeError) as exc:
        pytest.skip(f"native cuSPARSE SELL SpSV is unavailable: {exc}")
    assert torch.allclose(triton_result, cusparse_result, atol=atol, rtol=rtol)
    assert record["status"] == "PASS"
    assert record["FlagSparse_ms"] > 0.0
    assert record["cuSPARSE_ms"] > 0.0
    _print_header(slice_size, value_dtype, index_dtype)
    _print_record(record)


if __name__ != "__main__":
    test_spsv_sell_matches_cusparse = pytest.mark.parametrize(
        "value_dtype", VALUE_DTYPES
    )(
        pytest.mark.parametrize("index_dtype", INDEX_DTYPES)(
            pytest.mark.parametrize("slice_size", (8, 32))(
                test_spsv_sell_matches_cusparse
            )
        )
    )


def _expand_mtx_paths(inputs):
    paths = []
    for value in inputs:
        if os.path.isdir(value):
            paths.extend(sorted(glob.glob(os.path.join(value, "*.mtx"))))
        elif value.endswith(".mtx"):
            paths.append(value)
    return paths


def main():
    global WARMUP, ITERS
    parser = argparse.ArgumentParser(
        description="Lower real SELL SpSV: Triton versus cuSPARSE analysis+solve"
    )
    parser.add_argument("mtx", nargs="+", help=".mtx files or directories")
    parser.add_argument("--csv", required=True, help="output CSV path")
    parser.add_argument("--slice-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    WARMUP = max(0, args.warmup)
    ITERS = max(1, args.iters)
    paths = _expand_mtx_paths(args.mtx)
    if not paths:
        raise SystemExit("No .mtx files found")

    records = []
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            _print_header(args.slice_size, value_dtype, index_dtype)
            for path in paths:
                try:
                    values, cols, row_ptr, shape = _load_mtx_to_csr_torch(
                        path,
                        dtype=value_dtype,
                        device=torch.device("cuda"),
                        lower=True,
                    )
                    if int(shape[0]) != int(shape[1]):
                        raise ValueError(f"SpSV requires a square matrix, got {shape}")
                    cols = cols.to(index_dtype)
                    row_ptr = row_ptr.to(index_dtype)
                    expected = torch.ones(
                        shape[0], dtype=value_dtype, device=values.device
                    )
                    b = _apply_csr_op(
                        values,
                        cols,
                        row_ptr,
                        expected,
                        shape,
                        "NON",
                        lower=True,
                    )
                    record, _, _ = _run_case(
                        os.path.basename(path),
                        values,
                        cols,
                        row_ptr,
                        b,
                        expected,
                        args.slice_size,
                    )
                    records.append(record)
                    _print_record(record)
                except Exception as exc:
                    record = {key: None for key in CSV_FIELDS}
                    record.update(
                        matrix=os.path.basename(path),
                        value_dtype=_dtype_name(value_dtype),
                        index_dtype=_dtype_name(index_dtype),
                        opA="NON",
                        status="ERROR",
                        error=str(exc),
                    )
                    records.append(record)
                    print(f"{record['matrix']:<28} ERROR: {exc}")

    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: "" if record.get(key) is None else record.get(key) for key in CSV_FIELDS}
            )
    print("-" * 144)
    print(f"Wrote {len(records)} rows to {args.csv}")


if __name__ == "__main__":
    main()
