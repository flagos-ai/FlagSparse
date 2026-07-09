"""Native BSR SpMV benchmark and correctness script."""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None


VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = (torch.int32, torch.int64)
OPS = ("non", "trans", "conj")
SUPPORTED_OPS = ("non",)
TEST_SIZES = ((64, 96), (160, 1024), (128, 256))
DEFAULT_BLOCK_DIMS = (4,)
WARMUP = 10
ITERS = 50


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {"int32": torch.int32, "int64": torch.int64}


def _parse_csv_tokens(value, mapping, option_name):
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [token for token in tokens if token not in mapping]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: {', '.join(mapping)}"
        )
    return [mapping[token] for token in tokens]


def _parse_ops(value):
    token = "non" if value is None else str(value).strip().lower()
    if token == "all":
        return list(OPS)
    ops = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [op for op in ops if op not in OPS]
    if not ops or invalid:
        raise ValueError(f"unsupported --ops: {', '.join(invalid or ops)}")
    return ops


def _parse_block_dims(value):
    token = str(value or "4").strip().lower()
    if token == "auto":
        return ["auto"]
    dims = []
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim <= 1:
            raise ValueError("--block-dims values must be greater than 1")
        dims.append(dim)
    if not dims:
        raise ValueError("--block-dims must not be empty")
    return dims


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


def _reference_tolerance(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    return 1e-6, 1e-5


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        return complex(raw_value)
    return float(raw_value.real if isinstance(raw_value, complex) else raw_value)


def _zero_value(dtype):
    return 0j if dtype in (torch.complex64, torch.complex128) else 0.0


def _choose_auto_block_dim(entries, shape):
    n_rows, n_cols = shape
    nnz = max(1, len(entries))
    for block_dim in (16, 8, 4, 2):
        blocks = {
            (int(row) // block_dim, int(col) // block_dim)
            for row, col in entries.keys()
        }
        stored = len(blocks) * block_dim * block_dim
        if stored <= 2.0 * nnz:
            return block_dim
    return 4 if max(n_rows, n_cols) >= 4 else 2


def _entries_to_bsr_torch(entries, shape, dtype, index_dtype, block_dim, device):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    block_dim = int(block_dim)
    blocks = {}
    for (row, col), value in entries.items():
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            [_zero_value(dtype) for _ in range(block_dim * block_dim)],
        )
        block[inner_row * block_dim + inner_col] += _mtx_value_for_dtype(value, dtype)
    n_block_rows = (n_rows + block_dim - 1) // block_dim
    rows = [[] for _ in range(n_block_rows)]
    for key in sorted(blocks):
        rows[key[0]].append(key)
    data_values = []
    indices_values = []
    indptr_values = [0]
    for row_blocks in rows:
        for key in row_blocks:
            indices_values.append(key[1])
            data_values.extend(blocks[key])
        indptr_values.append(len(indices_values))
    data = torch.tensor(data_values, dtype=dtype, device=device)
    data = data.reshape(-1, block_dim, block_dim).contiguous()
    indices = torch.tensor(indices_values, dtype=index_dtype, device=device)
    indptr = torch.tensor(indptr_values, dtype=index_dtype, device=device)
    return data, indices.contiguous(), indptr.contiguous()


def _dense_to_bsr(dense, index_dtype, block_dim):
    rows, cols = dense.nonzero(as_tuple=True)
    entries = {
        (int(row.item()), int(col.item())): dense[row, col].item()
        for row, col in zip(rows, cols)
    }
    return _entries_to_bsr_torch(
        entries,
        tuple(dense.shape),
        dense.dtype,
        index_dtype,
        block_dim,
        dense.device,
    )


def _bsr_block_rows(indptr):
    counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(indptr.numel() - 1, dtype=torch.int64, device=indptr.device),
        counts,
    )


def _bsr_to_torch_coo(data, indices, indptr, shape, block_dim):
    block_rows = _bsr_block_rows(indptr)
    nnzb = int(data.shape[0])
    if nnzb == 0:
        empty = torch.empty(0, dtype=torch.int64, device=data.device)
        return torch.sparse_coo_tensor(
            torch.stack([empty, empty]),
            data.reshape(-1),
            size=shape,
            device=data.device,
            dtype=data.dtype,
        ).coalesce()
    local = torch.arange(block_dim * block_dim, dtype=torch.int64, device=data.device)
    inner_rows = local // block_dim
    inner_cols = local % block_dim
    rows = block_rows[:, None] * block_dim + inner_rows[None, :]
    cols = indices.to(torch.int64)[:, None] * block_dim + inner_cols[None, :]
    values = data.reshape(nnzb, block_dim * block_dim)
    mask = (rows < int(shape[0])) & (cols < int(shape[1])) & (values != 0)
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        size=shape,
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _pytorch_reference(data, indices, indptr, x, shape, dtype, block_dim):
    ref_dtype = _reference_dtype(dtype)
    A = _bsr_to_torch_coo(
        data.to(ref_dtype),
        indices,
        indptr,
        shape,
        block_dim,
    )
    return torch.sparse.mm(A, x.to(ref_dtype).unsqueeze(1)).squeeze(1).to(dtype)


def _allclose_error_ratio(actual, expected, atol, rtol):
    if expected.numel() == 0:
        return 0.0
    diff = torch.abs(actual - expected).to(torch.float64)
    denom = atol + rtol * torch.abs(expected).to(torch.float64)
    return float(torch.max(diff / denom).item())


def _cuda_event_benchmark(op, warmup, iters):
    out = None
    count = max(1, int(iters))
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / count


def _time_flagsparse_bsr(data, indices, indptr, x, shape, block_dim, warmup, iters, timing=False):
    prepared = fs.prepare_spmv_bsr(data, indices, indptr, shape, block_dim, op="non")
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmv_bsr(x=x, prepared=prepared),
        warmup,
        iters,
    )
    return {
        "out": out,
        "ms": gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": gpu_ms if timing else None,
    }


def _time_pytorch(data, indices, indptr, x, shape, block_dim, warmup, iters):
    A = _bsr_to_torch_coo(data, indices, indptr, shape, block_dim)
    fn = lambda: torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    _, ms = _cuda_event_benchmark(fn, warmup, iters)
    return ms


def _time_cusparse(data, indices, indptr, x, shape, block_dim, warmup, iters):
    if cp is None or cpx_sparse is None:
        return None
    if data.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        return None
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.to(torch.int64)))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    A = cpx_sparse.bsr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    fn = lambda: A @ x_cp
    for _ in range(max(0, int(warmup))):
        _ = fn()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        _ = fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / count


def _fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _spd(base, other):
    if base is None or other is None or other <= 0:
        return "N/A"
    return f"{base / other:.2f}x"


def _status(ok):
    return "PASS" if ok else "FAIL"


def _header(timing=False):
    split = f" {'ProcGPU':>9} {'Compute':>9}" if timing else ""
    return (
        f"{'Matrix':<28} {'Op':>5} {'BDim':>5} {'Out':>7} {'Rows':>7} {'Cols':>7} {'NNZB':>9} {'Pad':>7}  "
        f"{'BSR(ms)':>9} {'BSRGPU':>9} {'CPUProc':>9}{split} "
        f"{'PT(ms)':>9} {'CU(ms)':>9}  {'BSR/PT':>8} {'BSR/CU':>8} "
        f"{'Err':>10} {'Status':>6}"
    )


def _sep(timing=False):
    return "-" * (158 if timing else 138)


def _print_row(row, timing=False):
    name = str(row["matrix"])[:27]
    if len(str(row["matrix"])) > 27:
        name += "..."
    split = (
        f" {_fmt(row.get('process_gpu_ms')):>9} {_fmt(row.get('compute_ms')):>9}"
        if timing
        else ""
    )
    print(
        f"{name:<28} {row['op']:>5} {row['block_dim']:>5} {row['out_size']:>7} {row['n_rows']:>7} {row['n_cols']:>7} {row['nnzb']:>9} {row['padding_ratio']:>7}  "
        f"{_fmt(row['bsr_ms']):>9} {_fmt(row['bsr_gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9}{split} "
        f"{_fmt(row['pytorch_ms']):>9} {_fmt(row['cusparse_ms']):>9}  "
        f"{_spd(row['pytorch_ms'], row['bsr_ms']):>8} {_spd(row['cusparse_ms'], row['bsr_ms']):>8} "
        f"{_fmt_err(row['err']):>10} {row['status']:>6}"
    )
    error = row.get("error")
    if error:
        print(f"  error: {str(error)[:240]}")


def _base_row(
    matrix_name,
    dtype,
    index_dtype,
    op,
    shape,
    data,
    block_dim,
    logical_nnz=None,
    status="ERROR",
):
    nnzb = int(data.shape[0])
    stored_nnz = int(data.numel())
    logical_nnz = max(1, int(logical_nnz if logical_nnz is not None else stored_nnz))
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "block_dim": int(block_dim),
        "out_size": int(shape[0]) if op == "non" else "UNSUP",
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnzb": nnzb,
        "logical_nnz": logical_nnz,
        "stored_nnz": stored_nnz,
        "padding_ratio": f"{stored_nnz / logical_nnz:.2f}",
        "bsr_ms": None,
        "bsr_gpu_ms": None,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": None,
        "compute_ms": None,
        "pytorch_ms": None,
        "cusparse_ms": None,
        "err": None,
        "status": status,
        "error": None,
    }


def _run_one_case(
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    op,
    matrix_name,
    block_dim,
    warmup,
    iters,
    timing=False,
    run_cusparse=True,
    logical_nnz=None,
):
    data = data.contiguous()
    indices = indices.to(index_dtype).contiguous()
    indptr = indptr.to(index_dtype).contiguous()
    row = _base_row(
        matrix_name,
        dtype,
        index_dtype,
        op,
        shape,
        data,
        block_dim,
        logical_nnz=logical_nnz,
    )
    row["process_gpu_ms"] = 0.0 if timing else None
    if op not in SUPPORTED_OPS:
        row["status"] = "SKIP"
        row["error"] = "BSR SpMV v1 only supports op=non"
        return row
    x = _random_values((int(shape[1]),), dtype, data.device)
    atol, rtol = _reference_tolerance(dtype)
    try:
        bsr = _time_flagsparse_bsr(data, indices, indptr, x, shape, block_dim, warmup, iters, timing=timing)
    except Exception as exc:
        row["error"] = f"flagsparse_spmv_bsr failed: {exc}"
        return row
    row.update(
        {
            "bsr_ms": bsr["ms"],
            "bsr_gpu_ms": bsr["gpu_ms"],
            "process_cpu_ms": bsr["process_cpu_ms"],
            "process_gpu_ms": bsr["process_gpu_ms"],
            "compute_ms": bsr["compute_ms"],
        }
    )
    try:
        y_ref = _pytorch_reference(data, indices, indptr, x, shape, dtype, block_dim)
        err = _allclose_error_ratio(bsr["out"], y_ref, atol, rtol)
    except Exception as exc:
        row["error"] = f"reference failed after BSR run: {exc}"
        return row
    try:
        row["pytorch_ms"] = _time_pytorch(data, indices, indptr, x, shape, block_dim, warmup, iters)
    except Exception:
        pass
    if run_cusparse:
        try:
            row["cusparse_ms"] = _time_cusparse(data, indices, indptr, x, shape, block_dim, warmup, iters)
        except Exception:
            pass
    ok = (not math.isnan(err)) and err <= 1.0
    row["err"] = err
    row["status"] = _status(ok)
    row["error"] = None if ok else "correctness check failed"
    return row


def load_mtx_entries(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    mm_field = "real"
    mm_symmetry = "general"
    header = None
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%%MatrixMarket"):
            parts = stripped.split()
            if len(parts) >= 5:
                mm_field = parts[3].lower()
                mm_symmetry = parts[4].lower()
            continue
        if stripped.startswith("%"):
            continue
        if header is None and stripped:
            parts = stripped.split()
            header = (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            continue
        if stripped:
            data_lines.append(stripped)
    if header is None:
        raise ValueError(f"Cannot parse .mtx header: {path}")
    n_rows, n_cols, nnz = header
    entries = {}

    def add_entry(r, c, value):
        key = (int(r), int(c))
        entries[key] = entries.get(key, 0.0) + value

    is_pattern = mm_field == "pattern"
    is_complex = mm_field == "complex"
    is_symmetric = mm_symmetry == "symmetric"
    is_skew = mm_symmetry == "skew-symmetric"
    is_hermitian = mm_symmetry == "hermitian"
    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            continue
        if is_pattern:
            value = 1.0
        elif is_complex:
            value = complex(float(parts[2]), float(parts[3]))
        else:
            value = float(parts[2])
        add_entry(r, c, value)
        if r != c:
            if is_symmetric and 0 <= c < n_rows and 0 <= r < n_cols:
                add_entry(c, r, value)
            elif is_skew and 0 <= c < n_rows and 0 <= r < n_cols:
                add_entry(c, r, -value)
            elif is_hermitian and 0 <= c < n_rows and 0 <= r < n_cols:
                add_entry(c, r, value.conjugate() if isinstance(value, complex) else value)
    return entries, (n_rows, n_cols)


def _resolve_block_dims(block_dims, entries, shape):
    if block_dims == ["auto"]:
        return [_choose_auto_block_dim(entries, shape)]
    return block_dims


def run_synthetic(value_dtypes=None, index_dtypes=None, block_dims=None, ops=None, warmup=WARMUP, iters=ITERS, timing=False, run_cusparse=True):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    block_dims = list(DEFAULT_BLOCK_DIMS) if block_dims is None else block_dims
    ops = SUPPORTED_OPS if ops is None else ops
    print("=" * 140)
    print("FLAGSPARSE SpMV BSR BENCHMARK (native BSR Triton)")
    print("=" * 140)
    print("Timing policy: bsr_ms = process_cpu_ms + bsr_gpu_ms; BSR construction is setup.")
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for block_dim in block_dims:
                for op in ops:
                    print(_sep(timing))
                    print(f"dtype: {_dtype_name(dtype)} | index_dtype: {_dtype_name(index_dtype)} | block_dim: {block_dim} | op: {op}")
                    print(_sep(timing))
                    print(_header(timing))
                    print(_sep(timing))
                    for m, n in TEST_SIZES:
                        dense = _random_values((m, n), dtype, device)
                        dense *= (torch.rand(m, n, device=device) < 0.1).to(dtype=dtype)
                        logical_nnz = int(torch.count_nonzero(dense).item())
                        data, indices, indptr = _dense_to_bsr(dense, index_dtype, int(block_dim))
                        row = _run_one_case(
                            data,
                            indices,
                            indptr,
                            (m, n),
                            dtype,
                            index_dtype,
                            op,
                            f"{m}x{n}",
                            int(block_dim),
                            warmup,
                            iters,
                            timing=timing,
                            run_cusparse=run_cusparse,
                            logical_nnz=logical_nnz,
                        )
                        _print_row(row, timing=timing)
                    print(_sep(timing))
                    print()


def run_csv(mtx_paths, csv_path, value_dtypes=None, index_dtypes=None, block_dims=None, ops=None, warmup=WARMUP, iters=ITERS, timing=False, run_cusparse=True, fail_fast=False):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    block_dims = list(DEFAULT_BLOCK_DIMS) if block_dims is None else block_dims
    ops = SUPPORTED_OPS if ops is None else ops
    rows = []
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print(_sep(timing))
                print(f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: {op}")
                print(_sep(timing))
                print(_header(timing))
                print(_sep(timing))
                for path in mtx_paths:
                    try:
                        entries, shape = load_mtx_entries(path)
                        for block_dim in _resolve_block_dims(block_dims, entries, shape):
                            data, indices, indptr = _entries_to_bsr_torch(
                                entries, shape, dtype, index_dtype, int(block_dim), device
                            )
                            row = _run_one_case(
                                data,
                                indices,
                                indptr,
                                shape,
                                dtype,
                                index_dtype,
                                op,
                                os.path.basename(path),
                                int(block_dim),
                                warmup,
                                iters,
                                timing=timing,
                                run_cusparse=run_cusparse,
                                logical_nnz=len(entries),
                            )
                            if fail_fast and row.get("status") == "ERROR":
                                raise RuntimeError(row.get("error") or "BSR SpMV case failed")
                            rows.append(row)
                            _print_row(row, timing=timing)
                    except Exception as exc:
                        if fail_fast:
                            raise
                        row = {
                            "matrix": os.path.basename(path),
                            "value_dtype": _dtype_name(dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "op": op,
                            "block_dim": "ERR",
                            "out_size": "ERR",
                            "n_rows": "ERR",
                            "n_cols": "ERR",
                            "nnzb": "ERR",
                            "logical_nnz": "ERR",
                            "stored_nnz": "ERR",
                            "padding_ratio": "ERR",
                            "bsr_ms": None,
                            "bsr_gpu_ms": None,
                            "process_cpu_ms": None,
                            "process_gpu_ms": None,
                            "compute_ms": None,
                            "pytorch_ms": None,
                            "cusparse_ms": None,
                            "err": None,
                            "status": "ERROR",
                            "error": str(exc),
                        }
                        rows.append(row)
                        _print_row(row, timing=timing)
                print(_sep(timing))
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "block_dim",
        "out_size",
        "n_rows",
        "n_cols",
        "nnzb",
        "logical_nnz",
        "stored_nnz",
        "padding_ratio",
        "bsr_ms",
        "bsr_gpu_ms",
        "process_cpu_ms",
        "process_gpu_ms",
        "compute_ms",
        "pytorch_ms",
        "cusparse_ms",
        "err",
        "status",
        "error",
    ]
    if not timing:
        fieldnames = [
            field
            for field in fieldnames
            if field not in ("process_gpu_ms", "compute_ms")
        ]
    csv_parent = Path(csv_path).parent
    if str(csv_parent) not in ("", "."):
        csv_parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Native BSR SpMV benchmark/test.")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-bsr", type=str, default=None, metavar="FILE")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--block-dims", default="4")
    parser.add_argument("--ops", default="non")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    try:
        value_dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
        index_dtypes = _parse_csv_tokens(args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes")
        block_dims = _parse_block_dims(args.block_dims)
        ops = _parse_ops(args.ops)
    except ValueError as exc:
        parser.error(str(exc))
    if args.synthetic:
        run_synthetic(
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            block_dims=block_dims,
            ops=ops,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
        )
        return
    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    if args.csv_bsr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-bsr")
            return
        run_csv(
            paths,
            args.csv_bsr,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            block_dims=block_dims,
            ops=ops,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
            fail_fast=args.fail_fast,
        )
        return
    if not paths:
        print("No .mtx files. Use --synthetic or --csv-bsr with inputs.")
        return
    run_csv(
        paths,
        "spmv_bsr_results.csv",
        value_dtypes=value_dtypes,
        index_dtypes=index_dtypes,
        block_dims=block_dims,
        ops=ops,
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
        run_cusparse=not args.no_cusparse,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
