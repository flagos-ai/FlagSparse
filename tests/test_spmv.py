"""
SpMV tests (CSR): load SuiteSparse .mtx, batch run, output error and performance.
Supports: multi .mtx files, value_dtype / index_dtype, --csv-csr to run all dtypes and export CSV.
"""
import argparse
import csv
import glob
import math
import os

import torch
import flagsparse as ast

VALUE_DTYPES = [
    torch.float32,
    torch.float64,
]
INDEX_DTYPES = [torch.int32]
TEST_CASES = [
    (512, 512, 4096),
    (1024, 1024, 16384),
    (2048, 2048, 65536),
    (4096, 4096, 131072),
]
WARMUP = 10
ITERS = 50


def load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    """
    Load SuiteSparse / Matrix Market .mtx file into CSR as torch tensors.
    Returns (data, indices, indptr, shape) on device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = []
    header_info = None
    for line in lines:
        line = line.strip()
        if line.startswith("%"):
            continue
        if not header_info and line:
            parts = line.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if line:
            data_lines.append(line)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info
    if nnz == 0:
        data = torch.tensor([], dtype=dtype, device=device)
        indices = torch.tensor([], dtype=torch.int64, device=device)
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
        return data, indices, indptr, (n_rows, n_cols)
    rows_host = []
    cols_host = []
    vals_host = []
    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) >= 3:
            rows_host.append(int(parts[0]) - 1)
            cols_host.append(int(parts[1]) - 1)
            vals_host.append(float(parts[2]))
    order = sorted(range(len(rows_host)), key=lambda i: (rows_host[i], cols_host[i]))
    rows_s = [rows_host[i] for i in order]
    cols_s = [cols_host[i] for i in order]
    vals_s = [vals_host[i] for i in order]
    indptr_list = [0]
    idx = 0
    for r in range(n_rows):
        while idx < len(rows_s) and rows_s[idx] == r:
            idx += 1
        indptr_list.append(idx)
    data = torch.tensor(vals_s, dtype=dtype, device=device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    block_nnz=256,
    max_segments=64,
):
    """Run SpMV on one .mtx: load, compute ref, run Triton and optional cuSPARSE, return errors and timings."""
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(mtx_path, dtype=value_dtype, device=device)
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    x = torch.randn(n_cols, dtype=value_dtype, device=device)
    ref_y = None
    try:
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        coo = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data,
            shape,
            device=device,
        ).coalesce()
        ref_y = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
    except Exception as e:
        return {
            "path": mtx_path,
            "shape": shape,
            "nnz": nnz,
            "error": f"ref: {e}",
            "triton_ms": None,
            "cusparse_ms": None,
            "triton_err": None,
            "ref_err": None,
            "status": "REF_FAIL",
        }
    triton_y, triton_ms = ast.flagsparse_spmv_csr(
        data, indices, indptr, x, shape,
        block_nnz=block_nnz, max_segments=max_segments, return_time=True,
    )
    triton_err = float(torch.max(torch.abs(triton_y - ref_y)).item()) if n_rows else 0.0
    atol, rtol = 1e-5, 1e-4
    if value_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 2e-3, 2e-3
    elif value_dtype == torch.float64 or value_dtype == torch.complex128:
        atol, rtol = 1e-10, 1e-8
    triton_ok = torch.allclose(triton_y, ref_y, atol=atol, rtol=rtol)
    pytorch_ms = None
    pytorch_ok = True
    pytorch_err = 0.0
    try:
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = torch.sparse.mm(coo, x.unsqueeze(1))
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            _ = torch.sparse.mm(coo, x.unsqueeze(1))
        end_ev.record()
        torch.cuda.synchronize()
        pytorch_ms = start_ev.elapsed_time(end_ev) / iters
    except Exception:
        pytorch_ms = None
    cusparse_ms = None
    cusparse_ok = None
    cusparse_err = None
    csc_ms = None
    if run_cusparse and value_dtype not in (torch.bfloat16,):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
            ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
            x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
            A_csr = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = A_csr @ x_cp
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = A_csr @ x_cp
            end.record()
            torch.cuda.synchronize()
            cusparse_ms = start.elapsed_time(end) / iters
            cs_y = A_csr @ x_cp
            cs_y_t = torch.utils.dlpack.from_dlpack(cs_y.toDlpack())
            cusparse_ok = torch.allclose(cs_y_t, ref_y, atol=atol, rtol=rtol)
            cusparse_err = float(torch.max(torch.abs(cs_y_t - ref_y)).item()) if n_rows else 0.0
            A_csc = A_csr.tocsc()
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = A_csc @ x_cp
            torch.cuda.synchronize()
            start.record()
            for _ in range(iters):
                _ = A_csc @ x_cp
            end.record()
            torch.cuda.synchronize()
            csc_ms = start.elapsed_time(end) / iters
        except Exception:
            cusparse_ms = None
            cusparse_ok = None
            cusparse_err = None
            csc_ms = None
    else:
        cusparse_err = None
    status = "PASS" if triton_ok and (cusparse_ok is None or cusparse_ok) else "FAIL"
    return {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "error": None,
        "triton_ms": triton_ms,
        "cusparse_ms": cusparse_ms,
        "pytorch_ms": pytorch_ms,
        "csc_ms": csc_ms,
        "triton_err": triton_err,
        "cusparse_err": cusparse_err,
        "pytorch_err": pytorch_err,
        "triton_ok": triton_ok,
        "cusparse_ok": cusparse_ok,
        "pytorch_ok": pytorch_ok,
        "status": status,
    }


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
):
    """Batch run SpMV on multiple .mtx files; return list of result dicts."""
    results = []
    for path in mtx_paths:
        r = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            run_cusparse=run_cusparse,
        )
        results.append(r)
    return results


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def print_mtx_results(results, value_dtype, index_dtype):
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}"
    )
    print("Formats: FlagSparse=CSR, cuSPARSE=CSR/CSC, PyTorch=COO")
    print("-" * 150)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
        f"{'Triton(ms)':>10} {'CSR(ms)':>10} {'CSC(ms)':>10} {'PyTorch(ms)':>11} "
        f"{'T/CSR':>7} {'T/PT':>7} {'Status':>6} {'Err(T)':>10} {'Err(CSR)':>10}"
    )
    print("-" * 150)
    for r in results:
        name = os.path.basename(r["path"])[:27]
        if len(os.path.basename(r["path"])) > 27:
            name = name + "…"
        n_rows, n_cols = r["shape"]
        triton_ms = r.get("triton_ms")
        csr_ms = r.get("cusparse_ms")
        csc_ms = r.get("csc_ms")
        pt_ms = r.get("pytorch_ms")
        err_csr = _fmt_err(r.get("cusparse_err"))
        print(
            f"{name:<28} {n_rows:>7} {n_cols:>7} {r['nnz']:>10} "
            f"{_fmt_ms(triton_ms):>10} {_fmt_ms(csr_ms):>10} {_fmt_ms(csc_ms):>10} {_fmt_ms(pt_ms):>11} "
            f"{_fmt_speedup(csr_ms, triton_ms):>7} {_fmt_speedup(pt_ms, triton_ms):>7} "
            f"{r.get('status', r.get('error', '?')):>6} {_fmt_err(r.get('triton_err')):>10} {err_csr:>10}"
        )
    print("-" * 150)


def _dtype_str(d):
    return str(d).replace("torch.", "")


def run_all_dtypes_export_csv(paths, csv_path, warmup=10, iters=50, run_cusparse=True):
    """Run SpMV for all VALUE_DTYPES x INDEX_DTYPES on each .mtx and write results to CSV."""
    rows = []
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            results = run_mtx_batch(
                paths,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
            )
            # 同时把每组 dtype/index 的表格打印出来，方便直接查看过程。
            print("=" * 120)
            print_mtx_results(results, value_dtype, index_dtype)
            for r in results:
                n_rows, n_cols = r["shape"]
                rows.append({
                    "matrix": os.path.basename(r["path"]),
                    "value_dtype": _dtype_str(value_dtype),
                    "index_dtype": _dtype_str(index_dtype),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "nnz": r["nnz"],
                    "triton_ms": r.get("triton_ms"),
                    "cusparse_ms": r.get("cusparse_ms"),
                    "pytorch_ms": r.get("pytorch_ms"),
                    "csc_ms": r.get("csc_ms"),
                    "status": r.get("status", r.get("error", "")),
                    "triton_err": r.get("triton_err"),
                    "cusparse_err": r.get("cusparse_err"),
                })
    fieldnames = [
        "matrix", "value_dtype", "index_dtype", "n_rows", "n_cols", "nnz",
        "triton_ms", "cusparse_ms", "pytorch_ms", "csc_ms",
        "status", "triton_err", "cusparse_err",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def run_comprehensive_synthetic():
    """Synthetic benchmark with per-case table (like test_gather)."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    print("=" * 110)
    print("FLAGSPARSE SpMV BENCHMARK (synthetic CSR)")
    print("=" * 110)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Warmup: {WARMUP}  Iters: {ITERS}")
    print("Formats: FlagSparse=CSR, cuSPARSE=CSR (when supported), Reference=CuPy CSR or PyTorch COO")
    print("When CuPy does not support dtype (e.g. bfloat16/float16), reference = PyTorch (float32 then cast).")
    print()
    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 110)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12}  |  Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 110)
            print(
                f"{'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                f"{'Triton(ms)':>11} {'cuSPARSE(ms)':>12} {'T/CS':>8} "
                f"{'Status':>6} {'Err(T)':>10} {'Err(CS)':>10}"
            )
            print("-" * 110)
            for n_rows, n_cols, nnz in TEST_CASES:
                result = ast.benchmark_spmv_case(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    nnz=nnz,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=WARMUP,
                    iters=ITERS,
                    run_cusparse=True,
                )
                total += 1
                perf = result["performance"]
                verify = result["verification"]
                backend = result["backend_status"]
                ok = verify["triton_match_reference"]
                cs_ok = verify.get("cusparse_match_reference")
                status = "PASS" if (ok and (cs_ok is None or cs_ok)) else "FAIL"
                if not ok or (cs_ok is False):
                    failed += 1
                triton_ms = perf["triton_ms"]
                cusparse_ms = perf["cusparse_ms"]
                speedup = perf.get("triton_speedup_vs_cusparse")
                if speedup is not None and speedup != "N/A":
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
                print(
                    f"{n_rows:>7} {n_cols:>7} {nnz:>10} "
                    f"{_fmt_ms(triton_ms):>11} {_fmt_ms(cusparse_ms):>12} {speedup_str:>8} "
                    f"{status:>6} {_fmt_err(verify.get('triton_max_error')):>10} "
                    f"{_fmt_err(verify.get('cusparse_max_error')):>10}"
                )
            print("-" * 110)
            if backend.get("cusparse_unavailable_reason"):
                print(f"  cuSPARSE: {backend['cusparse_unavailable_reason']}")
                print("  Reference: PyTorch (float32 compute then cast to value dtype).")
            print()
    print("=" * 110)
    print(f"Total: {total}  Failed: {failed}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="SpMV test: SuiteSparse .mtx batch run, error and performance."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic benchmark instead of .mtx",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64", "complex64", "complex128"],
        help="Value dtype (default: float32)",
    )
    parser.add_argument(
        "--index-dtype",
        default="int32",
        choices=["int32", "int64"],
        help="Index dtype (default: int32)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--no-cusparse", action="store_true", help="Skip cuSPARSE baseline")
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all value_dtype x index_dtype on all .mtx (CSR) and write results to CSV",
    )
    args = parser.parse_args()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    index_map = {"int32": torch.int32, "int64": torch.int64}
    value_dtype = dtype_map[args.dtype]
    index_dtype = index_map[args.index_dtype]
    if args.synthetic:
        run_comprehensive_synthetic()
        return
    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if not paths and not args.csv_csr:
        print("No .mtx files given. Use: python test_spmv.py <file.mtx> [file2.mtx ...] or <dir/>")
        print("Or run synthetic: python test_spmv.py --synthetic")
        print("Or run all dtypes and export CSR CSV: python test_spmv.py <dir/> --csv-csr results.csv")
        return
    if args.csv_csr is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        print("=" * 80)
        print("FLAGSPARSE SpMV (CSR) — all dtypes, export to CSV")
        print("=" * 80)
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  CSV: {args.csv_csr}")
        run_all_dtypes_export_csv(
            paths,
            args.csv_csr,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
        )
        return
    print("=" * 120)
    print("FLAGSPARSE SpMV — SuiteSparse .mtx batch (error + performance)")
    print("=" * 120)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  warmup: {args.warmup}  iters: {args.iters}")
    print()
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=args.warmup,
        iters=args.iters,
        run_cusparse=not args.no_cusparse,
    )
    print_mtx_results(results, value_dtype, index_dtype)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    print(f"Passed: {passed} / {len(results)}")


if __name__ == "__main__":
    main()
