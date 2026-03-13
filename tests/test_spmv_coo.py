"""SpMV COO tests: float32/float64 + int32, synthetic + optional .mtx, compare FlagSparse vs PyTorch/CuPy."""
import argparse
import glob
import csv
import math
import os

import torch
import flagsparse as fs
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_sparse = None

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPE = torch.int32
TEST_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
WARMUP = 5
ITERS = 20


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _dense_to_coo(A):
    rows, cols = A.nonzero(as_tuple=True)
    data = A[rows, cols]
    return data, rows, cols


def run_synthetic():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    print("=" * 110)
    print("FLAGSPARSE SpMV COO BENCHMARK (synthetic dense -> COO)")
    print("=" * 110)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iters: {ITERS}")
    print()

    for dtype in VALUE_DTYPES:
        print("-" * 110)
        print(f"dtype: {_dtype_name(dtype)}  index_dtype: int32")
        print("-" * 110)
        print(
            f"{'M':>6} {'N':>6} {'NNZ':>10} "
            f"{'FS(ms)':>10} {'PT(ms)':>10} {'CU(ms)':>10} "
            f"{'FS/PT':>8} {'FS/CU':>8} {'Err(PT)':>12} {'Err(CU)':>12}"
        )
        print("-" * 110)
        for m, n in TEST_SIZES:
            # 构造稠密矩阵
            A = torch.randn(m, n, dtype=dtype, device=device)
            A *= (torch.rand_like(A) < 0.1)
            data, row, col = _dense_to_coo(A)
            nnz = int(data.numel())
            x = torch.randn(n, dtype=dtype, device=device)

            # FlagSparse COO
            torch.cuda.synchronize()
            y_fs, fs_ms = fs.flagsparse_spmv_coo(
                data, row, col, x, (m, n), return_time=True
            )
            torch.cuda.synchronize()

            # PyTorch 稀疏 COO 基线
            if nnz > 0:
                coo = torch.sparse_coo_tensor(
                    torch.stack([row.to(torch.int64), col.to(torch.int64)]),
                    data,
                    (m, n),
                    device=device,
                ).coalesce()
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True)
                e1 = torch.cuda.Event(True)
                e0.record()
                for _ in range(ITERS):
                    y_pt = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
                e1.record()
                torch.cuda.synchronize()
                pt_ms = e0.elapsed_time(e1) / ITERS
            else:
                y_pt = torch.zeros(m, dtype=dtype, device=device)
                pt_ms = 0.0

            err_pt = float(torch.max(torch.abs(y_fs - y_pt)).item()) if m > 0 else 0.0

            # CuPy CSR 基线（若可用）
            cu_ms = None
            err_cu = None
            if cp is not None and cpx_sparse is not None:
                try:
                    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                    row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64)))
                    col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64)))
                    A_cp = cpx_sparse.coo_matrix(
                        (data_cp, (row_cp, col_cp)), shape=(m, n)
                    ).tocsr()
                    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
                    for _ in range(WARMUP):
                        _ = A_cp @ x_cp
                    cp.cuda.runtime.deviceSynchronize()
                    c0 = cp.cuda.Event()
                    c1 = cp.cuda.Event()
                    c0.record()
                    for _ in range(ITERS):
                        y_cu = A_cp @ x_cp
                    c1.record()
                    c1.synchronize()
                    cu_ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
                    y_cu_t = torch.utils.dlpack.from_dlpack(y_cu.toDlpack()).to(dtype)
                    err_cu = float(torch.max(torch.abs(y_fs - y_cu_t)).item()) if m > 0 else 0.0
                except Exception:
                    cu_ms = None
                    err_cu = None

            fs_vs_pt = (pt_ms / fs_ms) if (fs_ms and fs_ms > 0) else None
            fs_vs_cu = (cu_ms / fs_ms) if (cu_ms is not None and fs_ms and fs_ms > 0) else None
            fs_vs_pt_s = f"{fs_vs_pt:.2f}x" if fs_vs_pt is not None else "N/A"
            fs_vs_cu_s = f"{fs_vs_cu:.2f}x" if fs_vs_cu is not None else "N/A"

            print(
                f"{m:>6} {n:>6} {nnz:>10} "
                f"{_fmt_ms(fs_ms):>10} {_fmt_ms(pt_ms):>10} {_fmt_ms(cu_ms):>10} "
                f"{fs_vs_pt_s:>8} {fs_vs_cu_s:>8} {_fmt_err(err_pt):>12} {_fmt_err(err_cu):>12}"
            )
        print("-" * 110)
        print()


def _load_mtx_to_coo_torch(file_path, dtype=torch.float32, device=None):
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
    rows_host = []
    cols_host = []
    vals_host = []
    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) >= 3:
            rows_host.append(int(parts[0]) - 1)
            cols_host.append(int(parts[1]) - 1)
            vals_host.append(float(parts[2]))
    rows = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols = torch.tensor(cols_host, dtype=torch.int64, device=device)
    vals = torch.tensor(vals_host, dtype=dtype, device=device)
    return vals, rows, cols, (n_rows, n_cols)


def run_all_dtypes_coo_csv(mtx_paths, csv_path):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    for dtype in VALUE_DTYPES:
        print("=" * 110)
        print(f"COO SpMV CSV export — dtype: {_dtype_name(dtype)}")
        print("=" * 110)
        print(
            f"{'Matrix':<32} {'M':>6} {'N':>6} {'NNZ':>10} "
            f"{'FS(ms)':>10} {'PT(ms)':>10} {'CU(ms)':>10} "
            f"{'FS/PT':>8} {'FS/CU':>8} {'Err(PT)':>12} {'Err(CU)':>12}"
        )
        print("-" * 110)
        for path in mtx_paths:
            try:
                data, row, col, shape = _load_mtx_to_coo_torch(
                    path, dtype=dtype, device=device
                )
                m, n = shape
                x = torch.randn(n, dtype=dtype, device=device)
                y_fs, fs_ms = fs.flagsparse_spmv_coo(
                    data, row, col, x, shape, return_time=True
                )
                # PyTorch baseline
                if data.numel() > 0:
                    coo = torch.sparse_coo_tensor(
                        torch.stack([row, col]),
                        data,
                        shape,
                        device=device,
                    ).coalesce()
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(True)
                    e1 = torch.cuda.Event(True)
                    e0.record()
                    for _ in range(ITERS):
                        y_pt = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
                    e1.record()
                    torch.cuda.synchronize()
                    pt_ms = e0.elapsed_time(e1) / ITERS
                else:
                    y_pt = torch.zeros(m, dtype=dtype, device=device)
                    pt_ms = 0.0
                err_pt = float(torch.max(torch.abs(y_fs - y_pt)).item()) if m > 0 else 0.0
                # CuPy baseline (optional)
                cu_ms = None
                err_cu = None
                if cp is not None and cpx_sparse is not None:
                    try:
                        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                        row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row))
                        col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col))
                        A_cp = cpx_sparse.coo_matrix(
                            (data_cp, (row_cp, col_cp)), shape=shape
                        ).tocsr()
                        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
                        for _ in range(WARMUP):
                            _ = A_cp @ x_cp
                        cp.cuda.runtime.deviceSynchronize()
                        c0 = cp.cuda.Event()
                        c1 = cp.cuda.Event()
                        c0.record()
                        for _ in range(ITERS):
                            y_cu = A_cp @ x_cp
                        c1.record()
                        c1.synchronize()
                        cu_ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
                        y_cu_t = torch.utils.dlpack.from_dlpack(y_cu.toDlpack()).to(dtype)
                        err_cu = float(torch.max(torch.abs(y_fs - y_cu_t)).item()) if m > 0 else 0.0
                    except Exception:
                        cu_ms = None
                        err_cu = None
                fs_vs_pt = (pt_ms / fs_ms) if (fs_ms and fs_ms > 0) else None
                fs_vs_cu = (cu_ms / fs_ms) if (cu_ms is not None and fs_ms and fs_ms > 0) else None
                rows_out.append(
                    {
                        "matrix": os.path.basename(path),
                        "value_dtype": _dtype_name(dtype),
                        "m": m,
                        "n": n,
                        "nnz": int(data.numel()),
                        "flagsparse_ms": fs_ms,
                        "pytorch_ms": pt_ms,
                        "cupy_ms": cu_ms,
                        "fs_vs_pytorch": fs_vs_pt,
                        "fs_vs_cupy": fs_vs_cu,
                        "err_vs_pytorch": err_pt,
                        "err_vs_cupy": err_cu,
                    }
                )
                name = os.path.basename(path)
                fs_vs_pt_s = f"{fs_vs_pt:.2f}x" if fs_vs_pt is not None else "N/A"
                fs_vs_cu_s = f"{fs_vs_cu:.2f}x" if fs_vs_cu is not None else "N/A"
                print(
                    f"{name:<32} {m:>6} {n:>6} {int(data.numel()):>10} "
                    f"{_fmt_ms(fs_ms):>10} {_fmt_ms(pt_ms):>10} {_fmt_ms(cu_ms):>10} "
                    f"{fs_vs_pt_s:>8} {fs_vs_cu_s:>8} {_fmt_err(err_pt):>12} {_fmt_err(err_cu):>12}"
                )
            except Exception as e:
                rows_out.append(
                    {
                        "matrix": os.path.basename(path),
                        "value_dtype": _dtype_name(dtype),
                        "m": "ERR",
                        "n": "ERR",
                        "nnz": "ERR",
                        "flagsparse_ms": None,
                        "pytorch_ms": None,
                        "cupy_ms": None,
                        "fs_vs_pytorch": None,
                        "fs_vs_cupy": None,
                        "err_vs_pytorch": str(e),
                        "err_vs_cupy": None,
                    }
                )
    fieldnames = [
        "matrix",
        "value_dtype",
        "m",
        "n",
        "nnz",
        "flagsparse_ms",
        "pytorch_ms",
        "cupy_ms",
        "fs_vs_pytorch",
        "fs_vs_cupy",
        "err_vs_pytorch",
        "err_vs_cupy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpMV COO test: synthetic dense->COO and optional .mtx, export CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic dense->COO tests"
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on given .mtx and export COO SpMV results to CSV",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_coo_csv(paths, args.csv_coo)
        return

    print("Use --synthetic or --csv-coo to run COO SpMV tests.")


if __name__ == "__main__":
    main()
