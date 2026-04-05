"""
Diagnose float32 SpMM-opt accuracy issues on a single CSR matrix.

Usage:
    python tests/diagnose_spmm_opt.py path/to/matrix.mtx --dense-cols 32 --seed 0
    python tests/diagnose_spmm_opt.py path/to/matrix.mtx --dense-cols 32 --seed 0 --row-csv rows.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
from test_spmm_opt import _build_reference, _seeded_dense_matrix, load_mtx_to_csr_torch


def _dtype_from_name(name):
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[name]


def _bucket_row_nnz(prepared, rows):
    if rows.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    return prepared.row_lengths.index_select(0, rows.to(prepared.row_lengths.device)).to(torch.int64)


def _describe_quantiles(values):
    if values.numel() == 0:
        return "empty"
    values = values.to(torch.float64)
    q = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99], device=values.device))
    return (
        f"min={float(values.min().item()):.2f}, "
        f"p50={float(q[0].item()):.2f}, "
        f"p90={float(q[1].item()):.2f}, "
        f"p99={float(q[2].item()):.2f}, "
        f"max={float(values.max().item()):.2f}"
    )


def _write_row_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "row_nnz",
                "max_ratio",
                "mean_ratio",
                "max_abs_diff",
                "worst_col",
                "worst_ref",
                "worst_opt",
                "worst_base",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def diagnose_one(path, dense_cols, dtype, seed, topk, row_csv):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)

    ref = _build_reference(data, indices, indptr, B, shape, dtype)
    base = fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    prepared = fs.prepare_spmm_csr_opt(data, indices, indptr, shape)
    opt = fs.flagsparse_spmm_csr_opt(B=B, prepared=prepared)

    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-2
    else:
        atol, rtol = 1e-12, 1e-10

    diff_opt = torch.abs(opt - ref).to(torch.float64)
    diff_base = torch.abs(base - ref).to(torch.float64)
    denom = (atol + rtol * torch.abs(ref)).to(torch.float64)
    ratio_opt = diff_opt / denom
    ratio_base = diff_base / denom
    row_max_ratio = torch.max(ratio_opt, dim=1).values
    row_mean_ratio = torch.mean(ratio_opt, dim=1)
    row_max_abs = torch.max(diff_opt, dim=1).values
    row_worst_col = torch.argmax(ratio_opt, dim=1)
    bad_row_mask = row_max_ratio > 1.0

    print("=" * 120)
    print(f"Matrix: {os.path.basename(path)}")
    print(f"dtype={str(dtype).replace('torch.', '')}  dense_cols={dense_cols}  seed={seed}")
    print(f"shape={shape}  nnz={data.numel()}  avg_nnz_per_row={data.numel() / max(1, n_rows):.2f}")
    print(f"global_err_base={float(torch.max(ratio_base).item()):.6f}  global_err_opt={float(torch.max(ratio_opt).item()):.6f}")
    print(f"row_nnz_quantiles: {_describe_quantiles(prepared.row_lengths)}")
    print(f"row_err_quantiles: {_describe_quantiles(row_max_ratio)}")
    print(f"rows_with_err_gt_1: {int(torch.count_nonzero(bad_row_mask).item())} / {n_rows}")
    print("-" * 120)
    print("Bucket summary")
    for bucket in prepared.row_buckets:
        rows = bucket["rows"]
        if rows.numel() == 0:
            continue
        bucket_row_nnz = _bucket_row_nnz(prepared, rows)
        bucket_ratio = row_max_ratio.index_select(0, rows.to(row_max_ratio.device))
        bucket_bad = bucket_ratio > 1.0
        print(
            f"kind={bucket['kind']:<7} rows={rows.numel():>8} "
            f"row_nnz[min/mean/max]={int(bucket_row_nnz.min().item()):>4}/"
            f"{float(bucket_row_nnz.to(torch.float64).mean().item()):>8.2f}/"
            f"{int(bucket_row_nnz.max().item()):>4} "
            f"bad_rows={int(torch.count_nonzero(bucket_bad).item()):>6} "
            f"worst_ratio={float(bucket_ratio.max().item()):>10.4f}"
        )

    topk = max(1, min(int(topk), n_rows))
    top_values, top_rows = torch.topk(row_max_ratio, k=topk)
    print("-" * 120)
    print(f"Top-{topk} bad rows")
    csv_rows = []
    for rank, (row_id, row_ratio) in enumerate(zip(top_rows.tolist(), top_values.tolist()), start=1):
        worst_col = int(row_worst_col[row_id].item())
        row_nnz = int(prepared.row_lengths[row_id].item())
        worst_ref = float(ref[row_id, worst_col].item())
        worst_opt = float(opt[row_id, worst_col].item())
        worst_base = float(base[row_id, worst_col].item())
        max_abs_diff = float(row_max_abs[row_id].item())
        mean_ratio = float(row_mean_ratio[row_id].item())
        print(
            f"{rank:>2}. row={row_id:>8} row_nnz={row_nnz:>6} "
            f"max_ratio={row_ratio:>10.4f} mean_ratio={mean_ratio:>10.4f} "
            f"max_abs_diff={max_abs_diff:>10.4e} worst_col={worst_col:>4} "
            f"ref={worst_ref:>12.4e} opt={worst_opt:>12.4e} base={worst_base:>12.4e}"
        )
        csv_rows.append(
            {
                "row": row_id,
                "row_nnz": row_nnz,
                "max_ratio": row_ratio,
                "mean_ratio": mean_ratio,
                "max_abs_diff": max_abs_diff,
                "worst_col": worst_col,
                "worst_ref": worst_ref,
                "worst_opt": worst_opt,
                "worst_base": worst_base,
            }
        )

    if row_csv:
        _write_row_csv(row_csv, csv_rows)
        print("-" * 120)
        print(f"Wrote top-row diagnostics to {row_csv}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SpMM-opt accuracy on one matrix.")
    parser.add_argument("mtx", help="Path to one .mtx file")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--row-csv", type=str, default=None, help="Optional CSV path for top-row diagnostics")
    args = parser.parse_args()

    diagnose_one(
        path=args.mtx,
        dense_cols=args.dense_cols,
        dtype=_dtype_from_name(args.dtype),
        seed=args.seed,
        topk=args.topk,
        row_csv=args.row_csv,
    )


if __name__ == "__main__":
    main()
