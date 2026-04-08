"""
Batch-capable diagnostics for CSR SpMM-opt native accuracy.

Usage:
    python tests/diagnose_spmm_opt.py path/to/matrix.mtx --dense-cols 32 --seed 0
    python tests/diagnose_spmm_opt.py path/to/mtx_dir --dense-cols 32 --seed 0 --out-dir diag_out
    python tests/diagnose_spmm_opt.py path/to/mtx_dir --csv spmm_opt.csv --only-status fail --out-dir diag_out
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
import flagsparse.sparse_operations.spmm_csr as fs_spmm_csr
from test_spmm_opt import _seeded_dense_matrix, load_mtx_to_csr_torch


SUMMARY_FIELDS = [
    "matrix",
    "value_dtype",
    "dense_cols",
    "seed",
    "n_rows",
    "n_cols",
    "nnz",
    "avg_nnz_per_row",
    "base_err_native",
    "base_err_hp",
    "legacy_err_native",
    "candidate_err_native",
    "legacy_err_hp",
    "candidate_err_hp",
    "legacy_rows_native_gt_1",
    "candidate_rows_native_gt_1",
    "legacy_rows_native_gt_0_5",
    "candidate_rows_native_gt_0_5",
    "legacy_worst_row",
    "candidate_worst_row",
    "legacy_worst_row_nnz",
    "candidate_worst_row_nnz",
    "legacy_worst_col",
    "candidate_worst_col",
    "legacy_worst_native_ratio",
    "candidate_worst_native_ratio",
    "legacy_status",
    "candidate_status",
    "status",
]

ROW_FIELDS = [
    "row",
    "row_nnz",
    "bucket_kind",
    "legacy_max_native_ratio",
    "candidate_max_native_ratio",
    "legacy_mean_native_ratio",
    "candidate_mean_native_ratio",
    "legacy_max_hp_ratio",
    "candidate_max_hp_ratio",
    "legacy_max_abs_native_diff",
    "candidate_max_abs_native_diff",
    "legacy_max_abs_hp_diff",
    "candidate_max_abs_hp_diff",
    "legacy_worst_col",
    "candidate_worst_col",
    "legacy_worst_native_ref",
    "candidate_worst_native_ref",
    "legacy_worst_hp_ref",
    "candidate_worst_hp_ref",
    "legacy_worst_opt",
    "candidate_worst_opt",
]

BUCKET_FIELDS = [
    "bucket_kind",
    "row_count",
    "row_nnz_min",
    "row_nnz_mean",
    "row_nnz_max",
    "legacy_bad_rows_native_gt_1",
    "candidate_bad_rows_native_gt_1",
    "legacy_bad_rows_native_gt_0_5",
    "candidate_bad_rows_native_gt_0_5",
    "legacy_worst_native_ratio",
    "candidate_worst_native_ratio",
    "legacy_p50_native_ratio",
    "candidate_p50_native_ratio",
    "legacy_p90_native_ratio",
    "candidate_p90_native_ratio",
    "legacy_p99_native_ratio",
    "candidate_p99_native_ratio",
    "legacy_p99_hp_ratio",
    "candidate_p99_hp_ratio",
]


def _dtype_from_name(name):
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_matrix_name(name):
    base = os.path.basename(str(name).strip())
    if not base:
        return ""
    return base if base.endswith(".mtx") else f"{base}.mtx"


def _safe_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_input_paths(input_path):
    if os.path.isfile(input_path):
        if not str(input_path).lower().endswith(".mtx"):
            raise ValueError(f"Input file must be a .mtx file, got: {input_path}")
        return [os.path.abspath(input_path)]
    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "*.mtx")))
        if not paths:
            raise ValueError(f"No .mtx files found under directory: {input_path}")
        return [os.path.abspath(path) for path in paths]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _load_status_metadata(csv_path):
    if csv_path is None:
        return {}
    metadata = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            matrix = _normalize_matrix_name(row.get("matrix", ""))
            if matrix:
                metadata[matrix] = row
    return metadata


def _build_native_reference(data, indices, indptr, B, shape, dtype):
    sparse = torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data.to(dtype),
        size=shape,
        device=data.device,
    )
    return torch.sparse.mm(sparse, B.to(dtype)).to(dtype)


def _build_hp_reference(data, indices, indptr, B, shape, dtype):
    if dtype == torch.float32:
        ref_dtype = torch.float64
    else:
        ref_dtype = dtype
    sparse = torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data.to(ref_dtype),
        size=shape,
        device=data.device,
    )
    return torch.sparse.mm(sparse, B.to(ref_dtype)).to(dtype)


def _reference_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-2
    return 1e-12, 1e-10


def _quantiles(values):
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    compare = values.to(torch.float64)
    q = torch.quantile(
        compare,
        torch.tensor([0.5, 0.9, 0.99], device=compare.device, dtype=compare.dtype),
    )
    return float(q[0].item()), float(q[1].item()), float(q[2].item())


def _describe_quantiles(values):
    if values.numel() == 0:
        return "empty"
    p50, p90, p99 = _quantiles(values)
    compare = values.to(torch.float64)
    return (
        f"min={float(compare.min().item()):.2f}, "
        f"p50={p50:.2f}, "
        f"p90={p90:.2f}, "
        f"p99={p99:.2f}, "
        f"max={float(compare.max().item()):.2f}"
    )


def _compute_error_metrics(candidate, reference, dtype):
    atol, rtol = _reference_tolerance(dtype)
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    ratio = diff / denom
    row_max_ratio = torch.max(ratio, dim=1).values
    row_mean_ratio = torch.mean(ratio, dim=1)
    row_max_abs = torch.max(diff, dim=1).values
    row_worst_col = torch.argmax(ratio, dim=1)
    if row_max_ratio.numel() == 0:
        worst_row = 0
        worst_col = 0
        worst_ratio = 0.0
    else:
        worst_row = int(torch.argmax(row_max_ratio).item())
        worst_col = int(row_worst_col[worst_row].item())
        worst_ratio = float(row_max_ratio[worst_row].item())
    return {
        "global_err": float(torch.max(ratio).item()) if ratio.numel() > 0 else 0.0,
        "ratio": ratio,
        "row_max_ratio": row_max_ratio,
        "row_mean_ratio": row_mean_ratio,
        "row_max_abs": row_max_abs,
        "row_worst_col": row_worst_col,
        "rows_err_gt_1": int(torch.count_nonzero(row_max_ratio > 1.0).item()),
        "rows_err_gt_0_5": int(torch.count_nonzero(row_max_ratio > 0.5).item()),
        "worst_row": worst_row,
        "worst_col": worst_col,
        "worst_ratio": worst_ratio,
        "status": "PASS" if (worst_ratio <= 1.0) else "FAIL",
    }


def _candidate_bucket_rows(prepared):
    bucket_rows = {}
    for bucket in fs_spmm_csr._build_spmm_opt_candidate_buckets(prepared):
        bucket_rows[bucket["label"]] = bucket["rows"]
    return bucket_rows


def _row_bucket_names(prepared):
    names = ["unassigned"] * prepared.n_rows
    for label, rows in _candidate_bucket_rows(prepared).items():
        for row in rows.to(torch.int64).cpu().tolist():
            names[row] = label
    return names


def _bucket_row_nnz(prepared, rows):
    if rows.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=prepared.row_lengths.device)
    return prepared.row_lengths.index_select(0, rows.to(prepared.row_lengths.device)).to(torch.int64)


def _build_bucket_rows(prepared, legacy_native, candidate_native, legacy_hp, candidate_hp):
    rows = []
    for label, bucket_rows in _candidate_bucket_rows(prepared).items():
        if bucket_rows.numel() == 0:
            continue
        bucket_row_nnz = _bucket_row_nnz(prepared, bucket_rows)
        rows_device = bucket_rows.to(legacy_native["row_max_ratio"].device)
        legacy_native_ratio = legacy_native["row_max_ratio"].index_select(0, rows_device)
        candidate_native_ratio = candidate_native["row_max_ratio"].index_select(0, rows_device)
        legacy_hp_ratio = legacy_hp["row_max_ratio"].index_select(0, rows_device)
        candidate_hp_ratio = candidate_hp["row_max_ratio"].index_select(0, rows_device)
        legacy_p50_native, legacy_p90_native, legacy_p99_native = _quantiles(legacy_native_ratio)
        candidate_p50_native, candidate_p90_native, candidate_p99_native = _quantiles(candidate_native_ratio)
        _, _, legacy_p99_hp = _quantiles(legacy_hp_ratio)
        _, _, candidate_p99_hp = _quantiles(candidate_hp_ratio)
        rows.append(
            {
                "bucket_kind": label,
                "row_count": int(bucket_rows.numel()),
                "row_nnz_min": int(bucket_row_nnz.min().item()),
                "row_nnz_mean": float(bucket_row_nnz.to(torch.float64).mean().item()),
                "row_nnz_max": int(bucket_row_nnz.max().item()),
                "legacy_bad_rows_native_gt_1": int(torch.count_nonzero(legacy_native_ratio > 1.0).item()),
                "candidate_bad_rows_native_gt_1": int(torch.count_nonzero(candidate_native_ratio > 1.0).item()),
                "legacy_bad_rows_native_gt_0_5": int(torch.count_nonzero(legacy_native_ratio > 0.5).item()),
                "candidate_bad_rows_native_gt_0_5": int(torch.count_nonzero(candidate_native_ratio > 0.5).item()),
                "legacy_worst_native_ratio": float(legacy_native_ratio.max().item()),
                "candidate_worst_native_ratio": float(candidate_native_ratio.max().item()),
                "legacy_p50_native_ratio": legacy_p50_native,
                "candidate_p50_native_ratio": candidate_p50_native,
                "legacy_p90_native_ratio": legacy_p90_native,
                "candidate_p90_native_ratio": candidate_p90_native,
                "legacy_p99_native_ratio": legacy_p99_native,
                "candidate_p99_native_ratio": candidate_p99_native,
                "legacy_p99_hp_ratio": legacy_p99_hp,
                "candidate_p99_hp_ratio": candidate_p99_hp,
            }
        )
    return rows


def _build_row_rows(
    prepared,
    native_ref,
    hp_ref,
    legacy_out,
    candidate_out,
    legacy_native,
    candidate_native,
    legacy_hp,
    candidate_hp,
    bucket_names,
):
    rows = []
    for row_id in range(prepared.n_rows):
        legacy_col = int(legacy_native["row_worst_col"][row_id].item())
        candidate_col = int(candidate_native["row_worst_col"][row_id].item())
        rows.append(
            {
                "row": row_id,
                "row_nnz": int(prepared.row_lengths[row_id].item()),
                "bucket_kind": bucket_names[row_id],
                "legacy_max_native_ratio": float(legacy_native["row_max_ratio"][row_id].item()),
                "candidate_max_native_ratio": float(candidate_native["row_max_ratio"][row_id].item()),
                "legacy_mean_native_ratio": float(legacy_native["row_mean_ratio"][row_id].item()),
                "candidate_mean_native_ratio": float(candidate_native["row_mean_ratio"][row_id].item()),
                "legacy_max_hp_ratio": float(legacy_hp["row_max_ratio"][row_id].item()),
                "candidate_max_hp_ratio": float(candidate_hp["row_max_ratio"][row_id].item()),
                "legacy_max_abs_native_diff": float(legacy_native["row_max_abs"][row_id].item()),
                "candidate_max_abs_native_diff": float(candidate_native["row_max_abs"][row_id].item()),
                "legacy_max_abs_hp_diff": float(legacy_hp["row_max_abs"][row_id].item()),
                "candidate_max_abs_hp_diff": float(candidate_hp["row_max_abs"][row_id].item()),
                "legacy_worst_col": legacy_col,
                "candidate_worst_col": candidate_col,
                "legacy_worst_native_ref": float(native_ref[row_id, legacy_col].item()),
                "candidate_worst_native_ref": float(native_ref[row_id, candidate_col].item()),
                "legacy_worst_hp_ref": float(hp_ref[row_id, legacy_col].item()),
                "candidate_worst_hp_ref": float(hp_ref[row_id, candidate_col].item()),
                "legacy_worst_opt": float(legacy_out[row_id, legacy_col].item()),
                "candidate_worst_opt": float(candidate_out[row_id, candidate_col].item()),
            }
        )
    return rows


def _print_matrix_summary(summary_row, prepared, legacy_native, candidate_native, bucket_rows):
    print("=" * 120)
    print(f"Matrix: {summary_row['matrix']}")
    print(
        f"dtype={summary_row['value_dtype']}  dense_cols={summary_row['dense_cols']}  "
        f"seed={summary_row['seed']}"
    )
    print(
        f"shape=({summary_row['n_rows']}, {summary_row['n_cols']})  "
        f"nnz={summary_row['nnz']}  avg_nnz_per_row={summary_row['avg_nnz_per_row']:.2f}"
    )
    print(
        f"base_err_native={summary_row['base_err_native']:.6f}  "
        f"base_err_hp={summary_row['base_err_hp']:.6f}"
    )
    print(
        f"legacy_err_native={summary_row['legacy_err_native']:.6f}  "
        f"candidate_err_native={summary_row['candidate_err_native']:.6f}  "
        f"legacy_err_hp={summary_row['legacy_err_hp']:.6f}  "
        f"candidate_err_hp={summary_row['candidate_err_hp']:.6f}  "
        f"status={summary_row['status']}"
    )
    print(f"row_nnz_quantiles: {_describe_quantiles(prepared.row_lengths)}")
    print(f"legacy_native_row_err_quantiles: {_describe_quantiles(legacy_native['row_max_ratio'])}")
    print(f"candidate_native_row_err_quantiles: {_describe_quantiles(candidate_native['row_max_ratio'])}")
    print(
        f"legacy rows native err>1: {summary_row['legacy_rows_native_gt_1']} / {summary_row['n_rows']}  "
        f"candidate rows native err>1: {summary_row['candidate_rows_native_gt_1']} / {summary_row['n_rows']}"
    )
    print("-" * 120)
    print("Bucket summary")
    for bucket_row in bucket_rows:
        print(
            f"kind={bucket_row['bucket_kind']:<11} rows={bucket_row['row_count']:>8} "
            f"row_nnz[min/mean/max]={bucket_row['row_nnz_min']:>4}/"
            f"{bucket_row['row_nnz_mean']:>8.2f}/"
            f"{bucket_row['row_nnz_max']:>4} "
            f"legacy_bad_gt_1={bucket_row['legacy_bad_rows_native_gt_1']:>6} "
            f"candidate_bad_gt_1={bucket_row['candidate_bad_rows_native_gt_1']:>6} "
            f"legacy_worst={bucket_row['legacy_worst_native_ratio']:>10.4f} "
            f"candidate_worst={bucket_row['candidate_worst_native_ratio']:>10.4f}"
        )


def _print_top_rows(row_rows, topk):
    top_rows = sorted(
        row_rows,
        key=lambda row: float(row["candidate_max_native_ratio"]),
        reverse=True,
    )[:topk]
    print("-" * 120)
    print(f"Top-{len(top_rows)} rows by candidate native ratio")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:>2}. row={row['row']:>8} row_nnz={row['row_nnz']:>6} "
            f"bucket={row['bucket_kind']:<11} "
            f"legacy_native={row['legacy_max_native_ratio']:>10.4f} "
            f"candidate_native={row['candidate_max_native_ratio']:>10.4f} "
            f"legacy_hp={row['legacy_max_hp_ratio']:>10.4f} "
            f"candidate_hp={row['candidate_max_hp_ratio']:>10.4f}"
        )


def diagnose_one(path, dense_cols, dtype, seed, topk):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)

    native_ref = _build_native_reference(data, indices, indptr, B, shape, dtype)
    hp_ref = _build_hp_reference(data, indices, indptr, B, shape, dtype)
    base = fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    prepared = fs.prepare_spmm_csr_opt(data, indices, indptr, shape)
    legacy_out = fs.flagsparse_spmm_csr_opt(B=B, prepared=prepared)
    candidate_out = fs_spmm_csr._flagsparse_spmm_csr_opt_candidate_for_diagnose(prepared, B)

    base_native = _compute_error_metrics(base, native_ref, dtype)
    base_hp = _compute_error_metrics(base, hp_ref, dtype)
    legacy_native = _compute_error_metrics(legacy_out, native_ref, dtype)
    candidate_native = _compute_error_metrics(candidate_out, native_ref, dtype)
    legacy_hp = _compute_error_metrics(legacy_out, hp_ref, dtype)
    candidate_hp = _compute_error_metrics(candidate_out, hp_ref, dtype)
    bucket_names = _row_bucket_names(prepared)

    summary_row = {
        "matrix": os.path.basename(path),
        "value_dtype": str(dtype).replace("torch.", ""),
        "dense_cols": int(dense_cols),
        "seed": seed,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nnz": int(data.numel()),
        "avg_nnz_per_row": float(data.numel()) / max(1, int(n_rows)),
        "base_err_native": base_native["global_err"],
        "base_err_hp": base_hp["global_err"],
        "legacy_err_native": legacy_native["global_err"],
        "candidate_err_native": candidate_native["global_err"],
        "legacy_err_hp": legacy_hp["global_err"],
        "candidate_err_hp": candidate_hp["global_err"],
        "legacy_rows_native_gt_1": legacy_native["rows_err_gt_1"],
        "candidate_rows_native_gt_1": candidate_native["rows_err_gt_1"],
        "legacy_rows_native_gt_0_5": legacy_native["rows_err_gt_0_5"],
        "candidate_rows_native_gt_0_5": candidate_native["rows_err_gt_0_5"],
        "legacy_worst_row": legacy_native["worst_row"],
        "candidate_worst_row": candidate_native["worst_row"],
        "legacy_worst_row_nnz": int(prepared.row_lengths[legacy_native["worst_row"]].item()) if n_rows > 0 else 0,
        "candidate_worst_row_nnz": int(prepared.row_lengths[candidate_native["worst_row"]].item()) if n_rows > 0 else 0,
        "legacy_worst_col": legacy_native["worst_col"],
        "candidate_worst_col": candidate_native["worst_col"],
        "legacy_worst_native_ratio": legacy_native["worst_ratio"],
        "candidate_worst_native_ratio": candidate_native["worst_ratio"],
        "legacy_status": legacy_native["status"],
        "candidate_status": candidate_native["status"],
        "status": candidate_native["status"],
    }

    row_rows = _build_row_rows(
        prepared,
        native_ref,
        hp_ref,
        legacy_out,
        candidate_out,
        legacy_native,
        candidate_native,
        legacy_hp,
        candidate_hp,
        bucket_names,
    )
    bucket_rows = _build_bucket_rows(prepared, legacy_native, candidate_native, legacy_hp, candidate_hp)

    _print_matrix_summary(summary_row, prepared, legacy_native, candidate_native, bucket_rows)
    _print_top_rows(row_rows, max(1, min(int(topk), len(row_rows))))
    return summary_row, row_rows, bucket_rows


def _collect_selected_paths(all_paths, csv_metadata, only_status, only_matrices):
    allowed_names = None
    if only_matrices:
        allowed_names = {
            _normalize_matrix_name(name)
            for name in only_matrices.split(",")
            if _normalize_matrix_name(name)
        }

    selected = []
    for path in all_paths:
        matrix_name = _normalize_matrix_name(os.path.basename(path))
        if allowed_names is not None and matrix_name not in allowed_names:
            continue
        if only_status != "all" and matrix_name in csv_metadata:
            csv_status = str(csv_metadata[matrix_name].get("status", "")).strip().lower()
            if csv_status != only_status:
                continue
        selected.append(path)
    return selected


def _filter_results_by_status(results, only_status):
    if only_status == "all":
        return results
    target = only_status.upper()
    return [result for result in results if result["summary"]["status"] == target]


def _default_output_paths(out_dir):
    abs_out_dir = os.path.abspath(out_dir)
    rows_dir = os.path.join(abs_out_dir, "diag_rows")
    buckets_dir = os.path.join(abs_out_dir, "diag_buckets")
    summary_csv = os.path.join(abs_out_dir, "spmm_diag_summary.csv")
    os.makedirs(rows_dir, exist_ok=True)
    os.makedirs(buckets_dir, exist_ok=True)
    return summary_csv, rows_dir, buckets_dir


def run_batch(input_path, dtype, dense_cols, seed, topk, csv_path=None, out_dir="spmm_diag", only_status="all", only_matrices=None):
    all_paths = _resolve_input_paths(input_path)
    csv_metadata = _load_status_metadata(csv_path)
    selected_paths = _collect_selected_paths(all_paths, csv_metadata, only_status, only_matrices)
    if not selected_paths:
        raise ValueError("No matrices matched the requested filters.")

    summary_csv, rows_dir, buckets_dir = _default_output_paths(out_dir)
    results = []
    for index, path in enumerate(selected_paths, start=1):
        print(f"[{index}/{len(selected_paths)}] Diagnosing {os.path.basename(path)}")
        summary_row, row_rows, bucket_rows = diagnose_one(
            path=path,
            dense_cols=dense_cols,
            dtype=dtype,
            seed=seed,
            topk=topk,
        )
        results.append({"path": path, "summary": summary_row, "rows": row_rows, "buckets": bucket_rows})

    filtered_results = _filter_results_by_status(results, only_status)
    if not filtered_results:
        raise ValueError("Diagnostics completed, but no matrices matched the requested status filter.")

    _write_csv(summary_csv, [result["summary"] for result in filtered_results], SUMMARY_FIELDS)
    for result in filtered_results:
        stem = _safe_stem(result["path"])
        _write_csv(os.path.join(rows_dir, f"{stem}.rows.csv"), result["rows"], ROW_FIELDS)
        _write_csv(os.path.join(buckets_dir, f"{stem}.buckets.csv"), result["buckets"], BUCKET_FIELDS)

    print("=" * 120)
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote row diagnostics under {rows_dir}")
    print(f"Wrote bucket diagnostics under {buckets_dir}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose native CSR SpMM-opt accuracy for one matrix or a directory of matrices.")
    parser.add_argument("input_path", help="Path to one .mtx file or a directory containing .mtx files")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10, help="How many top bad rows to print")
    parser.add_argument("--csv", type=str, default=None, help="Optional existing CSV used for metadata or prefiltering")
    parser.add_argument("--out-dir", type=str, default="spmm_diag", help="Output directory for batch summary and detail CSVs")
    parser.add_argument("--only-status", choices=["all", "fail", "pass"], default="all", help="If CSV metadata is provided, prefilter by CSV status; otherwise filter final outputs by diagnosed status")
    parser.add_argument("--only-matrices", type=str, default=None, help="Comma-separated matrix basenames to include, with or without .mtx suffix")
    parser.add_argument("--row-csv", type=str, default=None, help="Single-file mode only: explicit output path for row diagnostics CSV")
    args = parser.parse_args()

    dtype = _dtype_from_name(args.dtype)
    input_is_file = os.path.isfile(args.input_path)

    if input_is_file:
        summary_row, row_rows, bucket_rows = diagnose_one(
            path=args.input_path,
            dense_cols=args.dense_cols,
            dtype=dtype,
            seed=args.seed,
            topk=args.topk,
        )
        summary_csv, rows_dir, buckets_dir = _default_output_paths(args.out_dir)
        stem = _safe_stem(args.input_path)
        _write_csv(summary_csv, [summary_row], SUMMARY_FIELDS)
        if args.row_csv:
            _write_csv(args.row_csv, row_rows, ROW_FIELDS)
            print("-" * 120)
            print(f"Wrote row diagnostics to {args.row_csv}")
        else:
            _write_csv(os.path.join(rows_dir, f"{stem}.rows.csv"), row_rows, ROW_FIELDS)
            print("-" * 120)
            print(f"Wrote row diagnostics to {os.path.join(rows_dir, f'{stem}.rows.csv')}")
        _write_csv(os.path.join(buckets_dir, f"{stem}.buckets.csv"), bucket_rows, BUCKET_FIELDS)
        print(f"Wrote summary CSV to {summary_csv}")
        print(f"Wrote bucket diagnostics to {os.path.join(buckets_dir, f'{stem}.buckets.csv')}")
        return

    if args.row_csv:
        raise ValueError("--row-csv is only supported in single-file mode; use --out-dir in directory mode")

    run_batch(
        input_path=args.input_path,
        dtype=dtype,
        dense_cols=args.dense_cols,
        seed=args.seed,
        topk=args.topk,
        csv_path=args.csv,
        out_dir=args.out_dir,
        only_status=args.only_status,
        only_matrices=args.only_matrices,
    )


if __name__ == "__main__":
    main()
