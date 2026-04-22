#!/usr/bin/env python3
"""Run FlagSparse pytest accuracy suites per operator and summarize results."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openpyxl import Workbook
except Exception:
    Workbook = None


DEFAULT_OPS = (
    "gather",
    "scatter",
    "spmv_csr",
    "spmv_coo",
    "spmv_coo_tocsr",
    "spmm_csr",
    "spmm_csr_opt",
    "spmm_coo",
    "spsv_csr",
    "spsv_coo",
    "spsm_csr",
    "spsm_coo",
    "spgemm_csr",
    "sddmm_csr",
)

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SUMMARY_RE = re.compile(r"(\d+)\s+([A-Za-z_]+)")
SUMMARY_LOCK = threading.Lock()


def now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_pytest_summary(text: str) -> dict[str, int]:
    clean = ANSI_RE.sub("", text)
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in SUMMARY_RE.finditer(clean):
        key = match.group(2).lower()
        if key in counts:
            counts[key] = int(match.group(1))
    counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
    return counts


def status_from_counts(counts: dict[str, int], returncode: int) -> str:
    if returncode not in (0, 5) and not any(
        counts[key] for key in ("passed", "failed", "skipped", "errors")
    ):
        return "CRASH"
    if counts["failed"] or counts["errors"] or returncode not in (0, 5):
        return "FAIL"
    if counts["passed"]:
        return "PASS"
    if counts["skipped"]:
        return "SKIP"
    return "NO_TESTS"


def read_ops(op_list: str | None, ops_arg: str | None) -> list[str]:
    if ops_arg:
        return [op.strip() for op in ops_arg.split(",") if op.strip()]
    if op_list:
        with open(op_list, encoding="utf-8") as handle:
            return [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    return list(DEFAULT_OPS)


def run_one_op(
    project_root: Path,
    op: str,
    gpu_id: int,
    mode: str,
    results_dir: Path,
    extra_pytest_args: list[str],
) -> dict[str, object]:
    op_dir = results_dir / op
    ensure_dir(op_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/pytest",
        "-m",
        op,
        "--mode",
        mode,
        "-vs",
        "-p",
        "no:cacheprovider",
        *extra_pytest_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = op_dir / "accuracy.log"
    log_path.write_text(combined, encoding="utf-8")

    counts = parse_pytest_summary(combined)
    return {
        "operator": op,
        "gpu": gpu_id,
        "status": status_from_counts(counts, proc.returncode),
        "returncode": proc.returncode,
        "log_path": str(log_path),
        **counts,
    }


def run_gpu_ops(
    project_root: Path,
    gpu_id: int,
    ops: list[str],
    mode: str,
    results_dir: Path,
    extra_pytest_args: list[str],
    results: list[dict[str, object]],
) -> None:
    for op in ops:
        result = run_one_op(
            project_root,
            op,
            gpu_id,
            mode,
            results_dir,
            extra_pytest_args,
        )
        with SUMMARY_LOCK:
            results.append(result)
            write_summary(results, results_dir)
        print(
            f"[{result['status']}] {result['operator']} "
            f"gpu={result['gpu']} passed={result['passed']} failed={result['failed']} "
            f"skipped={result['skipped']} errors={result['errors']}"
        )


def write_summary(results: list[dict[str, object]], results_dir: Path) -> None:
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    csv_path = results_dir / "summary.csv"
    headers = [
        "operator",
        "gpu",
        "status",
        "passed",
        "failed",
        "skipped",
        "errors",
        "total",
        "returncode",
        "log_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row.get(key, "") for key in headers})

    if Workbook is None:
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(headers)
    for row in ordered:
        ws.append([row.get(key, "") for key in headers])
    wb.save(results_dir / "summary.xlsx")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-list", default=None, help="File with one pytest marker per line.")
    parser.add_argument("--ops", default=None, help="Comma-separated pytest markers; overrides --op-list.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--mode", default="quick", choices=("quick", "normal"))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Extra pytest args appended to every per-op invocation.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ops = read_ops(args.op_list, args.ops)
    if not ops:
        raise SystemExit("no operators to run")
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("no GPUs provided")
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else project_root / f"pytest_results_{now_ts()}"
    )
    ensure_dir(results_dir)
    extra_pytest_args = shlex.split(args.pytest_args) if args.pytest_args else []

    tasks = {gpu: [] for gpu in gpus}
    for index, op in enumerate(ops):
        tasks[gpus[index % len(gpus)]].append(op)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for gpu, gpu_ops in tasks.items():
            if not gpu_ops:
                continue
            futures.append(
                executor.submit(
                    run_gpu_ops,
                    project_root,
                    gpu,
                    gpu_ops,
                    args.mode,
                    results_dir,
                    extra_pytest_args,
                    results,
                )
            )
        for future in as_completed(futures):
            future.result()

    write_summary(results, results_dir)
    return 1 if any(result["status"] in ("FAIL", "NO_TESTS", "CRASH") for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
