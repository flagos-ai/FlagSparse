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

"""AlphaSparse-style CSR SpMM route benchmark.

This test is intentionally route-based: each output row represents one
algorithm for one matrix/dtype/op case, so future CSR algorithms can be swept
and ranked without changing the CSV schema.
"""

import argparse
import csv
import glob
import json
import itertools
import platform
import subprocess
import os
import sys
from pathlib import Path

import torch

from benchmark_utils import ACCEL, accelerator_device

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
from flagsparse.sparse_operations import _common as fs_common
import flagsparse.sparse_operations.spmm_csr as spmm_ops

from test_spmm import (
    _build_dense_matrix,
    _build_pytorch_reference,
    _normalize_csv_path,
    load_mtx_to_csr_torch,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}
DEFAULT_DTYPE_NAMES = ("float32", "float64", "complex64", "complex128")
DEFAULT_RUN_DTYPE_NAMES = ("float32", "float64")
DEFAULT_INDEX_DTYPE_NAMES = ("int32", "int64")
DEFAULT_OP_NAMES = tuple(spmm_ops.SPMM_OP_NAMES.values())
CUSPARSE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
TLE_CSR_SPMM_ALGORITHMS = {"alpha_alg1_tle_opt", "alpha_alg1_tle_opt2"}


PERF_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "alg",
    "n_rows",
    "n_cols",
    "nnz",
    "dense_cols",
    "b_stride",
    "c_stride",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "torch_ms",
    "cusparse_ms",
    "torch_vs_alg_speedup",
    "cusparse_vs_alg_speedup",
    "err_vs_torch",
    "err_vs_cusparse",
    "status",
    "reason",
    "cusparse_reason",
]

PERF_FIELDS += ["indptr_dtype", "vendor_ms", "vendor_raw_ms", "vendor_backend", "vendor_alg",
                "speedup_vs_vendor", "vendor_reason", "vendor_status", "vendor_error",
                "max_abs_error", "metadata", "correctness_ref"]

TIMING_FIELDS = ["process_gpu_ms", "compute_ms"]

DIAG_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "alg",
    "launch_config_scope",
    "launch_config_count",
    "bucket_count",
    "long_row_count",
    "long_part_count",
    "num_warps",
    "num_stages",
    "block_n",
    "block_nnz",
    "warp_size",
    "factor",
    "block_rows",
    "block_cols",
    "grid_m",
    "grid_n",
    "launch_version",
    "dense_layout",
    "b_stride",
    "c_stride",
    "output_layout",
]

DIAG_FIELDS += ["indptr_dtype", "dense_cols", "config", "launch_configs", "segment_count",
                "reduction_levels", "workspace_peak_bytes", "descriptor_peak_bytes_estimate"]

BEST_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "best_alg",
    "best_ms",
    "best_gpu_ms",
    "best_torch_speedup",
    "best_cusparse_speedup",
]

BEST_FIELDS += ["indptr_dtype", "dense_cols"]

LAYOUT_NAMES = ("row", "col")


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _normalize_layout_name(layout):
    token = str(layout).strip().lower()
    if token in ("row", "row_major", "row-major", "c", "c_order"):
        return "row"
    if token in (
        "col",
        "column",
        "col_major",
        "column_major",
        "col-major",
        "column-major",
        "f",
        "fortran",
    ):
        return "col"
    raise ValueError("layout must be one of: row, col, all")


def _layout_names(value):
    value = str(value).strip().lower()
    if value == "all":
        return list(LAYOUT_NAMES)
    return [_normalize_layout_name(value)]


def _materialize_dense_layout_for_test(tensor, layout):
    layout = _normalize_layout_name(layout)
    if layout == "row":
        return tensor.contiguous()
    out = torch.empty_strided(
        tuple(tensor.shape),
        (1, max(1, int(tensor.shape[0]))),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    out.copy_(tensor)
    return out


def _stride_string(tensor):
    if tensor is None:
        return ""
    return "x".join(str(int(v)) for v in tensor.stride())


def _reference_tolerance(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    if dtype == torch.float16:
        return 1e-3, 2e-3
    if dtype == torch.bfloat16:
        return 0.016, 1e-1
    return 1e-6, 1e-5


def _error_profile(candidate, reference, dtype):
    if candidate is None or reference is None:
        return {"global_err": None, "status": "SKIP"}
    if candidate.shape != reference.shape or candidate.dtype != dtype:
        return {"global_err": float("inf"), "max_abs_error": float("inf"), "status": "FAIL"}
    atol, rtol = _reference_tolerance(dtype)
    if candidate.numel() == 0:
        return {"global_err": 0.0, "max_abs_error": 0.0, "status": "PASS"}
    component_type = torch.complex128 if dtype.is_complex else torch.float64
    candidate = candidate.detach().cpu().to(component_type)
    reference = reference.detach().cpu().to(component_type)
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    ratio = diff / denom
    global_err = float(torch.max(ratio).item()) if ratio.numel() > 0 else 0.0
    return {"global_err": global_err, "max_abs_error": float(diff.max().item()), "status": "PASS" if global_err <= 1.0 else "FAIL"}


def _correctness_reference(data, indices, indptr, shape, B, op):
    # Do not first invoke unsupported device sparse/FP64 operations on platforms
    # whose established correctness policy is CPU SciPy.
    if fs_common._use_scipy_accuracy_reference():
        import reference_utils
        dtype = reference_utils.reference_dtype(data.dtype)
        matrix = reference_utils.scipy_csr(data.cpu(), indices.cpu(), indptr.cpu(), shape, dtype)
        product = reference_utils.spmm(matrix, B.cpu(), dtype, op=op)
        return reference_utils.as_torch(product, dtype, "cpu"), "SciPy/CSR CPU"
    ref, _, fmt = _build_pytorch_reference(data, indices, indptr, shape, B, op=op)
    return ref, f"PyTorch/{fmt}"


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _parse_csv_names(value, all_names, option_name, explicit_names=None):
    value = str(value).strip().lower()
    if value == "all":
        return list(all_names)
    allowed = tuple(all_names if explicit_names is None else explicit_names)
    names = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not names:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: all,{','.join(allowed)}"
        )
    return names


def _parse_algs(value):
    value = str(value).strip().lower()
    if value in ("auto", "all", "compare"):
        return ["all" if value == "compare" else value]
    allowed = set(fs.SPMM_CSR_ALGORITHMS)
    names = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not names:
        raise ValueError("--alg must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported --alg: {', '.join(invalid)}; allowed: auto,all,{','.join(sorted(allowed))}"
        )
    return names


def _expand_algs(alg_names, op, dtype, exclude_tle=False):
    expanded = []
    for alg in alg_names:
        if alg == "all":
            expanded.extend(fs.list_spmm_csr_algorithms(op=op, dtype=dtype))
        elif alg == "auto":
            expanded.append("auto")
        else:
            expanded.append(alg)
    if exclude_tle:
        expanded = [
            alg
            for alg in expanded
            if alg not in TLE_CSR_SPMM_ALGORITHMS
            and fs.resolve_spmm_csr_algorithm(alg, op, dtype).name
            not in TLE_CSR_SPMM_ALGORITHMS
        ]
    deduped = []
    for alg in expanded:
        if alg not in deduped:
            deduped.append(alg)
    return [a for a in deduped if a != "auto" or "csr_base" not in deduped]


def _selected_tle_algs(alg_names):
    selected = set()
    for alg in alg_names:
        if alg == "all":
            selected.update(("alpha_alg1_tle_opt", "alpha_alg1_tle_opt2"))
        elif alg in ("alpha_alg1_tle_opt", "alpha_alg1_tle_opt2"):
            selected.add(alg)
    return tuple(
        name
        for name in ("alpha_alg1_tle_opt", "alpha_alg1_tle_opt2")
        if name in selected
    )


def _print_tle_availability(alg_names):
    selected = _selected_tle_algs(alg_names)
    if not selected:
        return
    checks = {
        "alpha_alg1_tle_opt": (
            fs.is_alpha_spmm_alg1_tle_opt_available,
            fs.alpha_spmm_alg1_tle_opt_unavailable_reason,
        ),
        "alpha_alg1_tle_opt2": (
            fs.is_alpha_spmm_alg1_tle_opt2_available,
            fs.alpha_spmm_alg1_tle_opt2_unavailable_reason,
        ),
    }
    print("TLE runtime availability:")
    for name in selected:
        available_fn, reason_fn = checks[name]
        available = bool(available_fn())
        print(f"{name}: {'available' if available else 'unavailable'}")
        if not available:
            print(f"  reason: {reason_fn()}")


def _cuda_event_benchmark(op, warmup, iters):
    out = op()  # Mandatory cold/JIT call stays outside events, even with warmup=0.
    for _ in range(max(0, int(warmup))):
        out = op()
    ACCEL.synchronize()
    start = ACCEL.Event(enable_timing=True)
    end = ACCEL.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, int(iters))):
        out = op()
    end.record()
    ACCEL.synchronize()
    return out, start.elapsed_time(end) / max(1, int(iters))


def _cupy_event_benchmark(op, warmup, iters):
    import cupy as cp

    out = op()
    for _ in range(max(0, int(warmup))):
        out = op()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(max(1, int(iters))):
        out = op()
    end.record()
    end.synchronize()
    return out, cp.cuda.get_elapsed_time(start, end) / max(1, int(iters))


def _time_route(
    prepared, B, alg, warmup, iters, timing=False, diagnose=False, layout="row"
):
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmm_csr_run(prepared, B, alg=alg, dense_layout=layout),
        warmup,
        iters,
    )
    _, meta = fs.flagsparse_spmm_csr_run(
        prepared,
        B,
        alg=alg,
        dense_layout=layout,
        return_meta=True,
        timing=bool(timing),
        diagnostics=bool(diagnose),
    )
    process_cpu_ms = float(meta.get("process_cpu_ms", 0.0) or 0.0)
    # The CSV represents the benchmark mean, not the extra metadata call.
    meta["gpu_ms"] = gpu_ms
    meta["operator_ms"] = process_cpu_ms + gpu_ms
    meta["measurement"] = "mean_complete_run_without_phase_events"
    row = {
        "alg": meta.get("alg", alg),
        "ms": process_cpu_ms + gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": None,
        "compute_ms": None,
        "dense_layout": meta.get("dense_layout", layout),
        "b_stride": meta.get("b_stride"),
        "c_stride": meta.get("c_stride"),
        "output_layout": meta.get("output_layout"),
        "diagnostics": meta.get("diagnostics", {}),
        "out": out,
        "metadata": json.dumps(meta, default=str, sort_keys=True),
    }
    if timing:
        row["process_gpu_ms"] = meta.get("process_gpu_ms")
        row["compute_ms"] = meta.get("compute_ms")
        if row["compute_ms"] is None and row["alg"] == "csr_base":
            row["compute_ms"] = gpu_ms
        if row["process_gpu_ms"] is None and row["alg"] == "csr_base":
            row["process_gpu_ms"] = 0.0
    return row


def _skip_row(
    path,
    dtype,
    index_dtype_name,
    op,
    layout,
    alg,
    shape,
    nnz,
    dense_cols,
    b_stride,
    torch_ms,
    cusparse_ms,
    reason,
    timing,
    cusparse_reason="",
):
    n_rows, n_cols = shape
    row = {
        "matrix": os.path.basename(path),
        "dtype": _dtype_name(dtype),
        "index_dtype": index_dtype_name,
        "op": op,
        "layout": layout,
        "alg": alg,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(nnz),
        "dense_cols": dense_cols,
        "b_stride": b_stride,
        "c_stride": "",
        "ms": None,
        "gpu_ms": None,
        "process_cpu_ms": None,
        "torch_ms": torch_ms,
        "cusparse_ms": cusparse_ms,
        "torch_vs_alg_speedup": None,
        "cusparse_vs_alg_speedup": None,
        "err_vs_torch": None,
        "err_vs_cusparse": None,
        "status": "SKIP",
        "reason": reason,
        "cusparse_reason": cusparse_reason or "",
    }
    if timing:
        row["process_gpu_ms"] = None
        row["compute_ms"] = None
    return row


def _vendor_label():
    return fs_common._expected_vendor_sparse_label()


def _vendor_column_label():
    return fs_common._expected_vendor_sparse_short().lower()


def _time_vendor_sparse_ref(
    data, indices, indptr, shape, B, op, warmup, iters, layout="row"
):
    vendor = fs_common._expected_vendor_sparse_backend()
    if vendor == "hipsparse":
        try:
            sparse_ref = spmm_ops._benchmark_spmm_csr_sparse_ref(
                data,
                indices,
                indptr,
                B,
                shape,
                warmup=warmup,
                iters=iters,
                op=op,
                dense_layout=layout,
            )
        except Exception as exc:
            raise RuntimeError(f"hipSPARSE setup/run failed: {exc}") from exc
        if sparse_ref["backend"] is None:
            return None, None, sparse_ref["reason"]
        return sparse_ref["values"], sparse_ref["ms"], sparse_ref.get("reason")
    if vendor != "cupy_cusparse":
        return (
            None,
            None,
            f"{fs_common._sparse_backend_label(vendor)} CSR SpMM baseline is not wired for this runner",
        )

    if data.dtype not in CUSPARSE_DTYPES:
        return None, None, "dtype not supported by CuPy/cuSPARSE reference"
    if op != "non":
        return (
            None,
            None,
            f"CuPy/cuSPARSE CSR SpMM baseline supports op=non only in this runner; op={op} is unsupported",
        )
    if layout != "row":
        return (
            None,
            None,
            f"CuPy/cuSPARSE CSR SpMM baseline supports row-major dense RHS only in this runner; layout={layout} is unsupported",
        )
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
    except Exception as exc:
        return None, None, f"CuPy/cuSPARSE unavailable: {exc}"
    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
        indices_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indices.to(torch.int64))
        )
        indptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
        A = cpx_sparse.csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
        out_cp, ms = _cupy_event_benchmark(lambda: A @ B_cp, warmup, iters)
        out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
        return out, ms, None
    except Exception as exc:
        raise RuntimeError(f"CuPy/cuSPARSE setup/run failed: {exc}") from exc


def run_one_case(
    path,
    dtype,
    index_dtype_name,
    index_dtype,
    op,
    layout,
    alg_names,
    dense_cols,
    warmup,
    iters,
    run_cusparse,
    timing,
    diagnose,
    exclude_tle=False,
    indptr_dtype_name=None,
    emit=None,
):
    device = accelerator_device()
    data, indices, indptr, shape = (_synthetic_case(path, dtype, device)
        if path.startswith("synthetic:") else load_mtx_to_csr_torch(path, dtype=dtype, device=device))
    indices = indices.to(index_dtype)
    indptr_dtype_name = indptr_dtype_name or index_dtype_name
    indptr = indptr.to(INDEX_DTYPE_MAP[indptr_dtype_name])
    n_rows, n_cols = shape
    b_rows = n_rows if op in ("trans", "conj") else n_cols
    B = _materialize_dense_layout_for_test(
        _build_dense_matrix(b_rows, dense_cols, dtype, device),
        layout,
    )
    b_stride = _stride_string(B)
    ref, reference_name = _correctness_reference(
        data, indices, indptr, shape, B, op=op
    )
    # Timing a correctness-only fallback format is not a CSR performance baseline.
    torch_ms = None
    cusparse_out = None
    cusparse_ms = None
    cusparse_reason = "disabled by --no-vendor" if not run_cusparse else ""
    if run_cusparse:
        cusparse_out, cusparse_ms, cusparse_reason = _time_vendor_sparse_ref(
            data, indices, indptr, shape, B, op, warmup, iters, layout=layout
        )

    vendor_raw_ms = cusparse_ms
    vendor_profile = _error_profile(cusparse_out, ref, dtype)
    if cusparse_out is not None and vendor_profile["status"] != "PASS":
        cusparse_reason = f"vendor correctness failed: {vendor_profile}"
        # Preserve a completed vendor measurement even when validation fails.
        # Availability, correctness and eligibility for speedup are separate.
    if cusparse_out is None and not cusparse_reason:
        cusparse_reason = "vendor interface returned no output"
    _print_vendor_result(path, dtype, index_dtype_name, indptr_dtype_name, op, layout,
                         dense_cols, vendor_profile, cusparse_ms, vendor_raw_ms, cusparse_reason)
    def record(row):
        row["indptr_dtype"] = indptr_dtype_name
        row["correctness_ref"] = f"{reference_name} (correctness only)"
        row["vendor_ms"] = cusparse_ms
        row["vendor_raw_ms"] = vendor_raw_ms
        row["vendor_backend"] = fs_common._expected_vendor_sparse_backend()
        row["vendor_alg"] = "default" if cusparse_out is not None else None
        row["vendor_reason"] = cusparse_reason or ""
        row["vendor_status"] = vendor_profile["status"]
        row["vendor_error"] = vendor_profile["global_err"]
        row["speedup_vs_vendor"] = (
            row.get("cusparse_vs_alg_speedup")
            if row["status"] == "PASS" and vendor_profile["status"] == "PASS" else None
        )
        if emit:
            emit(row)
    rows = _StreamingRows(record)
    diag_rows = []
    prepared = fs.prepare_spmm_csr_route(
        data, indices, indptr, shape, op=op, alg="auto"
    )
    selected = _expand_algs(alg_names, op, dtype, exclude_tle=exclude_tle)
    for alg in selected:
        try:
            try:
                resolved = fs.resolve_spmm_csr_algorithm(alg, op, dtype)
            except (ValueError, TypeError) as exc:
                raise fs.SpmmCsrAlgorithmUnavailable(str(exc)) from exc
            if layout not in resolved.supported_layouts:
                raise fs.SpmmCsrAlgorithmUnavailable(f"unsupported layout {layout}")
            result = _time_route(
                prepared,
                B,
                alg,
                warmup,
                iters,
                timing=timing,
                diagnose=diagnose,
                layout=layout,
            )
            out = result.pop("out")
            diagnostics = result.pop("diagnostics")
            torch_profile = _error_profile(out, ref, dtype)
            cusparse_profile = _error_profile(out, cusparse_out, dtype)
        except Exception as exc:
            unavailable = isinstance(exc, fs.SpmmCsrAlgorithmUnavailable)
            failure = _skip_row(
                    path,
                    dtype,
                    index_dtype_name,
                    op,
                    layout,
                    alg,
                    shape,
                    data.numel(),
                    dense_cols,
                    b_stride,
                    torch_ms,
                    cusparse_ms,
                    f"{type(exc).__name__}: {exc}",
                    timing,
                    cusparse_reason=cusparse_reason,
                )
            failure["status"] = "SKIP" if unavailable else "ERROR"
            rows.append(failure)
            if not unavailable:
                raise
            continue
        row = {
            "matrix": os.path.basename(path),
            "dtype": _dtype_name(dtype),
            "index_dtype": index_dtype_name,
            "op": op,
            "layout": layout,
            "alg": result["alg"],
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": int(data.numel()),
            "dense_cols": dense_cols,
            "b_stride": _stride_string(B),
            "c_stride": _stride_string(out),
            "ms": result["ms"],
            "gpu_ms": result["gpu_ms"],
            "process_cpu_ms": result["process_cpu_ms"],
            "torch_ms": torch_ms,
            "cusparse_ms": cusparse_ms,
            "torch_vs_alg_speedup": _ratio(torch_ms, result["ms"]) if torch_profile["status"] == "PASS" else None,
            "cusparse_vs_alg_speedup": (
                _ratio(cusparse_ms, result["ms"])
                if torch_profile["status"] == "PASS" and vendor_profile["status"] == "PASS" else None
            ),
            "err_vs_torch": torch_profile["global_err"],
            "err_vs_cusparse": cusparse_profile["global_err"],
            "status": torch_profile["status"],
            "reason": "" if torch_profile["status"] == "PASS" else "correctness check failed",
            "max_abs_error": torch_profile["max_abs_error"],
            "metadata": result["metadata"],
            "cusparse_reason": cusparse_reason or "",
        }
        if timing:
            row["process_gpu_ms"] = result["process_gpu_ms"]
            row["compute_ms"] = result["compute_ms"]
        rows.append(row)
        if diagnose:
            diag = {
                "matrix": os.path.basename(path),
                "dtype": _dtype_name(dtype),
                "index_dtype": index_dtype_name,
                "op": op,
                "layout": layout,
                "alg": result["alg"],
            }
            diag.update(indptr_dtype=indptr_dtype_name, dense_cols=dense_cols)
            for field in DIAG_FIELDS:
                if field not in diag:
                    value = diagnostics.get(field)
                    diag[field] = json.dumps(value, default=str, sort_keys=True) if isinstance(value, (dict, list)) else value
            diag_rows.append(diag)
    return rows, diag_rows


def _best_rows(rows):
    groups = {}
    for row in rows:
        if row.get("status") != "PASS" or row.get("ms") is None:
            continue
        key = (
            row["matrix"],
            row["dtype"],
            row["index_dtype"],
            row["op"],
            row["layout"],
            row.get("indptr_dtype"), row["dense_cols"],
        )
        groups.setdefault(key, []).append(row)
    best = []
    for (matrix, dtype, index_dtype, op, layout, indptr_dtype, dense_cols), group in sorted(groups.items()):
        selected = min(group, key=lambda item: item["ms"])
        best.append(
            {
                "matrix": matrix,
                "dtype": dtype,
                "index_dtype": index_dtype,
                "op": op,
                "layout": layout,
                "indptr_dtype": indptr_dtype, "dense_cols": dense_cols,
                "best_alg": selected["alg"],
                "best_ms": selected["ms"],
                "best_gpu_ms": selected["gpu_ms"],
                "best_torch_speedup": selected["torch_vs_alg_speedup"],
                "best_cusparse_speedup": selected["cusparse_vs_alg_speedup"],
            }
        )
    return best


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _console_columns(timing=False):
    # Keep the terminal compact even with --timing; detailed fields stay in CSV.
    return [
        ("matrix", "Matrix", 25), ("dtype", "DType", 10),
        ("indices", "Idx/Ptr", 11), ("op", "Op", 5),
        ("layout", "Lay", 3), ("dense_cols", "N", 5),
        ("alg", "Alg", 24), ("ms", "ms", 9),
        ("vendor_ms", "Vendor_ms", 9), ("speedup_vs_vendor", "x", 7),
        ("status", "Check", 6), ("vendor_status", "VCheck", 6),
    ]


def _print_header(timing=False):
    print(" ".join(f"{label:<{width}}" for _, label, width in _console_columns(timing)), flush=True)


def _print_row(row, timing=False):
    cells = []
    for key, _, width in _console_columns(timing):
        value = row.get(key)
        if key == "indices":
            value = f"{row.get('index_dtype', 'N/A')}/{row.get('indptr_dtype', 'N/A')}"
        elif key in ("ms", "vendor_ms", "speedup_vs_vendor"):
            value = _fmt(value, 2 if key == "speedup_vs_vendor" else 4)
        elif value is None:
            value = "N/A"
        cells.append(f"{str(value):<{width}}")
    print(" ".join(cells), flush=True)


_PRINTED_VENDOR_UNAVAILABLE = set()


def _print_vendor_result(path, dtype, index_dtype, indptr_dtype, op, layout, n,
                         profile, ms, raw_ms, reason):
    """Normal performance/accuracy status belongs in the table, not extra lines."""
    if profile["status"] != "SKIP" or reason == "disabled by --no-vendor":
        return
    label = _vendor_label()
    key = (label, str(dtype), index_dtype, indptr_dtype, op, layout, reason)
    if key not in _PRINTED_VENDOR_UNAVAILABLE:
        _PRINTED_VENDOR_UNAVAILABLE.add(key)
        print(f"UNAVAILABLE {label} [{_dtype_name(dtype)} {index_dtype}/{indptr_dtype} {op}/{layout}]: "
              f"{reason}", file=sys.stderr, flush=True)



class _StreamingRows(list):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def append(self, row):
        super().append(row)
        self.callback(row)


def _synthetic_case(name, dtype, device):
    # Repeated columns deliberately exercise duplicate entries and cancellation.
    if name == "synthetic:empty":
        lengths, columns = [0, 0, 0], 7
    elif name == "synthetic:tail":
        lengths, columns = [0, 1, 32, 33, 2048, 2049, 65537, 3, 0], 71
    else:
        lengths, columns = [0, 1, 7, 8, 16, 32, 33, 4, 2], 41
    ptr = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int64, device=device)
    size = sum(lengths)
    idx = (torch.arange(size, device=device, dtype=torch.int64) * 17 + 5) % columns
    values = _build_dense_matrix(size, 1, dtype, device).reshape(-1) * 0.125
    return values, idx, ptr, (len(lengths), columns)


def main():
    parser = argparse.ArgumentParser(description="Native CSR SpMM registered algorithm benchmark")
    parser.add_argument("input", nargs="*", help="MatrixMarket files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--alg", default="auto", help="auto, all/compare, or comma-separated registered names")
    parser.add_argument("--dtypes", "--dtype", dest="dtype", default="float32,float64")
    parser.add_argument("--ops", "--op", dest="op", default="all")
    parser.add_argument("--index-dtypes", "--index-dtype", dest="index_dtype", default="all")
    parser.add_argument("--indptr-dtypes", default=None, help="default: match column index dtype")
    parser.add_argument("--layout", default="row")
    parser.add_argument("--dense-cols", default="32", help="positive integer or comma-separated list")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-csr", "--csv", dest="csv")
    parser.add_argument("--no-vendor", "--no-cusparse", "--no-hipsparse", dest="no_cusparse", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--exclude-tle", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    try:
        dtype_names = _parse_csv_names(args.dtype, DEFAULT_DTYPE_NAMES, "--dtypes", tuple(DTYPE_MAP))
        index_names = _parse_csv_names(args.index_dtype, DEFAULT_INDEX_DTYPE_NAMES, "--index-dtypes")
        ptr_names = None if args.indptr_dtypes is None else _parse_csv_names(args.indptr_dtypes, DEFAULT_INDEX_DTYPE_NAMES, "--indptr-dtypes")
        ops = _parse_csv_names(args.op, DEFAULT_OP_NAMES, "--ops")
        layouts = _layout_names(args.layout)
        algs = _parse_algs(args.alg)
        widths = list(dict.fromkeys(int(v) for v in args.dense_cols.split(",")))
        if not widths or min(widths) <= 0 or args.warmup < 0 or args.iters <= 0:
            raise ValueError("dense-cols and iters must be positive; warmup must be nonnegative")
    except ValueError as exc:
        parser.error(str(exc))
    paths = _resolve_input_paths(args.input)
    if args.synthetic:
        paths += ["synthetic:short", "synthetic:tail", "synthetic:empty"]
    if not paths:
        parser.error("no matrices selected; supply files/directories or --synthetic")
    if not ACCEL.is_available():
        raise RuntimeError("selected accelerator is unavailable")
    torch.manual_seed(args.seed)
    import triton
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    try:
        device_name = str(ACCEL.get_device_name())
    except (AttributeError, RuntimeError):
        device_name = str(accelerator_device())
    runtime_info = dict(device=device_name, backend=fs_common._backend_name(),
                        torch_version=torch.__version__, triton_version=triton.__version__,
                        hip_version=getattr(torch.version, "hip", None), cuda_version=getattr(torch.version, "cuda", None),
                        python_version=platform.python_version(), commit=commit,
                        warmup=args.warmup, iters=args.iters)
    vendor_label = "disabled" if args.no_cusparse else _vendor_label()
    print(f"{runtime_info['backend']} | {device_name} | Vendor={vendor_label} | times=ms, x=Vendor/ms",
          flush=True)
    _print_header(timing=args.timing)
    fields = PERF_FIELDS + (TIMING_FIELDS if args.timing else [])
    rows, diag_rows = [], []
    csv_path = _normalize_csv_path(args.csv) if args.csv else None
    handle = open(csv_path, "w", newline="", encoding="utf-8") if csv_path else None
    writer = csv.DictWriter(handle, fieldnames=fields) if handle else None
    if writer:
        writer.writeheader()
        handle.flush()
    reported_unavailable = set()
    def emit(row):
        metadata = json.loads(row.get("metadata") or "{}")
        metadata["environment"] = runtime_info
        row["metadata"] = json.dumps(metadata, default=str, sort_keys=True)
        rows.append(row)
        _print_row(row, timing=args.timing)
        if row["status"] == "SKIP" and row.get("reason") not in reported_unavailable:
            reported_unavailable.add(row["reason"])
            print(f"UNAVAILABLE {row['alg']}: {row['reason']}", file=sys.stderr, flush=True)
        if writer:
            writer.writerow(row)
            handle.flush()
        if args.fail_fast and row["status"] == "FAIL":
            raise SystemExit("--fail-fast: failed CSR SpMM result was saved")
    try:
        for dtype_name, index_name, op, layout, width, path in itertools.product(
                dtype_names, index_names, ops, layouts, widths, paths):
            for ptr_name in ptr_names or [index_name]:
                try:
                    _, diagnostics = run_one_case(path, DTYPE_MAP[dtype_name], index_name,
                        INDEX_DTYPE_MAP[index_name], op, layout, algs, width, args.warmup,
                        args.iters, not args.no_cusparse, args.timing, args.diagnose,
                        exclude_tle=args.exclude_tle, indptr_dtype_name=ptr_name, emit=emit)
                    diag_rows.extend(diagnostics)
                except Exception:
                    print(f"ERROR matrix={path} dtype={dtype_name} indices={index_name}/{ptr_name} "
                          f"op={op} layout={layout} N={width}", file=sys.stderr, flush=True)
                    raise
    finally:
        if handle:
            handle.close()
    if csv_path:
        root, ext = os.path.splitext(csv_path)
        _write_csv(f"{root}.best{ext}", _best_rows(rows), BEST_FIELDS)
        if args.diagnose:
            _write_csv(f"{root}.diagnose{ext}", diag_rows, DIAG_FIELDS)
        print(f"Wrote {len(rows)} rows to {csv_path}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
