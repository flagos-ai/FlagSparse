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

"""Accelerator selection shared by the standalone benchmark CLI scripts."""

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flagsparse.sparse_operations._common import _ACCEL, _ACCEL_DEVICE_TYPE


ACCEL = _ACCEL


def _write_benchmark_state(path, payload):
    """Publish a complete snapshot; readers must never see partial JSON."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, default=str, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def accelerator_available() -> bool:
    return bool(ACCEL.is_available())


def accelerator_device() -> torch.device:
    return torch.device(_ACCEL_DEVICE_TYPE)


def csr_tile_structure(indptr, *, tiles=(128, 256, 512, 1024), config=None):
    """CPU-only execution-CSR geometry; never kernel-consumed preparation.

    Counts model logical updates, not hardware transactions. Chunking bounds
    temporary memory even for very large input matrices.
    """
    import numpy as np

    p = np.asarray(indptr, dtype=np.int64)
    if p.ndim != 1 or not len(p) or p[0] != 0 or np.any(p[1:] < p[:-1]):
        raise ValueError("expected a monotone, zero-based CPU CSR indptr")
    nnz, rows = int(p[-1]), len(p) - 1
    lengths = np.diff(p)
    nonempty = lengths > 0
    result = dict(model="csr_tile_geometry_v2", source="cpu_execution_csr", rows=rows, nnz=nnz, tiles={})
    result["row_lengths"] = dict(empty_rows=int(np.count_nonzero(~nonempty)),
                                 mean=nnz / rows if rows else 0.0,
                                 maximum=int(lengths.max()) if rows else 0,
                                 p50=float(np.percentile(lengths, 50)) if rows else None,
                                 p90=float(np.percentile(lengths, 90)) if rows else None,
                                 p99=float(np.percentile(lengths, 99)) if rows else None)
    result["long_row_nnz_fraction"] = {
        str(limit): int(lengths[lengths > limit].sum()) / nnz if nnz else None
        for limit in (128, 256, 512, 1024, 4096)}
    result["atomic_count_basis"] = "per_real_component_logical_updates"
    result["scalar_groups"] = {}
    for width in (32, 64, 128):
        full = rows // width * width
        denominator = int(lengths[:full].reshape(-1, width).max(axis=1).sum()) * width if full else 0
        if full != rows:
            denominator += int(lengths[full:].max()) * (rows - full)
        result["scalar_groups"][str(width)] = dict(
            useful_positions=nnz, scheduled_positions=denominator,
            utilization=nnz / denominator if denominator else None,
            ratio_reason=None if denominator else "no_nonzeros",
            interpretation="logical_position_utilization_not_hardware_occupancy")

    for tile in tiles:
        if type(tile) is not int or tile <= 0:
            raise ValueError("tile sizes must be positive integers")
        count = (nnz + tile - 1) // tile
        homogeneous = 0
        # Histogram avoids retaining one span per NNZ tile.
        histogram = {}
        for first in range(0, count, 65536):
            starts = np.arange(first, min(count, first + 65536), dtype=np.int64) * tile
            ends = np.minimum(starts + tile, nnz) - 1
            lo = np.searchsorted(p, starts, side="right") - 1
            hi = np.searchsorted(p, ends, side="right") - 1
            homogeneous += int(np.count_nonzero(lo == hi))
            spans, frequencies = np.unique((hi - lo + 1)[hi != lo], return_counts=True)
            for span, frequency in zip(spans, frequencies):
                histogram[int(span)] = histogram.get(int(span), 0) + int(frequency)
        mixed = count - homogeneous
        def percentile(q):
            if not mixed:
                return None
            rank, cumulative = max(1, int(np.ceil(q * mixed))), 0
            for span, frequency in sorted(histogram.items()):
                cumulative += frequency
                if cumulative >= rank:
                    return span
        parts = (p[1:][nonempty] - 1) // tile - p[:-1][nonempty] // tile + 1
        result["tiles"][str(tile)] = dict(
            tile_count=count, single_owner_tile_count=homogeneous,
            single_owner_tile_ratio=homogeneous / count if count else None,
            ratio_reason=None if count else "no_nonzeros", mixed_tile_count=mixed,
            mixed_tile_row_span_p50=percentile(.5), mixed_tile_row_span_p90=percentile(.9),
            mixed_tile_row_span_max=max(histogram, default=None),
            full_row_count=int(np.count_nonzero(parts == 1)),
            crossing_row_count=int(np.count_nonzero(parts > 1)),
            predicted_boundary_atomic_updates=int(parts[parts > 1].sum()))
    cfg = config or {}
    if "long_row_threshold" in cfg and "segment_nnz" in cfg:
        threshold, segment = cfg["long_row_threshold"], cfg["segment_nnz"]
        if threshold <= 0 or segment <= 0:
            raise ValueError("queue threshold and segment must be positive")
        long_lengths = lengths[lengths > threshold]
        capacity = min(rows, nnz // (threshold + 1))
        jobs = nnz // segment + capacity if capacity else 0
        expected = int(((long_lengths + segment - 1) // segment).sum())
        result["queue"] = dict(expected_long_rows=int(long_lengths.size), expected_jobs=expected,
                               job_capacity=jobs, expected_active_job_ratio=expected / jobs if jobs else None,
                               ratio_reason=None if jobs else "zero_capacity")
    if {"short_row_threshold", "producer_num_warps", "rows_per_program"} <= cfg.keys():
        threshold = cfg["short_row_threshold"]
        if type(threshold) is not int or threshold < 0:
            raise ValueError("row-queue threshold must be a nonnegative integer")
        queued = lengths > threshold
        capacity = min(rows, nnz // (threshold + 1))
        queued_rows = int(np.count_nonzero(queued))
        queued_nnz = int(lengths[queued].sum())
        result["row_queue"] = dict(short_row_threshold=threshold,
            expected_queued_rows=queued_rows, expected_queued_nnz=queued_nnz,
            queued_nnz_fraction=queued_nnz / nnz if nnz else None,
            row_queue_capacity=capacity,
            expected_active_program_ratio=queued_rows / capacity if capacity else None,
            ratio_reason=None if capacity else "zero_capacity",
            count_source="cpu_structure_model_not_gpu_readback")
        if "grid_blocks" in cfg:
            limit = cfg["grid_blocks"]
            if type(limit) is not int or not 1 <= limit <= 1024:
                raise ValueError("row-queue grid_blocks must be an integer between 1 and 1024")
            grid = min(capacity, limit)
            result["row_queue"].update(
                consumer_grid_limit=limit, consumer_grid_blocks=grid,
                expected_active_programs=min(queued_rows, grid),
                expected_active_program_ratio=min(queued_rows, grid) / grid if grid else None,
                expected_tasks_per_program=queued_rows / grid if grid else None,
                expected_max_rounds=(queued_rows + grid - 1) // grid if grid else 0,
                consumer_rounds_capacity=(capacity + grid - 1) // grid if grid else 0,
            )
    if "cooperative_parts" in cfg:
        threshold, parts = cfg["long_row_threshold"], cfg["cooperative_parts"]
        block, limit = cfg["block_nnz"], cfg["grid_blocks"]
        if (any(type(v) is not int for v in (threshold, parts, block, limit)) or threshold < 0
                or parts not in (1, 2, 4, 8, 16) or block <= 0 or not 1 <= limit <= 1024):
            raise ValueError("invalid cooperative structure configuration")
        long_lengths = lengths[lengths > threshold]
        queued = int(long_lengths.size)
        capacity = min(rows, nnz // (threshold + 1))
        grid = min(capacity, limit)
        chunks = (long_lengths + block - 1) // block
        useful_parts = int(np.minimum(chunks, parts).sum())
        result["cooperative_queue"] = dict(
            long_row_threshold=threshold, cooperative_parts=parts,
            expected_queued_rows=queued, expected_queued_nnz=int(long_lengths.sum()),
            queued_nnz_fraction=int(long_lengths.sum()) / nnz if nnz else None,
            row_queue_capacity=capacity, partial_slot_capacity=capacity * parts,
            row_grid_blocks=grid, consumer_grid_blocks=grid * parts,
            expected_partial_slots=queued * parts, expected_nonempty_part_tasks=useful_parts,
            expected_empty_part_tasks=queued * parts - useful_parts,
            expected_max_rounds=(queued + grid - 1) // grid if grid else 0,
            consumer_rounds_capacity=(capacity + grid - 1) // grid if grid else 0,
            # Queue insertion order depends on GPU scheduling; do not invent
            # worker makespan or occupancy from the CPU row order.
            count_source="cpu_structure_model_not_gpu_readback")
    return result


def coo_run_structure(owners, *, tiles=(32, 64, 128, 256, 512, 1024)):
    """Classify original output-owner order on CPU, with bounded tile batches.

    The single-owner class has priority, including a one-element tail. No-adjacent
    tiles may still repeat an owner non-contiguously. No input sorting is allowed.
    """
    import numpy as np

    owners = np.asarray(owners, dtype=np.int64)
    if owners.ndim != 1:
        raise ValueError("expected nonnegative one-dimensional COO output owners")
    nnz = len(owners)
    for start in range(0, nnz, 2**20):
        if np.any(owners[start:start + 2**20] < 0):
            raise ValueError("expected nonnegative one-dimensional COO output owners")
    result = dict(model="coo_run_geometry_v1", source="cpu_original_output_owner_order",
                  nnz=nnz, tiles={})
    for tile in tiles:
        if type(tile) is not int or tile <= 0 or tile > 2**20:
            raise ValueError("tile must be an integer in [1, 2**20]")
        counts = dict(single_owner=0, no_adjacent_equal=0, segmented=0)
        weights = dict.fromkeys(counts, 0)
        chunk = max(1, 2**20 // tile) * tile
        full = nnz // tile * tile
        for start in range(0, full, chunk):
            group = owners[start:min(full, start + chunk)].reshape(-1, tile)
            single = np.all(group == group[:, :1], axis=1)
            no_adjacent = ~single & ~np.any(group[:, 1:] == group[:, :-1], axis=1)
            for name, count in (("single_owner", np.count_nonzero(single)),
                                ("no_adjacent_equal", np.count_nonzero(no_adjacent)),
                                ("segmented", np.count_nonzero(~single & ~no_adjacent))):
                counts[name] += int(count)
                weights[name] += int(count) * tile
        tail = owners[full:]
        if tail.size:
            name = ("single_owner" if np.all(tail == tail[0]) else
                    "no_adjacent_equal" if not np.any(tail[1:] == tail[:-1]) else "segmented")
            counts[name] += 1
            weights[name] += int(tail.size)
        total = (nnz + tile - 1) // tile
        result["tiles"][str(tile)] = dict(
            tile_count=total, counts=counts, nnz_by_class=weights,
            tile_fraction={key: value / total if total else None for key, value in counts.items()},
            nnz_fraction={key: value / nnz if nnz else None for key, value in weights.items()},
            ratio_reason=None if nnz else "no_nonzeros")
    return result


def csc_column_structure(indptr, indices, n_rows, *, op="non"):
    """CPU-only column lengths and non-scatter row contribution distribution."""
    import numpy as np

    p = np.asarray(indptr, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)
    if (p.ndim != 1 or not p.size or p[0] != 0 or np.any(p[1:] < p[:-1])
            or p[-1] != idx.size or idx.ndim != 1 or n_rows < 0
            or np.any(idx < 0) or np.any(idx >= n_rows)):
        raise ValueError("invalid CPU CSC structure")
    def distribution(lengths):
        total = int(lengths.sum())
        return dict(count=int(lengths.size), empty=int(np.count_nonzero(lengths == 0)),
                    maximum=int(lengths.max()) if lengths.size else 0,
                    percentiles={str(q): float(np.percentile(lengths, q)) if lengths.size else None
                                 for q in (50, 90, 99)},
                    long_nnz_fraction={str(limit): int(lengths[lengths > limit].sum()) / total if total else None
                                       for limit in (32, 128, 256, 1024, 4096)})
    result = dict(model="csc_column_geometry_v1", source="cpu_original_csc", op=op,
                  nnz=int(idx.size), columns=distribution(np.diff(p)))
    if op == "non":
        result["output_row_contributions"] = distribution(np.bincount(idx, minlength=n_rows))
    return result


def add_type_combo_arguments(parser, *, spmm=False):
    """Extend the existing CLIs without changing their legacy defaults."""
    parser.add_argument("--type-combos", help="A:RHS:output triples, comma-separated")
    if spmm:
        parser.add_argument("--op-b", default="non", help="non,trans,conj or all (type-combos)")
    parser.add_argument("--vendor-diagnostic", default=None, help="CSC SpMM: ones_fanin,rectangular")
    parser.add_argument("--diagnostic-lengths", default="128,256,512,1024,2048,4096")
    parser.add_argument("--vendor-alg", default="default", help="CUDA SpMM: default,csr_alg1,csr_alg2,csr_alg3")
    parser.add_argument("--vendor-preprocess", default=None, help="off,on; omit to preserve backend defaults")
    for option, kwargs in (
        ("--indptr-dtypes", dict(default=None)),
        ("--config", dict(default=None)),
        ("--fail-fast", dict(action="store_true")),
        ("--synthetic", dict(action="store_true")),
        ("--vendor-timeout", dict(type=float, default=180.0)),
        ("--reference-bandwidth-gbs", dict(type=float, default=None)),
        ("--reference-bandwidth-source", dict(default=None)),
        ("--reference-peak-gflops", dict(type=float, default=None)),
        ("--reference-peak-source", dict(default=None)),
        ("--perf-metrics", dict(action="store_true", help="Useful real-value throughput and modeled logical bandwidth (type combos or regular SpMV CSR; not hardware counters)")),
    ):
        if option not in parser._option_string_actions:
            parser.add_argument(option, **kwargs)
    for alias, candidates in (
        ("--ops", ("--op",)), ("--dtypes", ("--dtype",)),
        ("--index-dtypes", ("--index-dtype",)),
        ("--no-vendor", ("--no-cusparse",)),
    ):
        if alias not in parser._option_string_actions:
            for candidate in candidates:
                if candidate in parser._option_string_actions:
                    action = parser._option_string_actions[candidate]
                    action.option_strings.append(alias)
                    parser._option_string_actions[alias] = action
                    break
    if spmm and "--dense-cols" in parser._option_string_actions:
        # Preserve ints for old commands; explicit combinations may use an N grid.
        def dense_columns(value):
            return value if "," in value else int(value)
        parser._option_string_actions["--dense-cols"].type = dense_columns


def parse_type_combos(value, argv):
    """Pure CLI validation; parser defaults are not explicit dtype requests."""
    if any(token.split("=", 1)[0] in ("--dtype", "--dtypes") for token in argv):
        raise ValueError("--type-combos conflicts with explicit --dtype/--dtypes")
    allowed = {
        (storage, storage, output)
        for storage in ("float16", "bfloat16")
        for output in (storage, "float32")
    }
    combos = tuple(dict.fromkeys(tuple(part.strip() for part in item.split(":"))
                                for item in value.split(",")))
    if not combos or any(combo not in allowed for combo in combos):
        raise ValueError("expected FP16/BF16 matching inputs with same-type or FP32 output")
    return combos


def parse_vendor_options(operator, args):
    """Validate explicit diagnostics before device or baseline setup."""
    def tokens(value, allowed):
        values = tuple(dict.fromkeys(value.split(",")))
        if not values or any(v not in allowed for v in values):
            raise ValueError(f"expected values in {allowed}, got {value!r}")
        return values
    algorithms = tokens(args.vendor_alg, ("default", "csr_alg1", "csr_alg2", "csr_alg3"))
    preprocessing = tuple(v == "on" for v in tokens(args.vendor_preprocess, ("off", "on"))) if args.vendor_preprocess else (None,)
    cases = tokens(args.vendor_diagnostic, ("ones_fanin", "rectangular")) if args.vendor_diagnostic else ()
    if cases and operator != "spmm_csc":
        raise ValueError("--vendor-diagnostic currently requires spmm_csc")
    if (algorithms != ("default",) or preprocessing != (None,)) and operator not in ("spmm_csr", "spmm_csc"):
        raise ValueError("vendor algorithm/preprocess scanning requires CSR/CSC SpMM")
    lengths = tuple(dict.fromkeys(int(x) for x in args.diagnostic_lengths.split(",")))
    if not lengths or any(x < 1 or x > 65536 for x in lengths):
        raise ValueError("diagnostic lengths must be in 1..65536")
    if (cases or algorithms != ("default",) or preprocessing != (None,)) and not args.type_combos:
        raise ValueError("vendor diagnostics/scanning require --type-combos")
    return algorithms, preprocessing, cases, lengths


def run_type_combo_benchmark(operator, args, parser, argv=None):
    """Explicit combinations in the original six CLIs, using registered runs.

    This branch shares prepare, timing and CSV handling across formats. It does
    not call q4/mixed kernels, bind private kernels, or cache extra runtime work.
    Returning False leaves the script's historical delivery path unchanged.
    """
    import csv
    import importlib
    import itertools
    import json
    import traceback

    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        vendor_algs, vendor_preprocessing, diagnostic_cases, diagnostic_lengths = parse_vendor_options(operator, args)
    except ValueError as exc:
        parser.error(str(exc))
    reference_options = dict(reference_gbs=args.reference_bandwidth_gbs,
                             bandwidth_source=args.reference_bandwidth_source,
                             peak_gflops=args.reference_peak_gflops, peak_source=args.reference_peak_source)
    if any(value is not None for value in reference_options.values()) and not args.perf_metrics:
        parser.error("reference bandwidth/peak options require --perf-metrics")
    if not args.type_combos:
        if getattr(args, "perf_metrics", False) and operator != "spmv_csr":
            parser.error("--perf-metrics currently requires --type-combos")
        if getattr(args, "op_b", "non") != "non":
            parser.error("--op-b currently requires --type-combos")
        if isinstance(getattr(args, "dense_cols", None), str):
            parser.error("a --dense-cols list currently requires --type-combos")
        return False
    from flagsparse.sparse_operations._common import _backend_name
    from flagsparse.sparse_operations._spmv_csr_config import BackendCapabilityUnavailable
    from tests import cusparse_generic_baseline as vendor
    from tests import mtx_fast
    from tests.pytest.accuracy_utils import close_tolerances
    from tests.utils import (filtered_avg_ms, paired_timing_summary, sparse_work_metrics,
                             benchmark_sample_summary, sparse_reference_metrics, finite_speedup, tensor_accuracy_metrics)

    def selection(value, allowed):
        result = tuple(allowed) if value == "all" else tuple(dict.fromkeys(str(value).split(",")))
        if not result or any(item not in allowed for item in result):
            raise ValueError(f"expected one or more of {allowed}, got {value!r}")
        return result

    def option(*names, default=None):
        # A singular explicit option takes precedence over an unrelated plural
        # parser default (COO SpMM historically has both).
        for name in names:
            flag = "--" + name.replace("_", "-")
            if any(token.split("=", 1)[0] == flag for token in tokens):
                # Aliases can share a destination with their singular option.
                action = parser._option_string_actions[flag]
                return getattr(args, action.dest)
        return next((getattr(args, name) for name in names if getattr(args, name, None) is not None), default)

    matrix = operator.startswith("spmm")
    fmt = operator.rsplit("_", 1)[1]
    try:
        sparse_reference_metrics({}, **reference_options)  # Validate before device work.
        combos = parse_type_combos(args.type_combos, tokens)
        ops = selection(option("ops", "op", default="non") or "non", ("non", "trans", "conj"))
        op_bs = selection(args.op_b, ("non", "trans", "conj")) if matrix else ("non",)
        layouts = selection(option("layout", default="row"), ("row", "col")) if matrix else ("row",)
        indices = selection(option("index_dtypes", "index_dtype", default="int32"), ("int32", "int64"))
        pointers = selection(args.indptr_dtypes, ("int32", "int64")) if args.indptr_dtypes else None
        ns = tuple(int(n) for n in str(getattr(args, "dense_cols", 1)).split(","))
        if any(n < 0 for n in ns) or args.warmup < 0 or args.iters < 1 or args.vendor_timeout <= 0:
            raise ValueError("require N >= 0, warmup >= 0, iters >= 1 and vendor-timeout > 0")
        config = json.loads(args.config) if isinstance(args.config, str) else args.config
        requested = getattr(args, "alg", None) or "auto"
        if config is not None and (requested in ("auto", "compare", "all") or "," in requested):
            raise ValueError("--config requires one explicit algorithm")
        if getattr(args, "csv_tocsr", None):
            raise ValueError("--type-combos tests native COO; --csv-tocsr is a different operation")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    paths = []
    for item in option("paths", "mtx", "input", default=[]):
        path = Path(item)
        paths.extend(sorted(path.rglob("*.mtx")) if path.is_dir() else [path])
    if not paths and not args.synthetic and not diagnostic_cases:
        parser.error("supply MatrixMarket inputs or --synthetic")
    if not accelerator_available():
        raise RuntimeError("an accelerator is required for type-combination benchmarks")
    module = importlib.import_module("flagsparse.sparse_operations." + operator)
    query = getattr(module, "list_" + operator + "_algorithms")
    prepare = getattr(module, "prepare_" + operator + ("_route" if matrix else ""))
    run = getattr(module, "flagsparse_" + operator + "_run")
    backend, device = _backend_name(), accelerator_device()
    metrics_enabled = getattr(args, "perf_metrics", False)
    environment = {}
    if metrics_enabled:
        import platform
        import triton
        environment = dict(torch_version=str(torch.__version__), triton_version=str(triton.__version__),
                           python_version=platform.python_version(), hip_version=getattr(torch.version, "hip", None),
                           cuda_version=getattr(torch.version, "cuda", None), input_seed=2026)
        import subprocess
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
                                      capture_output=True, text=True, timeout=5)
            state = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_PROJECT_ROOT,
                                   capture_output=True, text=True, timeout=5)
            environment.update(source_commit=revision.stdout.strip() if revision.returncode == 0 else "unknown",
                               source_dirty=bool(state.stdout.strip()) if state.returncode == 0 else None)
        except (OSError, subprocess.TimeoutExpired) as exc:
            environment.update(source_commit="unknown", source_dirty=None, source_info_reason=str(exc))
        try:
            environment["device_name"] = ACCEL.get_device_name(device)
        except (AttributeError, TypeError, RuntimeError) as exc:
            environment.update(device_name=str(device), device_info_reason=str(exc))
    csv_path = option("csv_" + fmt, "csv")
    fields = ["matrix", "dtype", "type_combo", "a_dtype", "rhs_dtype", "compute_dtype", "output_dtype",
              "op", "op_a", "op_b", "layout", "dense_cols", "index_dtype", "indptr_dtype",
              "alg", "alg_requested", "alg_resolved", "backend", "m", "n", "nnz", "status", "reason", "max_error",
              "ms", "gpu_ms", "process_cpu_ms", "process_gpu_ms", "compute_ms", "fused_plan_compute_ms",
              "init_ms", "producer_ms", "consumer_ms", "finish_ms", "scatter_ms", "reduce_ms",
              "vendor_ms", "vendor_backend", "vendor_version", "vendor_alg", "vendor_preprocessed", "vendor_status",
              "vendor_op_a_requested", "vendor_op_b_requested", "vendor_op_a_executed", "vendor_op_b_executed",
              "vendor_operation_mapping", "vendor_reason", "vendor_max_error", "vendor_exit_code", "vendor_error_stage", "vendor_stdout", "vendor_stderr",
              "vendor_output_nonfinite", "source_nonfinite", "source_max_abs", "input_nonfinite", "rhs_nonfinite", "reference_nonfinite", "output_nonfinite", "speedup_vs_vendor", "metadata", "timed", "sample_rule",
              "max_relative_error", "normalized_max_error", "failed_elements", "diagnostic_case",
              "flagsparse_measurement_reused", "vendor_alg_requested", "vendor_preprocess_requested",
              "vendor_prepare_wall_ms", "vendor_workspace_bytes", "vendor_compute_dtype",
              "vendor_max_relative_error", "vendor_normalized_max_error", "vendor_failed_elements", "vendor_reference_nonfinite",
              "vendor_diagnostic_path", "vendor_diagnostics", "vendor_library_name", "vendor_library_path",
              "vendor_library_sha256", "vendor_library_identity_reason", "vendor_algorithm_id",
              "vendor_format", "vendor_a_dtype", "vendor_rhs_dtype", "vendor_output_dtype",
              "vendor_alpha", "vendor_beta", "reference", "rtol", "atol"]
    if operator == "spmv_csr":
        fields += ["row_queue_capacity", "consumer_grid_blocks", "consumer_grid_limit",
                   "consumer_rounds_capacity", "task_acquisition", "cooperative_parts",
                   "row_grid_blocks", "partial_slot_capacity"]
    if metrics_enabled:
        fields += ["metrics_model", "metrics_time_basis", "useful_flops", "sparse_input_bytes",
                   "rhs_gather_bytes", "output_write_bytes", "logical_bytes", "arithmetic_intensity",
                   "useful_gflops", "logical_gbs", "gnnz_s", "gpu_useful_gflops", "gpu_logical_gbs", "metrics_reason",
                   "sample_count", "sample_kept", "sample_min_ms", "sample_median_ms", "sample_max_ms", "sample_cv",
                   "matrix_fingerprint", "rhs_fingerprint", "input_seed", "device_name", "device_info_reason",
                   "torch_version", "triton_version", "python_version", "hip_version", "cuda_version",
                   "source_commit", "source_dirty", "source_info_reason",
                   "prepared_run_ms", "prepared_run_time_basis", "prepared_useful_gflops", "prepared_logical_gbs",
                   "cpu_gpu_overlap_risk", "reference_bandwidth_gbs", "reference_bandwidth_source",
                   "reference_peak_gflops", "reference_peak_source", "logical_bw_ratio", "gpu_logical_bw_ratio",
                   "logical_roofline_gflops", "gpu_logical_roofline_ratio", "csr_structure", "coo_structure", "csc_structure"]
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    stream = open(csv_path, "w", newline="", encoding="utf-8") if csv_path else None
    writer = csv.DictWriter(stream, fields, extrasaction="ignore") if stream else None
    if writer:
        writer.writeheader()
        stream.flush()
    print("Registered type combinations; prepare excluded; full run includes required conversions and reductions.")
    print("ms=process_cpu_ms+gpu_ms; paired total median +/-10% mean; phase diagnostics measured separately.")
    print(f"{'Matrix':24} {'A/RHS/Out':23} {'A/B':9} {'Lay/N':9} {'Algorithm':32} {'ms':>9} {'CPU':>8} {'ProcGPU':>8} {'Compute':>8} {'Vendor':>9} {'V/Alg':>8}"
          + (" Wall GPU-GF/s GPU-LGB/s" if metrics_enabled else "") + " Status")
    if metrics_enabled:
        print("Extra columns: Wall=synchronized prepared run; GPU-GF/s and GPU-LGB/s use full GPU event time (NOT DRAM/HBM bandwidth or pure kernel-active time).")
    reported_vendor_reasons = set()
    vendor_log_dir = Path(str(csv_path) + ".vendor_logs") if csv_path else Path("vendor_logs") / operator
    vendor_case = 0
    checkpoint = None
    progress_path = Path(str(csv_path) + ".progress.json") if csv_path else vendor_log_dir / "progress.json"

    def progress(stage, context):
        import os
        import socket
        from datetime import datetime, timezone

        keys = ("matrix", "type_combo", "op_a", "op_b", "layout", "dense_cols",
                "index_dtype", "indptr_dtype", "alg", "status", "vendor_status", "reason")
        _write_benchmark_state(progress_path, dict(
            operator=operator, stage=stage, pid=os.getpid(), hostname=socket.gethostname(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            **{key: context[key] for key in keys if key in context}))

    def emit(row):
        if writer:
            writer.writerow({k: json.dumps(v, default=str, sort_keys=True) if isinstance(v, dict) else v
                             for k, v in row.items()})
            stream.flush()
        if checkpoint is not None:
            _write_benchmark_state(checkpoint, dict(row, checkpoint_state="completed"))
        progress("case_completed", row)
        def number(key):
            value = row.get(key)
            return "N/A" if value is None else f"{value:.4f}"
        combo = "/".join(row[k].replace("float", "f").replace("bf", "bf") for k in ("a_dtype", "rhs_dtype", "output_dtype"))
        vendor_label = (f" V={row.get('vendor_alg_requested', 'default')}/{row.get('vendor_preprocess_requested', 'auto')}"
                        if len(vendor_algs) * len(vendor_preprocessing) > 1 else "")
        print(f"{Path(row['matrix']).name:24} {combo:23} {row['op_a']+'/'+row['op_b']:9} "
              f"{row['layout']+'/'+str(row['dense_cols']):9} {row['alg']:32} "
              + " ".join(f"{number(k):>8}" for k in ("ms", "process_cpu_ms", "process_gpu_ms", "compute_ms", "vendor_ms", "speedup_vs_vendor"))
              + (f" {number('prepared_run_ms')} {number('gpu_useful_gflops')} {number('gpu_logical_gbs')}" if metrics_enabled else "")
              + f" {row['status']}" + vendor_label)
        if row.get("reason"):
            print(row["reason"])
        reason = row.get("vendor_reason")
        if reason and reason not in reported_vendor_reasons:
            print(f"Vendor {row.get('vendor_status', 'SKIP')}: {reason}")
            if row.get("vendor_status") == "ERROR" and row.get("vendor_stderr"):
                print(f"Vendor log: {row['vendor_stderr']}")
                log = Path(row["vendor_stderr"])
                if log.exists():
                    print(log.read_text(encoding="utf-8", errors="replace")[-8000:])
            reported_vendor_reasons.add(reason)

    try:
        sources = paths + ([None] if args.synthetic else [])
        for case in diagnostic_cases:
            sources += [(case, length) for length in (diagnostic_lengths if case == "ones_fanin" else (0,))]
        for source, combo in itertools.product(sources, combos):
            diagnostic = source[0] if isinstance(source, tuple) else None
            source_name = f"diagnostic_{source[0]}_{source[1]}" if diagnostic else str(source or "synthetic")
            context = dict(matrix=source_name, type_combo=":".join(combo))
            progress("source_load", context)
            dtype, rhs_dtype, output_dtype = (getattr(torch, item) for item in combo)
            # Construct once per matrix/type; quantize before forming the CPU golden.
            if source is None or diagnostic:
                import scipy.sparse as sparse
                if diagnostic == "ones_fanin":
                    length = source[1]
                    host = sparse.coo_matrix(([1.] * length, ([0] * length, list(range(length)))), shape=(1, length)).asformat(fmt)
                else:
                    host = sparse.coo_matrix(([1., .00390625, -.125, .5], ([0, 0, 0, 2], [0, 0, 1, 2])), shape=(3, 4)).asformat(fmt)
                a = torch.tensor(host.data, dtype=dtype)
                first = torch.tensor(host.row if fmt == "coo" else host.indices, dtype=torch.int64)
                second = torch.tensor(host.col if fmt == "coo" else host.indptr, dtype=torch.int64)
                shape = host.shape
            else:
                a, first, second, shape = getattr(mtx_fast, "load_" + fmt)(source, dtype=torch.float64, device="cpu")
            progress("input_conversion_and_fingerprint", context)
            source_nonfinite = int((~torch.isfinite(a)).sum())
            source_max_abs = float(a.abs().max()) if a.numel() else 0.0
            a = a.to(dtype)
            matrix_fingerprint = None
            if metrics_enabled:
                import hashlib
                digest = hashlib.sha256(f"{fmt}:{tuple(shape)}:{combo[0]}".encode())
                for tensor in (a, first.long(), second.long()):
                    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                matrix_fingerprint = digest.hexdigest()
            import scipy.sparse as sparse
            values = a.double().numpy()
            host = (sparse.coo_matrix((values, (first.numpy(), second.numpy())), shape=shape)
                    if fmt == "coo" else getattr(sparse, fmt + "_matrix")((values, first.numpy(), second.numpy()), shape=shape))
            for op, op_b, layout, ncols in itertools.product(ops, op_bs, layouts, ns):
                structure_cache = {}
                execution_ptr = ((host if op == "non" else host.T).tocsr().indptr
                                 if metrics_enabled and operator == "spmv_csr" else None)
                # Original storage order matters for COO. These CPU models are
                # outside prepare/run, never consumed by the device algorithm.
                coo_structure = (coo_run_structure((first if op == "non" else second).numpy())
                                 if metrics_enabled and fmt == "coo" else None)
                csc_structure = (csc_column_structure(second.numpy(), first.numpy(), shape[0], op=op)
                                 if metrics_enabled and fmt == "csc" else None)
                context.update(op_a=op, op_b=op_b, layout=layout, dense_cols=ncols)
                progress("reference_and_rhs", context)
                inner = shape[1] if op == "non" else shape[0]
                generator = torch.Generator().manual_seed(2026)
                effective = torch.randn((inner, ncols) if matrix else (inner,), generator=generator).to(rhs_dtype)
                if diagnostic == "ones_fanin":
                    effective = torch.ones((inner, ncols), dtype=rhs_dtype)
                elif diagnostic == "rectangular":
                    effective = ((torch.arange(inner * ncols).reshape(inner, ncols) % 17 - 8) / 8).to(rhs_dtype)
                rhs_fingerprint = None
                if metrics_enabled:
                    rhs_fingerprint = hashlib.sha256(effective.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                golden = (host if op == "non" else host.T) @ effective.double().numpy()
                rhs = (effective.T if matrix and op_b != "non" else effective).to(device)
                if matrix:
                    rhs = rhs.contiguous() if layout == "row" else rhs.T.contiguous().T
                supported = query(op=op, dtype=dtype, rhs_dtype=rhs_dtype, out_dtype=output_dtype,
                                  backend=backend, **({"layout": layout} if matrix else {}))
                selected = list(supported) if requested in ("all", "compare") else requested.split(",")
                if requested in ("all", "compare"):
                    print(f"[compare] {combo} {op}/{op_b} {layout}: {', '.join(selected) or '(none)'}")
                    for name in query():
                        if name not in supported:
                            print(f"  excluded {name}: unsupported type/op/layout/backend combination")
                elif requested != "auto" and any(name not in supported for name in selected):
                    raise ValueError(f"{requested} does not support {combo}, {op}, {layout}, {backend}")
                for index, ptr in ((i, p) for i in indices for p in (pointers or (i,))):
                    context.update(index_dtype=index, indptr_dtype=ptr)
                    progress("input_transfer", context)
                    arrays = (a.to(device), first.to(device=device, dtype=getattr(torch, index)),
                              second.to(device=device, dtype=getattr(torch, ptr)))
                    vendor_results = {}
                    vendor_case += 1
                    for alg in selected:
                        checkpoint = None
                        row = dict(matrix=source_name, dtype=combo[0], type_combo=":".join(combo), a_dtype=combo[0],
                                   rhs_dtype=combo[1], compute_dtype="float32", output_dtype=combo[2], op=op,
                                   op_a=op, op_b=op_b, layout=layout, dense_cols=ncols, index_dtype=index,
                                   indptr_dtype=ptr, alg=alg, alg_requested=alg, backend=backend, m=shape[0], n=shape[1], nnz=a.numel(),
                                   timed="prepared_run", sample_rule="paired total median +/-10% mean", status="ERROR",
                                   source_nonfinite=source_nonfinite, source_max_abs=source_max_abs,
                                   diagnostic_case=diagnostic, flagsparse_measurement_reused=False,
                                   reference="cpu_fp64_quantized_inputs", rtol=close_tolerances(output_dtype)[0],
                                   atol=close_tolerances(output_dtype)[1])
                        if metrics_enabled:
                            row.update(environment, matrix_fingerprint=matrix_fingerprint, rhs_fingerprint=rhs_fingerprint)
                            if coo_structure is not None:
                                row["coo_structure"] = coo_structure
                            if csc_structure is not None:
                                row["csc_structure"] = csc_structure
                        try:
                            if operator == "spmv_csr" and alg != "auto":
                                spec = module.get_spmv_csr_algorithm_spec(alg)
                                if index not in spec["index_dtypes"]:
                                    if requested in ("all", "compare"):
                                        row.update(status="SKIP", reason=f"{alg} requires column indices in {spec['index_dtypes']}")
                                        emit(row)
                                        continue
                                    raise ValueError(f"{alg} does not support column indices {index}")
                            progress("prepare", row)
                            prepared = prepare(*arrays, shape, op=op, alg=alg, config=config)
                            if execution_ptr is not None:
                                structure_key = json.dumps(prepared.config, sort_keys=True)
                                if structure_key not in structure_cache:
                                    structure_cache[structure_key] = csr_tile_structure(execution_ptr, config=prepared.config)
                                row["csr_structure"] = structure_cache[structure_key]
                            kwargs = dict(out_dtype=output_dtype, return_time=True, return_meta=True)
                            if matrix:
                                kwargs["op_b"] = op_b
                                if fmt in ("csr", "coo"):
                                    kwargs["dense_layout"] = layout
                            # Compile and validate once even when --warmup=0.
                            progress("first_run_compile_and_execute", row)
                            run(prepared, rhs, **kwargs)
                            progress("warmup", row)
                            for _ in range(args.warmup):
                                run(prepared, rhs, **kwargs)
                            progress("event_samples", row)
                            gpu, cpu = [], []
                            for _ in range(args.iters):
                                result, elapsed, meta = run(prepared, rhs, **kwargs)
                                gpu.append(float(meta.get("gpu_ms", elapsed)))
                                cpu.append(float(meta.get("process_cpu_ms", 0.0)))
                            row.update(paired_timing_summary(cpu, gpu))
                            row["ms"] = row["gpu_ms"] + row["process_cpu_ms"]
                            if metrics_enabled:
                                row.update(sparse_work_metrics(operator=operator, shape=tuple(map(int, shape)), nnz=a.numel(),
                                    dense_cols=ncols, a_dtype=combo[0], rhs_dtype=combo[1], output_dtype=combo[2],
                                    index_dtype=index, indptr_dtype=ptr, op=op, ms=row["ms"], gpu_ms=row["gpu_ms"]))
                                sample_summary = benchmark_sample_summary([c + g for c, g in zip(cpu, gpu)])
                                sample_summary.pop("sample_kept", None)  # retain the paired median fallback count
                                row.update(sample_summary)
                                row.update(sparse_reference_metrics(row, **reference_options))
                                # CPU processing inside the outer event can already be included
                                # in its elapsed interval. Keep historical ms, label the risk.
                                row["cpu_gpu_overlap_risk"] = any(value > 0 for value in cpu)
                                import time
                                progress("wall_samples", row)
                                wall_samples = []
                                wall_kwargs = dict(out_dtype=output_dtype)
                                if matrix:
                                    wall_kwargs["op_b"] = op_b
                                    if fmt in ("csr", "coo"):
                                        wall_kwargs["dense_layout"] = layout
                                # Separate sampling: no internal timing/meta/phase events.
                                for _ in range(args.iters):
                                    ACCEL.synchronize()
                                    started = time.perf_counter()
                                    wall_result = run(prepared, rhs, **wall_kwargs)
                                    ACCEL.synchronize()
                                    wall_samples.append((time.perf_counter() - started) * 1000.0)
                                    del wall_result
                                row["prepared_run_ms"] = filtered_avg_ms(wall_samples)
                                row["prepared_run_time_basis"] = "synchronized_wall_run_without_internal_timing"
                                wall_ms = row["prepared_run_ms"]
                                row["prepared_useful_gflops"] = row["useful_flops"] / (wall_ms * 1e6) if wall_ms > 0 else None
                                row["prepared_logical_gbs"] = row["logical_bytes"] / (wall_ms * 1e6) if wall_ms > 0 else None
                            progress("accuracy", row)
                            accuracy = tensor_accuracy_metrics(result, torch.as_tensor(golden).to(output_dtype),
                                                               *close_tolerances(output_dtype))
                            row.update({key: value for key, value in accuracy.items() if key != "passed"})
                            row["status"] = "PASS" if accuracy["passed"] else "FAIL"
                            if row["status"] == "FAIL":
                                row["reason"] = "correctness check failed"
                            row.update(input_nonfinite=int((~torch.isfinite(a)).sum()),
                                       rhs_nonfinite=int((~torch.isfinite(effective)).sum()),
                                       reference_nonfinite=int((~torch.isfinite(torch.as_tensor(golden).to(output_dtype))).sum()),
                                       output_nonfinite=int((~torch.isfinite(result)).sum()))
                            if any(row[key] for key in ("input_nonfinite", "rhs_nonfinite", "reference_nonfinite", "output_nonfinite")):
                                row.update(status="FAIL", reason="non-finite input/reference/output; see finite-count columns")
                            if args.timing:
                                progress("phase_diagnostics", row)
                                _, _, diagnostics = run(prepared, rhs, timing=True, **kwargs)
                                row.update({key: diagnostics.get(key) for key in ("process_gpu_ms", "compute_ms", "fused_plan_compute_ms", "init_ms", "producer_ms", "consumer_ms", "finish_ms", "scatter_ms", "reduce_ms")})
                            row["metadata"] = meta
                            if operator == "spmv_csr":
                                row.update({key: meta.get(key) for key in
                                            ("row_queue_capacity", "consumer_grid_blocks", "consumer_grid_limit",
                                             "consumer_rounds_capacity", "task_acquisition", "cooperative_parts",
                                             "row_grid_blocks", "partial_slot_capacity")})
                            row["alg_resolved"] = meta.get("alg_resolved", meta.get("alg", alg))
                            row["vendor_status"] = "SKIP"
                            row["vendor_reason"] = "disabled"
                            if not (getattr(args, "no_vendor", False) or getattr(args, "no_cusparse", False)):
                                vendor_log_dir.mkdir(parents=True, exist_ok=True)
                                native_row = dict(row)
                                output_stride = tuple(result.stride())
                                for variant_index, (vendor_alg, preprocess) in enumerate(itertools.product(vendor_algs, vendor_preprocessing)):
                                    row = dict(native_row, vendor_alg_requested=vendor_alg,
                                               vendor_preprocess_requested="auto" if preprocess is None else "on" if preprocess else "off",
                                               flagsparse_measurement_reused=variant_index > 0)
                                    checkpoint = vendor_log_dir / f"case_{vendor_case:06d}_{alg}_variant_{variant_index}.flagsparse.json"
                                    _write_benchmark_state(checkpoint, dict(row, checkpoint_state="before_vendor",
                                        vendor_status="PENDING", vendor_reason="baseline resolution not completed"))
                                    # Cache only results for this exact input condition, never device plans.
                                    vendor_key = (source_name, combo, tuple(shape), op, op_b, tuple(rhs.stride()),
                                                  output_stride, index, ptr, backend, vendor_alg, preprocess)
                                    vendor_result = vendor_results.get(vendor_key)
                                    if vendor_result is None:
                                        row.update(vendor_status="PENDING", vendor_reason="baseline resolution not completed")
                                        progress("vendor_payload", row)
                                        rtol, atol = close_tolerances(output_dtype)
                                        payload = dict(fmt=fmt, arrays=tuple(t.detach().cpu() for t in arrays),
                                                       shape=shape, rhs=rhs.detach().cpu(),
                                                       output_shape=tuple(result.shape), output_stride=output_stride, output_dtype=output_dtype,
                                                       op_a=op, op_b=op_b, backend=backend, matrix=matrix,
                                                       golden=torch.as_tensor(golden), rtol=rtol, atol=atol,
                                                       warmup=args.warmup, iters=args.iters,
                                                       device_index=rhs.device.index or 0, vendor_alg=vendor_alg,
                                                       vendor_preprocess=preprocess, diagnostic=bool(diagnostic),
                                                       narrow_tolerances=close_tolerances(dtype))
                                        progress("vendor_worker", row)
                                        vendor_result = vendor.benchmark_isolated(
                                            payload, log_prefix=vendor_log_dir / f"case_{vendor_case:06d}_variant_{len(vendor_results)}",
                                            timeout=args.vendor_timeout)
                                        vendor_results[vendor_key] = vendor_result
                                    row.update({key: value for key, value in vendor_result.items() if key.startswith("vendor_")})
                                    row["speedup_vs_vendor"] = finite_speedup(row.get("vendor_ms"), row.get("ms"))
                                    _write_benchmark_state(checkpoint, dict(row, checkpoint_state="vendor_completed"))
                                    emit(row)
                                    if row["vendor_status"] == "ERROR" and args.fail_fast:
                                        raise SystemExit("vendor execution failed (--fail-fast)")
                                continue
                        except BackendCapabilityUnavailable as exc:
                            row.update(status="SKIP", reason=str(exc))
                        except Exception as exc:
                            row.update(status="ERROR", reason=f"{operator}/{alg}/{combo}/{op}/{op_b}: {exc}")
                            if row.get("vendor_status") == "PENDING":
                                row.update(vendor_status="ERROR", vendor_reason=str(exc))
                            traceback.print_exc()
                            if args.fail_fast:
                                emit(row)
                                raise
                        emit(row)
        progress("benchmark_completed", {})
    finally:
        if stream:
            stream.close()
    return True
