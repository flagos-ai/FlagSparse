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

"""Shared benchmark timing helpers for standalone test scripts."""

import time


def filtered_avg_ms(times, tolerance=0.10):
    if not times:
        return None
    values = [float(t) for t in times]
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    if median == 0.0:
        kept = [t for t in ordered if t == 0.0]
    else:
        lo = median * (1.0 - float(tolerance))
        hi = median * (1.0 + float(tolerance))
        kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def cuda_event_benchmark_filtered(op, warmup, iters):
    import torch

    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    samples = []
    for _ in range(max(1, int(iters))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = op()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return out, filtered_avg_ms(samples)


def cupy_event_benchmark_filtered(op, warmup, iters):
    import cupy as cp

    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    cp.cuda.runtime.deviceSynchronize()
    samples = []
    for _ in range(max(1, int(iters))):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        out = op()
        end.record()
        end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, end))
    return out, filtered_avg_ms(samples)


def cpu_wall_benchmark_filtered(op, warmup, iters):
    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    samples = []
    for _ in range(max(1, int(iters))):
        start = time.perf_counter()
        out = op()
        samples.append((time.perf_counter() - start) * 1000.0)
    return out, filtered_avg_ms(samples)


def sparse_work_metrics(*, operator, shape, nnz, dense_cols, a_dtype, rhs_dtype,
                        output_dtype, index_dtype, indptr_dtype, op, ms, gpu_ms=None):
    """Algorithm-independent useful work, NOT hardware DRAM traffic/counter data.

    Model v1: sparse input once, RHS once per stored contribution/output column,
    and final output once. Excludes caches, padding, plans, scratch and atomics.
    Rates use the complete run, never the independently measured compute phase.
    """
    import math

    if operator not in {f"{kind}_{fmt}" for kind in ("spmv", "spmm") for fmt in ("csr", "coo", "csc")}:
        raise ValueError(f"unsupported work-metrics operator: {operator}")
    if len(shape) != 2 or any(type(v) is not int or v < 0 for v in shape):
        raise ValueError("shape must contain two nonnegative integers")
    if type(nnz) is not int or nnz < 0 or type(dense_cols) is not int or dense_cols < 0:
        raise ValueError("nnz and dense_cols must be nonnegative integers")
    if op not in ("non", "trans", "conj"):
        raise ValueError("invalid op")
    sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int32": 4, "int64": 8}
    names = [str(v).removeprefix("torch.") for v in (a_dtype, rhs_dtype, output_dtype, index_dtype, indptr_dtype)]
    if any(v not in ("float16", "bfloat16", "float32", "float64") for v in names[:3]) or any(v not in ("int32", "int64") for v in names[3:]):
        raise ValueError("work-metrics v1 requires real values and int32/int64 indices")
    av, bv, cv, ix, ip = (sizes[v] for v in names)
    m, n = shape
    width = 1 if operator.startswith("spmv") else dense_cols
    output_rows = m if op == "non" else n
    fmt = operator.rsplit("_", 1)[1]
    sparse_bytes = nnz * (av + ix)
    sparse_bytes += nnz * ip if fmt == "coo" else ((m if fmt == "csr" else n) + 1) * ip
    rhs_bytes = nnz * width * bv
    output_bytes = output_rows * width * cv
    work_nnz = nnz if width and output_rows else 0
    if not width or not output_rows:
        sparse_bytes = rhs_bytes = output_bytes = 0
    flops = 2 * work_nnz * width
    logical_bytes = sparse_bytes + rhs_bytes + output_bytes

    def rate(amount, elapsed):
        if elapsed is None or isinstance(elapsed, bool):
            return None
        value = float(elapsed)
        return amount / (value * 1e6) if math.isfinite(value) and value > 0 else None

    return dict(metrics_model="spmx_logical_access_v1", metrics_time_basis="process_cpu_ms+gpu_ms",
                useful_flops=flops, sparse_input_bytes=sparse_bytes, rhs_gather_bytes=rhs_bytes,
                output_write_bytes=output_bytes, logical_bytes=logical_bytes,
                arithmetic_intensity=flops / logical_bytes if logical_bytes else None,
                useful_gflops=rate(flops, ms), logical_gbs=rate(logical_bytes, ms),
                gnnz_s=rate(work_nnz, ms), gpu_useful_gflops=rate(flops, gpu_ms),
                gpu_logical_gbs=rate(logical_bytes, gpu_ms),
                metrics_reason="" if rate(1, ms) is not None else "missing/nonpositive/nonfinite full-run time")


def benchmark_sample_summary(samples, tolerance=0.10):
    """Describe exactly the median-window sampling rule, without changing timings."""
    import math
    import statistics

    values = [float(v) for v in samples]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        return dict(sample_count=len(values), sample_kept=None, sample_min_ms=None,
                    sample_median_ms=None, sample_max_ms=None, sample_cv=None)
    median = statistics.median(values)
    kept = [v for v in values if median * (1 - tolerance) <= v <= median * (1 + tolerance)]
    mean = statistics.mean(values)
    return dict(sample_count=len(values), sample_kept=len(kept), sample_min_ms=min(values),
                sample_median_ms=median, sample_max_ms=max(values),
                sample_cv=statistics.pstdev(values) / mean if mean else 0.0)


def sparse_reference_metrics(metrics, *, reference_gbs=None, bandwidth_source=None,
                             peak_gflops=None, peak_source=None):
    """Normalize a logical model, never claim measured hardware utilization."""
    import math

    for value, source, name in ((reference_gbs, bandwidth_source, "bandwidth"),
                                (peak_gflops, peak_source, "compute peak")):
        if value is not None and (not math.isfinite(value) or value <= 0 or not source or not source.strip()):
            raise ValueError(f"{name} requires a positive finite value and a measurement/source description")
        if value is None and source:
            raise ValueError(f"{name} source requires its numeric value")
    if peak_gflops is not None and reference_gbs is None:
        raise ValueError("logical Roofline requires reference bandwidth as well as compute peak")
    result = dict(reference_bandwidth_gbs=reference_gbs, reference_bandwidth_source=bandwidth_source,
                  reference_peak_gflops=peak_gflops, reference_peak_source=peak_source,
                  logical_bw_ratio=None, gpu_logical_bw_ratio=None,
                  logical_roofline_gflops=None, gpu_logical_roofline_ratio=None)
    if reference_gbs is not None:
        for field, key in (("logical_gbs", "logical_bw_ratio"), ("gpu_logical_gbs", "gpu_logical_bw_ratio")):
            if metrics.get(field) is not None:
                result[key] = metrics[field] / reference_gbs
    intensity = metrics.get("arithmetic_intensity")
    if peak_gflops is not None and intensity is not None:
        ceiling = min(peak_gflops, reference_gbs * intensity)
        result["logical_roofline_gflops"] = ceiling
        if ceiling > 0 and metrics.get("gpu_useful_gflops") is not None:
            result["gpu_logical_roofline_ratio"] = metrics["gpu_useful_gflops"] / ceiling
    return result
