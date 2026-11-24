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


CSR_LOCALITY_FIELDS = ["locality_diagnostic_status", "locality_diagnostic_reason", "locality_diagnostic_path",
                       "locality_summary", "structure_fingerprint", "matrix_fingerprint", "rhs_fingerprint"]
CSR_EXPERIMENT_FIELDS = ["event_count", "events_per_program", "atomic_scope", "local_ordering_scope",
                         "row_order", "compute_timing_scope"]


def csr_locality_structure(indptr, indices, shape, *, op="non", element_bytes=4, rhs_stride=1,
                           window_rows=64, rows_per_group=8, lanes_per_row=8, max_windows=4096):
    """CPU-only proxies; neither hardware cache counters nor execution plans."""
    import numpy as np

    if op not in ("non", "trans", "conj"):
        raise ValueError("invalid locality operation")
    if min(window_rows, rows_per_group, lanes_per_row, max_windows, element_bytes, rhs_stride) <= 0 or window_rows % rows_per_group:
        raise ValueError("invalid locality model dimensions")
    ptr = np.asarray(indptr, dtype=np.int64)
    col = np.asarray(indices, dtype=np.int64)
    if len(ptr) != int(shape[0]) + 1 or ptr[0] != 0 or ptr[-1] != len(col) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("invalid CSR structure for locality model")
    if len(col) and (col.min() < 0 or col.max() >= int(shape[1])):
        raise ValueError("locality column index out of range")
    if op != "non":
        order = np.argsort(col, kind="stable")
        owners = np.repeat(np.arange(shape[0], dtype=np.int64), np.diff(ptr))
        counts = np.bincount(col, minlength=shape[1])
        ptr = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
        col = owners[order]  # retain duplicates, zeros and stable storage order
    lengths = np.diff(ptr)
    count = (len(lengths) + window_rows - 1) // window_rows
    selected = np.unique(np.linspace(0, count - 1, min(max_windows, count), dtype=np.int64)) if count else np.empty(0, dtype=np.int64)
    def scheduled(values):
        padded = np.zeros(window_rows, dtype=np.int64)
        padded[:len(values)] = values
        # Python integers avoid overflow in aggregate cost proxies.
        return sum(rows_per_group * lanes_per_row * ((int(v) + lanes_per_row - 1) // lanes_per_row)
                   for v in padded.reshape(-1, rows_per_group).max(axis=1))
    def address_blocks(values, size):
        # Object arithmetic avoids overflow for large legal dense strides.
        addresses = values.astype(object) * int(element_bytes) * int(rhs_stride)
        return np.unique(addresses // size)
    def inspect(window):
        lo, hi = window * window_rows, min((window + 1) * window_rows, len(lengths))
        values = col[ptr[lo]:ptr[hi]]
        prev = col[ptr[max(0, lo - window_rows)]:ptr[lo]]
        result = dict(window=int(window), nnz=int(len(values)), rows=int(hi - lo),
                      distinct_x_ratio=float(len(np.unique(values)) / len(values)) if len(values) else None,
                      scheduled_original=scheduled(lengths[lo:hi]),
                      scheduled_grouped=scheduled(np.sort(lengths[lo:hi])[::-1]))
        for size in (32, 64, 128):
            blocks = address_blocks(values, size)
            result[f"model_blocks_{size}"] = int(len(blocks))
            result[f"adjacent_reuse_{size}"] = (float(len(np.intersect1d(blocks, address_blocks(prev, size))) / len(blocks))
                                                   if lo and len(blocks) else None)
        return result
    samples = [inspect(int(w)) for w in selected]
    # Supplemental extreme windows are explicitly separate from the grid sample.
    extremes = np.unique(np.argsort(lengths, kind="stable")[-min(8, len(lengths)):] // window_rows) if len(lengths) else []
    supplemental = sorted(set(map(int, extremes)) | ({0, count - 1} if count else set()))
    original = sum(s["scheduled_original"] for s in samples)
    grouped = sum(s["scheduled_grouped"] for s in samples)
    summary = dict(model="csr_locality_v1", op=op, total_windows=count, sampled_windows=len(samples),
                   coverage="exact" if len(samples) == count else "deterministic_even_windows_not_unbiased",
                   sampled_nnz=sum(s["nnz"] for s in samples), total_nnz=int(len(col)),
                   scheduled_original=original, scheduled_grouped=grouped,
                   grouped_position_ratio=grouped / original if original else None,
                   element_bytes=element_bytes, rhs_stride=rhs_stride,
                   window_rows=window_rows, rows_per_group=rows_per_group, lanes_per_row=lanes_per_row,
                   address_alignment="normalized_zero_offset_not_device_address",
                   metric_scope="logical_proxies_not_hardware_cache_or_occupancy",
                   grouping_scope="hypothetical_window_model_not_claimed_algorithm_schedule")
    return dict(summary=summary, windows=samples, supplemental_windows=[inspect(w) for w in supplemental])


def write_csr_locality_diagnostics(csv_path, row, arrays, rhs, shape, op, config, cache):
    """Best-effort diagnostics after timing; errors never erase the timed result."""
    import gzip
    import hashlib
    import json
    import traceback

    try:
        cfg = config or {}
        # Other algorithms use the same explicit comparison model, not their launch geometry.
        actual_window = row.get("alg") == "csr_window_grouped_reduce"
        w, r, v = ((cfg["window_rows"], cfg["rows_per_group"], cfg["lanes_per_row"])
                   if actual_window else (64, 8, 8))
        source_key = ("locality_source", row["matrix"], tuple(shape),
                      tuple((str(t.dtype), t.data_ptr(), tuple(t.stride())) for t in (*arrays, rhs)))
        if source_key not in cache:
            data, indices, indptr = (t.detach().cpu().contiguous() for t in arrays)
            digest = hashlib.sha256()
            for t in (data, indices, indptr):
                digest.update(str((str(t.dtype), tuple(t.shape))).encode())
                digest.update(t.view(torch.uint8).numpy().tobytes())
            structure = hashlib.sha256(str(tuple(shape)).encode() +
                                       indices.to(torch.int64).numpy().tobytes() +
                                       indptr.to(torch.int64).numpy().tobytes()).hexdigest()
            vector = rhs.detach().cpu().contiguous()
            fingerprint = hashlib.sha256(vector.view(torch.uint8).numpy().tobytes()).hexdigest()
            cache[source_key] = (indices.numpy(), indptr.numpy(), digest.hexdigest(), structure, fingerprint)
        indices, indptr, matrix_hash, structure_hash, rhs_hash = cache[source_key]
        key = (source_key, op, rhs.element_size(), rhs.stride(0), w, r, v)
        if key not in cache:
            payload = csr_locality_structure(indptr, indices, shape, op=op, element_bytes=rhs.element_size(),
                                             rhs_stride=rhs.stride(0), window_rows=w, rows_per_group=r, lanes_per_row=v)
            identity = dict(matrix=row["matrix"], structure=structure_hash, op=op,
                            element_bytes=rhs.element_size(), stride=rhs.stride(0), w=w, r=r, v=v)
            token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            folder = Path(str(csv_path) + ".locality")
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{token}.json.gz"
            with gzip.open(target, "wt", encoding="utf-8") as handle:
                json.dump(dict(identity=identity, **payload), handle, ensure_ascii=False)
            cache[key] = dict(locality_diagnostic_status="collected", locality_diagnostic_reason="",
                              locality_diagnostic_path=str(target), locality_summary=payload["summary"])
        return dict(cache[key], structure_fingerprint=structure_hash,
                    matrix_fingerprint=row.get("matrix_fingerprint") or matrix_hash,
                    rhs_fingerprint=row.get("rhs_fingerprint") or rhs_hash)
    except Exception as exc:
        print(f"[locality diagnostic ERROR] {row.get('matrix')} {row.get('alg')}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return dict(locality_diagnostic_status="ERROR", locality_diagnostic_reason=f"{type(exc).__name__}: {exc}")

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
    if {"short_row_threshold", "producer_num_warps", "rows_per_program"} <= cfg.keys() and "long_grid_blocks" not in cfg:
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
    if "cooperative_parts" in cfg and "long_grid_blocks" not in cfg:
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
    if "long_grid_blocks" in cfg:
        short, long = cfg["short_row_threshold"], cfg["long_row_threshold"]
        parts, block, limit = cfg["cooperative_parts"], cfg["block_nnz"], cfg["long_grid_blocks"]
        if (any(type(v) is not int for v in (short, long, parts, block, limit)) or
                not 0 <= short < long or parts not in (1, 2, 4, 8, 16) or block <= 0 or not 1 <= limit <= 1024):
            raise ValueError("invalid three-tier structure configuration")
        medium = lengths[(lengths > short) & (lengths <= long)]
        long_lengths = lengths[lengths > long]
        cm, cl = min(rows, nnz // (short + 1)), min(rows, nnz // (long + 1))
        medium_limit = cfg.get("medium_grid_blocks", cm)
        if "medium_grid_blocks" in cfg and (type(medium_limit) is not int or not 1 <= medium_limit <= 1024):
            raise ValueError("invalid bounded medium structure configuration")
        medium_grid = min(cm, medium_limit)
        grid = min(cl, limit)
        useful_parts = int(np.minimum((long_lengths + block - 1) // block, parts).sum())
        # Order-independent task sizes. Never infer worker load from CPU row order.
        chunks = (long_lengths + block - 1) // block
        quotient, remainder = chunks // parts, chunks % parts
        histogram = {}
        for values, weights in ((quotient, parts - remainder), (quotient + 1, remainder)):
            for size, weight in zip(values.tolist(), weights.tolist()):
                if weight:
                    histogram[size] = histogram.get(size, 0) + weight
        total_tasks = int(long_lengths.size) * parts
        def task_percentile(q):
            rank, count = max(1, int(np.ceil(q * total_tasks))), 0
            for size in sorted(histogram):
                count += histogram[size]
                if count >= rank:
                    return size
        task_distribution = dict(
            unit="K_sized_chunks_not_time", tasks=total_tasks,
            total_chunks=int(chunks.sum()), zero_tasks=histogram.get(0, 0),
            p50=task_percentile(.5), p90=task_percentile(.9), p99=task_percentile(.99),
            maximum=max(histogram, default=0), histogram=histogram,
            queue_order="not_observed", worker_balance="not_inferred")
        result["three_tier_queue"] = dict(
            short_row_threshold=short, long_row_threshold=long,
            expected_short_rows=int(np.count_nonzero(lengths <= short)),
            expected_medium_rows=int(medium.size), expected_long_rows=int(long_lengths.size),
            expected_medium_nnz=int(medium.sum()), expected_long_nnz=int(long_lengths.sum()),
            medium_nnz_fraction=int(medium.sum()) / nnz if nnz else None,
            long_nnz_fraction=int(long_lengths.sum()) / nnz if nnz else None,
            medium_queue_capacity=cm, long_queue_capacity=cl, medium_grid_blocks=medium_grid,
            long_row_grid_blocks=grid, long_consumer_grid_blocks=grid * parts,
            expected_nonempty_part_tasks=useful_parts,
            expected_empty_part_tasks=int(long_lengths.size) * parts - useful_parts,
            long_part_chunks=task_distribution,
            count_source="cpu_structure_model_not_gpu_readback")
        if "medium_grid_blocks" in cfg:
            result["three_tier_queue"].update(
                medium_grid_limit=medium_limit,
                medium_rounds_capacity=(cm + medium_grid - 1) // medium_grid if medium_grid else 0,
                expected_medium_active_programs=min(int(medium.size), medium_grid),
                expected_medium_empty_programs=max(0, medium_grid - int(medium.size)),
                expected_medium_max_rounds=(int(medium.size) + medium_grid - 1) // medium_grid if medium_grid else 0)
        if "long_schedule" in cfg:
            schedule, batch, requested = cfg["long_schedule"], cfg["ticket_batch"], cfg["long_worker_programs"]
            if (schedule not in ("static", "ticket") or type(batch) is not int or batch not in (1, 4)
                    or type(requested) is not int or requested not in (0, 32, 64, 128, 256, 512, 1024, 2048, 4096)
                    or (schedule == "static" and batch != 1)):
                raise ValueError("invalid ticket structure configuration")
            workers = min(cl * parts, requested) if requested else grid * parts
            tasks = int(long_lengths.size) * parts
            claims = ((tasks - workers + batch - 1) // batch + workers
                      if schedule == "ticket" and tasks > workers else 0)
            result["three_tier_queue"].update(
                long_schedule=schedule, ticket_batch=batch, long_worker_limit=requested,
                long_consumer_grid_blocks=workers, task_capacity=cl * parts,
                expected_long_tasks=tasks, expected_initial_tasks=min(tasks, workers),
                predicted_ticket_acquisitions=claims, predicted_ticket_final_value=batch * claims,
                ticket_model_source="cpu_structure_not_device_counter")
    return result


def csr_queue_work_model(indptr, medium_rows, long_rows, *, block, parts, medium_grid, workers,
                         long_grid, mapping="flat"):
    """Offline work model using the actually observed producer order."""
    import numpy as np

    ptr = np.asarray(indptr, dtype=np.int64)
    medium = np.asarray(medium_rows, dtype=np.int64)
    long = np.asarray(long_rows, dtype=np.int64)
    if (ptr.ndim != 1 or not ptr.size or ptr[0] != 0 or np.any(np.diff(ptr) < 0)
            or mapping not in ("flat", "two_dimensional") or block <= 0 or parts <= 0
            or medium_grid < 0 or workers < 0 or long_grid < 0):
        raise ValueError("invalid task diagnostic model parameters")
    for rows in (medium, long):
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= ptr.size - 1):
            raise ValueError("invalid diagnostic queue owner")
    if len(np.unique(np.concatenate((medium, long)))) != medium.size + long.size:
        raise ValueError("duplicate/overlapping diagnostic row queues")
    if (medium.size and not medium_grid) or (long.size and not workers):
        raise ValueError("nonempty queue requires a nonzero grid")
    if mapping == "two_dimensional" and workers != long_grid * parts:
        raise ValueError("2D grid does not match worker count")
    lengths = np.diff(ptr)
    medium_chunks = (lengths[medium] + block - 1) // block
    long_chunks = (lengths[long] + block - 1) // block
    mt = np.zeros(medium_grid, dtype=np.int64)
    mw = np.zeros(medium_grid, dtype=np.int64)
    if medium.size:
        owners = np.arange(medium.size, dtype=np.int64) % medium_grid
        np.add.at(mt, owners, 1)
        np.add.at(mw, owners, medium_chunks)
    tasks = np.zeros(workers, dtype=np.int64)
    work = np.zeros(workers, dtype=np.int64)
    maxima = np.zeros(workers, dtype=np.int64)
    initial = np.zeros(workers, dtype=np.int64)
    histogram = {}
    slots = np.arange(long.size, dtype=np.int64)
    for part in range(parts):
        sizes = long_chunks * (part + 1) // parts - long_chunks * part // parts
        if not long.size:
            continue
        ids = slots * parts + part
        owners = ids % workers if mapping == "flat" else part * long_grid + slots % long_grid
        np.add.at(tasks, owners, 1)
        np.add.at(work, owners, sizes)
        np.maximum.at(maxima, owners, sizes)
        first = ids < workers if mapping == "flat" else slots < long_grid
        np.add.at(initial, owners[first], sizes[first])
        unique, counts = np.unique(sizes, return_counts=True)
        for size, count in zip(unique.tolist(), counts.tolist()):
            histogram[size] = histogram.get(size, 0) + count
    def summary(values):
        return dict(total=int(values.sum()), maximum=int(values.max()) if values.size else 0,
                    mean=float(values.mean()) if values.size else 0.0,
                    max_over_mean=float(values.max() / values.mean()) if values.size and values.mean() else None,
                    nonzero=int(np.count_nonzero(values)))
    return dict(model="observed_queue_static_chunk_work_v1", unit="K_sized_chunks_not_time_or_occupancy",
                mapping=mapping, block_nnz=block, parts=parts,
                medium_row_ids=medium.tolist(), long_row_ids=long.tolist(),
                medium_row_chunks=medium_chunks.tolist(), long_row_chunks=long_chunks.tolist(),
                long_part_chunk_histogram=histogram,
                medium_worker_tasks=mt.tolist(), medium_worker_chunks=mw.tolist(),
                long_worker_tasks=tasks.tolist(), long_worker_chunks=work.tolist(),
                long_worker_max_task_chunks=maxima.tolist(), long_worker_initial_chunks=initial.tolist(),
                long_worker_remaining_chunks=(work-initial).tolist(),
                medium_summary=summary(mw), long_summary=summary(work),
                task_count=int(tasks.sum()), total_long_chunks=int(work.sum()))


def collect_csr_task_diagnostics(prepared, rhs, expected_output, *, repeats=1):
    """Separate, synchronous diagnostic launches; never used inside benchmark sampling.

    Temporarily wrap the producer to retain this run's real queue tensors. Only
    the ticket consumer uses an instrumented replacement; all overrides restore
    even after compilation/execution failure. Standalone benchmark use is serial.
    """
    import hashlib
    import importlib
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels
    from flagsparse.sparse_operations import _spmv_csr_config as policy

    if prepared.alg not in policy.THREE_TIER_ALGORITHMS:
        return dict(status="not_applicable", reason="requires a three-tier CSR route")
    if type(repeats) is not int or not 1 <= repeats <= 16:
        raise ValueError("task diagnostic repeats must be between 1 and 16")
    route = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
    producer, ticket = kernels.three_tier_producer_kernel, kernels.three_tier_ticket_consumer_kernel
    ptr = prepared.kernel_indptr.cpu().numpy()
    expected = expected_output.detach().cpu()
    cfg = prepared.config
    results = []
    for iteration in range(repeats):
        captured = {}
        class ProducerCapture:
            def __getitem__(self, grid):
                launch = producer[grid]
                def call(*args, **kwargs):
                    captured.update(counts=args[5], medium=args[6], long=args[7])
                    return launch(*args, **kwargs)
                return call
        class TicketTrace:
            def __getitem__(self, grid):
                def call(*args, **kwargs):
                    trace = torch.empty((kwargs["WORKERS"], 6), dtype=torch.int64, device=rhs.device)
                    captured["trace"] = trace
                    return kernels.three_tier_ticket_trace_kernel[grid](*args, trace, **kwargs)
                return call
        try:
            kernels.three_tier_producer_kernel = ProducerCapture()
            if prepared.alg == "csr_three_tier_ticket_reduce":
                kernels.three_tier_ticket_consumer_kernel = TicketTrace()
            output = route.flagsparse_spmv_csr_run(prepared, rhs, out_dtype=expected_output.dtype)
            # These synchronizations and reads are outside all performance samples.
            torch.testing.assert_close(output.detach().cpu(), expected, rtol=0, atol=0, equal_nan=True)
        finally:
            kernels.three_tier_producer_kernel = producer
            kernels.three_tier_ticket_consumer_kernel = ticket
        stats = policy.three_tier_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)
        gm = min(stats["medium_queue_capacity"], cfg.get("medium_grid_blocks", stats["medium_queue_capacity"]))
        gl = min(stats["long_queue_capacity"], cfg["long_grid_blocks"])
        workers = (policy.ticket_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)["long_worker_programs"]
                   if prepared.alg == "csr_three_tier_ticket_reduce" else gl * cfg["cooperative_parts"])
        counts = captured["counts"].cpu().tolist() if captured else [0, 0, 0]
        qm, ql = counts[:2]
        medium = captured["medium"][:qm].cpu().numpy() if captured else []
        long = captured["long"][:ql].cpu().numpy() if captured else []
        # Validate actual queues against the CSR, not just their counts.
        import numpy as np
        lengths = np.diff(ptr)
        assert np.array_equal(np.sort(medium), np.flatnonzero((lengths > cfg["short_row_threshold"]) & (lengths <= cfg["long_row_threshold"])))
        assert np.array_equal(np.sort(long), np.flatnonzero(lengths > cfg["long_row_threshold"]))
        mapping = "two_dimensional" if prepared.alg in ("csr_three_tier_queue_reduce", "csr_three_tier_bounded_reduce") else "flat"
        model = csr_queue_work_model(ptr, medium, long, block=cfg["block_nnz"], parts=cfg["cooperative_parts"],
                                     medium_grid=gm, workers=workers, long_grid=gl, mapping=mapping)
        model.update(iteration=iteration, queue_order_sha256=hashlib.sha256(np.asarray(long, dtype="<i8").tobytes()).hexdigest(),
                     medium_queue_sha256=hashlib.sha256(np.asarray(medium, dtype="<i8").tobytes()).hexdigest(),
                     counters=counts, output_matches_uninstrumented=True,
                     worker_model_basis="static_assignment_on_this_actual_queue")
        if "trace" in captured:
            trace = captured["trace"].cpu().numpy()
            total = ql * cfg["cooperative_parts"]
            claims = ((total - workers + cfg["ticket_batch"] - 1) // cfg["ticket_batch"] + workers
                      if cfg["long_schedule"] == "ticket" and total > workers else 0)
            assert int(trace[:, 0].sum()) == total
            assert int(trace[:, 1].sum()) == model["total_long_chunks"]
            assert int(trace[:, 3].sum()) == claims and counts[2] == claims * cfg["ticket_batch"]
            model.update(observed_worker_trace=trace.tolist(),
                         trace_columns=["tasks", "chunks", "max_task_chunks", "claims", "initial_chunks", "remaining_chunks"],
                         trace_scope="separate_instrumented_run_not_timed_run",
                         trace_schedule_perturbation="extra_loads_and_stores_can_change_ticket_acquisition_order")
            if cfg["long_schedule"] == "static":
                assert trace[:, 0].tolist() == model["long_worker_tasks"]
                assert trace[:, 1].tolist() == model["long_worker_chunks"]
        results.append(model)
    return dict(status="collected", algorithm=prepared.alg, config=dict(cfg),
                output_dtype=str(expected_output.dtype), repetitions=results,
                timing_scope="diagnostics_excluded_from_all_performance_samples")


def write_csr_task_diagnostics(csv_path, row, prepared, rhs, expected_output, repeats):
    """Persist detailed queues separately; keep terminal/CSV compact."""
    import gzip
    import hashlib
    import json

    payload = collect_csr_task_diagnostics(prepared, rhs, expected_output, repeats=repeats)
    if payload["status"] == "not_applicable":
        return dict(task_diagnostic_status="not_applicable", task_diagnostic_reason=payload["reason"])
    identity = {k: row.get(k) for k in ("matrix", "dtype", "type_combo", "op", "index_dtype", "indptr_dtype", "alg")}
    identity["config"] = prepared.config
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(str(csv_path) + ".task_diagnostics")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (digest + ".json.gz")
    payload.update(identity=identity, timing_samples={key: row.get(key) for key in ("timing_samples", "wall_samples")},
                   measurement={key: row.get(key) for key in ("ms", "gpu_ms", "prepared_run_ms", "sample_rule",
                       "output_reused", "event_time_boundary", "output_allocation_policy")})
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    return dict(task_diagnostic_status="collected", task_diagnostic_reason="", task_diagnostic_path=str(target),
                task_diagnostic_runs=len(payload["repetitions"]),
                task_queue_order_variants=len({(r["medium_queue_sha256"], r["queue_order_sha256"]) for r in payload["repetitions"]}))


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
        ("--locality-diagnostics", dict(action="store_true", help="CSR SpMV only: CPU locality proxies outside sampling; requires --perf-metrics and CSV")),
        ("--task-diagnostics", dict(action="store_true", help="CSR SpMV only: separate queue/worker diagnostics, outside performance samples")),
        ("--task-diagnostic-repeats", dict(type=int, default=1, help="Separate diagnostic runs per three-tier case (1..16)")),
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
    if getattr(args, "locality_diagnostics", False) and (operator != "spmv_csr" or not args.perf_metrics):
        parser.error("--locality-diagnostics requires SpMV CSR and --perf-metrics")
    if getattr(args, "task_diagnostics", False) and (operator != "spmv_csr" or not args.perf_metrics or not 1 <= args.task_diagnostic_repeats <= 16):
        parser.error("--task-diagnostics requires SpMV CSR, --perf-metrics, and diagnostic repeats in 1..16")
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
        fields += ["output_reused", "event_time_boundary", "output_allocation_policy", "consumer_fused_ms", "consumer_timing_scope", "task_diagnostic_status", "task_diagnostic_reason",
                   "task_diagnostic_path", "task_diagnostic_runs", "task_queue_order_variants", "timing_samples", "wall_samples"]
        fields += ["medium_consumer_ms", "long_consumer_ms", "row_queue_capacity", "consumer_grid_blocks", "consumer_grid_limit",
                   "consumer_rounds_capacity", "task_acquisition", "cooperative_parts",
                   "row_grid_blocks", "partial_slot_capacity", "medium_queue_capacity", "long_queue_capacity", "medium_grid_blocks", "medium_grid_limit", "medium_rounds_capacity", "long_row_grid_blocks", "long_consumer_grid_blocks", "long_schedule", "ticket_batch", "long_worker_programs", "long_worker_limit", "task_capacity", "ticket_counter_dtype", "ticket_counter_max_bound"]
    if operator == "spmv_csr":
        fields += CSR_EXPERIMENT_FIELDS
        if args.locality_diagnostics:
            fields += CSR_LOCALITY_FIELDS
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
    fields = list(dict.fromkeys(fields))
    if args.locality_diagnostics and not csv_path:
        parser.error("--locality-diagnostics requires a CSV output path")
    if getattr(args, "task_diagnostics", False) and not csv_path:
        parser.error("--task-diagnostics requires a CSV output path")
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
                            if getattr(args, "task_diagnostics", False):
                                row["timing_samples"] = dict(cpu_ms=cpu, gpu_ms=gpu)
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
                                if getattr(args, "task_diagnostics", False):
                                    row["wall_samples"] = wall_samples
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
                                if operator == "spmv_csr":
                                    row.update({key: diagnostics.get(key) for key in ("medium_consumer_ms", "long_consumer_ms", "consumer_fused_ms")})
                            row["metadata"] = meta
                            if operator == "spmv_csr":
                                row["consumer_timing_scope"] = meta.get("consumer_timing_scope")
                                row.update(output_reused=False, event_time_boundary="internal_event_after_public_validation",
                                           output_allocation_policy="allocated_inside_each_run")
                                if getattr(args, "task_diagnostics", False):
                                    progress("task_diagnostics_outside_sampling", row)
                                    row.update(write_csr_task_diagnostics(csv_path, row, prepared, rhs, result, args.task_diagnostic_repeats))
                            if operator == "spmv_csr":
                                row.update({key: meta.get(key) for key in
                                            ("row_queue_capacity", "consumer_grid_blocks", "consumer_grid_limit",
                                             "consumer_rounds_capacity", "task_acquisition", "cooperative_parts",
                                             "row_grid_blocks", "partial_slot_capacity", "medium_queue_capacity", "long_queue_capacity", "medium_grid_blocks", "medium_grid_limit", "medium_rounds_capacity", "long_row_grid_blocks", "long_consumer_grid_blocks", "long_schedule", "ticket_batch", "long_worker_programs", "long_worker_limit", "task_capacity", "ticket_counter_dtype", "ticket_counter_max_bound")})
                            if operator == "spmv_csr":
                                row.update({key: meta.get(key) for key in CSR_EXPERIMENT_FIELDS})
                                if args.locality_diagnostics:
                                    progress("locality_diagnostics_outside_sampling", row)
                                    row.update(write_csr_locality_diagnostics(csv_path, row, arrays, rhs, shape, op, prepared.config, structure_cache))
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
