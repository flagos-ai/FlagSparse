"""Per-call GPU execution plans for CSR SpMM. Nothing is attached to prepared."""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl

from . import _common as common
from . import _spmm_csr_config as policy
from ._spmm_csr_kernels import finish_kernel, reduce_kernel, rows_kernel


def backend_caps(device):
    backend = common._backend_name()
    try:
        target = triton.runtime.driver.active.get_current_target()
        props = common._ACCEL.get_device_properties(device)
    except (AttributeError, RuntimeError, NotImplementedError) as exc:
        raise NotImplementedError(f"cannot discover {backend} target capabilities: {exc}") from exc
    target_name = str(getattr(target, "backend", "unknown"))
    def property_value(*names, default=None):
        for name in names:
            value = getattr(props, name, None)
            if value is not None:
                return value
        return default

    def capability(*names, default=None):
        value = property_value(*names, default=default)
        return None if value is None else bool(value)

    subgroup = int(getattr(target, "warp_size", 0) or getattr(props, "warp_size", 0))
    known_target = target_name in ("cuda", "hip", "musa", "maca", "ascend", "xpu", "gcu", "mlu")
    # CUDA/HIP document these launch limits. Other targets must expose theirs.
    documented_cuda_hip = backend in ("cuda", "rocm") and target_name in ("cuda", "hip")
    limit = int(property_value("max_threads_per_block", "max_work_group_size", "maxThreadsPerBlock", default=0) or
                (1024 if documented_cuda_hip else 0))
    warps = tuple(getattr(props, "legal_num_warps", ()) or
                  (tuple(w for w in (1, 2, 4, 8) if w * subgroup <= limit)
                   if subgroup and limit else ()))
    return policy.BackendCaps(
        backend, target_name, str(getattr(target, "arch", "unknown")), subgroup, limit, warps,
        fp64=capability("supports_fp64", "has_fp64", default=True if documented_cuda_hip else None),
        int64=capability("supports_int64", default=True if known_target else None),
        reduction=capability("supports_reduction", default=True if known_target else None),
        stable_sort=capability("supports_stable_sort", default=True if backend in ("cuda", "rocm") else None),
        scan=capability("supports_scan", default=True if backend in ("cuda", "rocm") else None),
    )


class Phases:
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = {"process_gpu_ms": [], "compute_ms": []}

    @contextmanager
    def measure(self, phase):
        if not self.enabled:
            yield
            return
        start = common._ACCEL.Event(enable_timing=True)
        end = common._ACCEL.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.events[phase].append((start, end))

    def results(self):
        if not self.enabled:
            return {}
        common._ACCEL.synchronize()
        return {key: sum(a.elapsed_time(b) for a, b in pairs)
                for key, pairs in self.events.items()}


def _view(tensor):
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _prefix(counts):
    return torch.cat((counts.new_zeros(1), torch.cumsum(counts, 0, dtype=torch.int64)))


def _classify(lengths, short, long):
    codes = (lengths > short).to(torch.int64) + (lengths > long).to(torch.int64)
    order = torch.argsort(codes, stable=True)
    counts = torch.bincount(codes, minlength=3).tolist()
    a, b, c = counts
    return order[:a], order[a:a + b], order[a + b:a + b + c]


def run(prepared, B, *, algorithm, config, config_meta, timing=False,
        diagnostics=False, dense_layout="row"):
    cfg = config
    phases = Phases(timing)
    m, n = prepared.n_rows, int(B.shape[1])
    complex_input = prepared.data.is_complex()
    acc = tl.float64 if B.dtype in (torch.float64, torch.complex128) else tl.float32
    launch = dict(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    counts = {"tile_rows": 0, "kparallel_rows": 0, "split_rows": 0}
    stats = dict(segment_count=0, reduction_levels=0, workspace_peak_bytes=0,
                 descriptor_peak_bytes_estimate=0, segment_batches=0)
    with phases.measure("compute_ms"):
        # Resolving lazy views is numerical input handling and remains inside run.
        A = prepared.data.resolve_conj().resolve_neg().contiguous()
        B = B.resolve_conj().resolve_neg()
        C = (torch.empty_strided((m, n), (1, max(m, 1)), device=B.device, dtype=B.dtype)
             if dense_layout == "col" else torch.empty((m, n), device=B.device, dtype=B.dtype))
        C.zero_()

    def rows(row_ids, tile=False):
        size = m if row_ids is None else row_ids.numel()
        if not size or not n:
            return
        r = cfg["tile_rows"] if tile else 1
        bk = cfg["tile_k"] if tile else cfg["block_k"]
        bn = cfg["tile_n"] if tile else cfg["block_n"] * cfg["panels"]
        with phases.measure("compute_ms"):
            rows_kernel[(triton.cdiv(size, r), triton.cdiv(n, bn))](
                _view(A), prepared.kernel_indices, prepared.kernel_indptr, _view(B), _view(C),
                prepared.kernel_indptr if row_ids is None else row_ids,
                prepared.kernel_indptr, prepared.kernel_indptr, size, n,
                B.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
                r, bk, bn, row_ids is not None, False, complex_input, acc, **launch)

    if m and n:
        if algorithm == "csr_row_tile":
            rows(None, True)
            counts["tile_rows"] = m
        elif algorithm == "csr_row_kparallel":
            rows(None)
            counts["kparallel_rows"] = m
        else:
            with phases.measure("process_gpu_ms"):
                lengths = prepared.kernel_indptr[1:] - prepared.kernel_indptr[:-1]
                if algorithm == "csr_adaptive_tile_split":
                    short, medium, long = _classify(
                        lengths, cfg["short_row_threshold"], cfg["split_row_threshold"])
                else:
                    short, medium, long = _classify(lengths, cfg["segment_nnz"], cfg["segment_nnz"])
                    medium = short
                    short = short[:0]
            rows(short, True)
            rows(medium)
            counts.update(tile_rows=short.numel(), kparallel_rows=medium.numel(), split_rows=long.numel())
            if long.numel():
                _split(prepared, A, B, C, long, cfg, phases, stats, complex_input, acc, launch)
    meta = dict(config_meta, config=dict(cfg), algorithm=algorithm,
                implementation_version=policy.IMPLEMENTATION_VERSION,
                compute_dtype="float64" if acc == tl.float64 else "float32",
                complex_component_dtype=("float64" if acc == tl.float64 else "float32") if complex_input else None,
                validation="unverified", process_cpu_ms=0.0,
                dense_layout=dense_layout, b_stride=tuple(B.stride()), c_stride=tuple(C.stride()),
                output_layout=dense_layout, **counts, **stats)
    meta.update(phases.results())
    if diagnostics:
        meta["diagnostics"] = {**counts, **stats, "config": dict(cfg),
                               "launch_config_scope": "all_paths", "num_warps": cfg["num_warps"],
                               "num_stages": cfg["num_stages"]}
    return C, meta


def _split(prepared, A, B, C, rows, cfg, phases, stats, complex_input, acc, launch):
    n = B.shape[1]
    bn = cfg["block_n"]
    element_bytes = B.element_size()
    # Three equally sized numeric slabs bound X/Y plus allocation transition.
    wave, capacity = policy.workspace_geometry(n, element_bytes, cfg)
    with phases.measure("process_gpu_ms"):
        starts = prepared.kernel_indptr[rows]
        lengths = prepared.kernel_indptr[rows + 1] - starts
        segment_counts = torch.div(lengths + cfg["segment_nnz"] - 1, cfg["segment_nnz"], rounding_mode="floor")
        offsets = _prefix(segment_counts)
        total = int(offsets[-1].item())
    stats["segment_count"] = total
    stats["descriptor_peak_bytes_estimate"] = (starts.numel() * 3 + offsets.numel()) * 8
    for col in range(0, n, wave):
        width = min(wave, n - col)
        for first in range(0, total, capacity):
            last = min(total, first + capacity)
            size = last - first
            stats["segment_batches"] += 1
            with phases.measure("process_gpu_ms"):
                ids = torch.arange(first, last, device=B.device, dtype=torch.int64)
                owners = torch.searchsorted(offsets[1:], ids, right=True)
                seg_rows = rows[owners]
                seg_start = starts[owners] + (ids - offsets[owners]) * cfg["segment_nnz"]
                seg_end = torch.minimum(seg_start + cfg["segment_nnz"], prepared.kernel_indptr[seg_rows + 1])
                local_offsets = offsets.clamp(first, last) - first
                local_counts = local_offsets[1:] - local_offsets[:-1]
                # Full-row descriptor arrays are separate from the numeric workspace.
                stats["descriptor_peak_bytes_estimate"] = max(stats["descriptor_peak_bytes_estimate"],
                    (starts.numel() * 3 + offsets.numel() * 4 + size * 5) * 8)
            with phases.measure("compute_ms"):
                partial = torch.empty((size, width), device=B.device, dtype=B.dtype)
                rows_kernel[(size, triton.cdiv(width, bn))](
                    _view(A), prepared.kernel_indices, prepared.kernel_indptr, _view(B), _view(partial),
                    seg_rows, seg_start, seg_end, size, width, B.stride(0), B.stride(1), width, 1, col,
                    1, cfg["block_k"], bn, True, True, complex_input, acc, **launch)
            peak = partial.numel() * element_bytes
            levels = 0
            while True:
                with phases.measure("process_gpu_ms"):
                    maximum = int(local_counts.max().item())
                if maximum <= 1:
                    break
                with phases.measure("process_gpu_ms"):
                    next_counts = torch.div(local_counts + cfg["reduce_block_size"] - 1,
                                            cfg["reduce_block_size"], rounding_mode="floor")
                    next_offsets = _prefix(next_counts)
                    next_size = int(next_offsets[-1].item())
                    group_ids = torch.arange(next_size, device=B.device, dtype=torch.int64)
                    group_rows = torch.searchsorted(next_offsets[1:], group_ids, right=True)
                    reduce_starts = local_offsets[group_rows] + (group_ids - next_offsets[group_rows]) * cfg["reduce_block_size"]
                    reduce_ends = torch.minimum(reduce_starts + cfg["reduce_block_size"], local_offsets[group_rows + 1])
                with phases.measure("compute_ms"):
                    reduced = torch.empty((next_size, width), device=B.device, dtype=B.dtype)
                    peak = max(peak, (partial.numel() + reduced.numel()) * element_bytes)
                    reduce_kernel[(next_size, triton.cdiv(width, bn))](
                        _view(partial), _view(reduced), reduce_starts, reduce_ends, width,
                        cfg["reduce_block_size"], bn, complex_input, acc, **launch)
                    partial = reduced
                    del reduced
                local_counts, local_offsets = next_counts, next_offsets
                levels += 1
            with phases.measure("compute_ms"):
                finish_kernel[(rows.numel(), triton.cdiv(width, bn))](
                    _view(partial), _view(C), rows, local_offsets, local_counts, width, col,
                    C.stride(0), C.stride(1), bn, complex_input, **launch)
                del partial
            stats["workspace_peak_bytes"] = max(stats["workspace_peak_bytes"], peak)
            stats["reduction_levels"] = max(stats["reduction_levels"], levels)
