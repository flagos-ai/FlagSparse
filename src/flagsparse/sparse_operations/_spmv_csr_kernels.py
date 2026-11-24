"""Native CSR kernels. Runtime plans are built afresh on every invocation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _window_order(length, local_id, W: tl.constexpr, LOG_W: tl.constexpr):
    lane = tl.arange(0, W)
    for level in tl.static_range(1, LOG_W + 1):
        for shift in tl.static_range(level - 1, -1, -1):
            partner = lane ^ (1 << shift)
            other_length = tl.gather(length, partner, 0)
            other_id = tl.gather(local_id, partner, 0)
            other_first = (other_length > length) | ((other_length == length) & (other_id < local_id))
            self_first = (length > other_length) | ((length == other_length) & (local_id < other_id))
            take_first = ((lane & (1 << level)) == 0) == ((lane & (1 << shift)) == 0)
            swap = tl.where(take_first, other_first, self_first)
            length = tl.where(swap, other_length, length)
            local_id = tl.where(swap, other_id, local_id)
    return local_id


@triton.jit
def _csr_direct_store(Y, row, re, im, valid, YS, COMPLEX: tl.constexpr):
    offset = row.to(tl.int64) * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * offset, re, valid)
        tl.store(Y + 2 * offset + 1, im, valid)
    else:
        tl.store(Y + offset, re, valid)


@triton.jit
def window_grouped_reduce_kernel(A, I, P, X, Y, M, XS, YS,
                                 W: tl.constexpr, R: tl.constexpr, V: tl.constexpr,
                                 LOG_W: tl.constexpr, SORT: tl.constexpr,
                                 COMPLEX: tl.constexpr, ACC: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * W
    local_id = tl.arange(0, W)
    if SORT:
        rows = base + local_id.to(tl.int64)
        begin = tl.load(P + rows, rows < M, 0).to(tl.int64)
        end = tl.load(P + rows + 1, rows < M, 0).to(tl.int64)
        length = tl.where(rows < M, end - begin, -1)
        local_id = _window_order(length, local_id, W, LOG_W)
    lanes = tl.arange(0, V).to(tl.int64)
    for group in range(W // R):
        offsets = group * R + tl.arange(0, R)
        row = base + tl.gather(local_id, offsets, 0).to(tl.int64)
        valid = row < M
        start = tl.load(P + row, valid, 0).to(tl.int64)
        stop = tl.load(P + row + 1, valid, 0).to(tl.int64)
        re = tl.zeros((R, V), ACC)
        im = tl.zeros((R, V), ACC)
        for step in range(tl.cdiv(tl.max(stop - start, 0), V)):
            pos = start[:, None] + step * V + lanes[None, :]
            mask = valid[:, None] & (pos < stop[:, None])
            pr, pi = _short_row_product(A, I, X, pos, mask, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
        _csr_direct_store(Y, row, tl.sum(re, 1), tl.sum(im, 1), valid, YS, COMPLEX)


@triton.jit
def _merge_event_corank(P, D, M, Z, WIDTH: tl.constexpr):
    # Stable merge: row-end events precede NNZ events at equal positions.
    diagonal = D.to(tl.int64)
    lo = tl.maximum(diagonal - Z, 0)
    hi = tl.minimum(diagonal, M)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        active = lo < hi
        mid = (lo + hi) // 2
        boundary = tl.load(P + mid + 1, active & (mid < M), 0).to(tl.int64)
        right = boundary < diagonal - mid
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    return lo, diagonal - lo


@triton.jit
def _merge_event_scan(lr, li, la, lh, rr, ri, ra, rh):
    return (tl.where(rh, rr, lr + rr), tl.where(rh, ri, li + ri),
            tl.where(rh, ra, la | ra), lh | rh)


@triton.jit
def merge_event_atomic_kernel(A, I, P, X, Y, M, Z, XS, YS,
                              C: tl.constexpr, L: tl.constexpr,
                              COMPLEX: tl.constexpr, ACC: tl.constexpr):
    Q: tl.constexpr = C * L
    start_d = tl.program_id(0).to(tl.int64) * Q
    total = tl.cast(M, tl.int64) + tl.cast(Z, tl.int64)
    edges = tl.minimum(start_d + tl.arange(0, 2).to(tl.int64) * Q, total)
    edge_r, edge_p = _merge_event_corank(P, edges, M, Z, 2)
    pb, pe = tl.min(edge_p, 0), tl.max(edge_p, 0)
    pos = pb + tl.arange(0, Q).to(tl.int64)
    prod_r, prod_i = _short_row_product(A, I, X, pos, pos < pe, XS, COMPLEX, ACC)
    diagonal = tl.minimum(start_d + tl.arange(0, C).to(tl.int64) * L, total)
    row, p = _merge_event_corank(P, diagonal, M, Z, C)
    p0 = p
    first_owner, last_owner = row, row
    seen = tl.full((C,), False, tl.int1)
    has = tl.full((C,), False, tl.int1)
    re, im = tl.zeros((C,), ACC), tl.zeros((C,), ACC)
    first_re, first_im = tl.zeros((C,), ACC), tl.zeros((C,), ACC)
    first_active = tl.full((C,), False, tl.int1)
    for step in tl.static_range(0, L):
        active = diagonal + step < total
        stop = tl.load(P + row + 1, active & (row < M), 0).to(tl.int64)
        boundary = active & (row < M) & (stop <= p)
        nz = active & ~boundary & (p < Z)
        row_start = tl.load(P + row, boundary & has, 0).to(tl.int64)
        complete = boundary & has & (row_start >= p0)
        _csr_direct_store(Y, row, re, im, complete, YS, COMPLEX)
        fragment = boundary & has & ~complete
        first_re = tl.where(fragment, re, first_re)
        first_im = tl.where(fragment, im, first_im)
        first_active = first_active | fragment
        first_owner = tl.where(nz & ~seen, row, first_owner)
        last_owner = tl.where(nz, row, last_owner)
        seen = seen | nz
        offset = tl.minimum(tl.maximum(p - pb, 0), Q - 1).to(tl.int32)
        value_r = tl.gather(prod_r, offset, 0)
        value_i = tl.gather(prod_i, offset, 0)
        re = tl.where(boundary, 0.0, re) + tl.where(nz, value_r, 0.0)
        im = tl.where(boundary, 0.0, im) + tl.where(nz, value_i, 0.0)
        has = (has & ~boundary) | nz
        row = row + boundary.to(tl.int64)
        p = p + nz.to(tl.int64)
    start = tl.load(P + row, has & (row < M), 0).to(tl.int64)
    stop = tl.load(P + row + 1, has & (row < M), 0).to(tl.int64)
    full = has & (start >= p0) & (stop <= p)
    _csr_direct_store(Y, row, re, im, full, YS, COMPLEX)
    tail = has & ~full
    # If a strip contains only part of one row, keep it in its first slot.
    to_first = tail & (first_owner == row)
    first_re = tl.where(to_first, re, first_re)
    first_im = tl.where(to_first, im, first_im)
    first_active = first_active | to_first
    owner = tl.reshape(tl.join(first_owner, last_owner), (2 * C,))
    rr = tl.reshape(tl.join(first_re, tl.where(tail & ~to_first, re, 0.0)), (2 * C,))
    ri = tl.reshape(tl.join(first_im, tl.where(tail & ~to_first, im, 0.0)), (2 * C,))
    live = tl.reshape(tl.join(first_active, tail & ~to_first), (2 * C,))
    slot = tl.arange(0, 2 * C)
    previous = tl.gather(owner, tl.maximum(slot - 1, 0), 0)
    following = tl.gather(owner, tl.minimum(slot + 1, 2 * C - 1), 0)
    head = (slot == 0) | (owner != previous)
    sr, si, sa, _ = tl.associative_scan((rr, ri, live, head), 0, _merge_event_scan)
    emit = sa & (owner < M) & ((slot == 2 * C - 1) | (owner != following))
    row_begin = tl.load(P + owner, emit, 0).to(tl.int64)
    row_end = tl.load(P + owner + 1, emit, 0).to(tl.int64)
    _nnz_local_write(Y, owner, sr, si, emit, (row_begin >= pb) & (row_end <= pe), YS, COMPLEX, True)


def run_window_or_merge(prepared, x, out, timing, phase):
    """Independent window ordering / event-merge routes; no reusable plans."""
    cfg = prepared.config
    merge = prepared.alg == "csr_merge_event_atomic"
    component = torch.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else torch.float32
    complex_input = prepared.data.is_complex()
    dtype = (torch.complex128 if component == torch.float64 else torch.complex64) if complex_input else component
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    extra = prepared.n_rows * (16 if dtype == torch.complex128 else 8 if dtype in (torch.float64, torch.complex64) else 4) if merge and y.dtype != dtype else 0
    if merge and extra > cfg["workspace_limit_bytes"]:
        raise ValueError(f"merge event numeric workspace {extra} exceeds workspace_limit_bytes")
    if prepared.n_rows > (1 << 63) - 1 - prepared.data.numel():
        raise ValueError("CSR row/NNZ event count exceeds int64")
    def compute():
        data, vector = prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg()
        target = torch.empty(prepared.n_rows, dtype=dtype, device=y.device) if extra else y
        view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
        args = (view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(target))
        launch = dict(COMPLEX=complex_input, ACC=tl.float64 if component == torch.float64 else tl.float32,
                      num_warps=cfg["num_warps"], num_stages=1, enable_fp_fusion=False)
        if merge:
            target.zero_()
            if data.numel():
                q = cfg["strips_per_program"] * cfg["events_per_strip"]
                merge_event_atomic_kernel[(triton.cdiv(prepared.n_rows + data.numel(), q),)](
                    *args, prepared.n_rows, data.numel(), vector.stride(0), target.stride(0),
                    C=cfg["strips_per_program"], L=cfg["events_per_strip"], **launch)
        elif prepared.n_rows:
            w = cfg["window_rows"]
            window_grouped_reduce_kernel[(triton.cdiv(prepared.n_rows, w),)](
                *args, prepared.n_rows, vector.stride(0), target.stride(0), W=w,
                LOG_W=w.bit_length() - 1, R=cfg["rows_per_group"], V=cfg["lanes_per_row"],
                SORT=cfg["row_order"] == "length_desc", **launch)
        if target is not y:
            y.copy_(target)
    _, compute_ms = phase(compute, timing)
    meta = dict(process_cpu_ms=0.0, process_gpu_ms=0.0 if timing else None, compute_ms=compute_ms,
                execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                plan_strategy="merge_event_in_kernel" if merge else "window_order_in_kernel",
                workspace_peak_bytes=extra, descriptor_peak_bytes=0,
                output_buffer_identity="compute_precision_buffer" if extra else "output",
                initialization="compute_target_zero" if merge else "kernel_writes_all_rows",
                output_conversion="final_copy_cast" if extra else "final_store",
                x_stride=tuple(x.stride()), y_stride=tuple(y.stride()))
    if merge:
        meta.update(event_count=prepared.n_rows + prepared.data.numel(),
                    events_per_program=cfg["strips_per_program"] * cfg["events_per_strip"],
                    atomic_scope="cross_program_row_fragments")
    else:
        meta.update(local_ordering_scope="inside_compute_kernel", row_order=cfg["row_order"],
                    compute_timing_scope="local_order_and_numeric_compute",
                    fused_plan_compute_ms=compute_ms)
    return y, meta


@triton.jit
def _nnz_local_owner(P, POS, VALID, LOW, HIGH, WIDTH: tl.constexpr):
    lo = tl.full((WIDTH,), 0, tl.int64) + LOW
    hi = tl.full((WIDTH,), 0, tl.int64) + HIGH
    while tl.sum(((lo < hi) & VALID).to(tl.int32), 0) > 0:
        active = (lo < hi) & VALID
        mid = (lo + hi + 1) // 2
        boundary = tl.load(P + mid, active, 0).to(tl.int64)
        right = boundary <= POS
        lo = tl.where(active & right, mid, lo)
        hi = tl.where(active & ~right, mid - 1, hi)
    return lo


@triton.jit
def _nnz_local_scan(lr, li, lh, rr, ri, rh):
    return tl.where(rh, rr, lr + rr), tl.where(rh, ri, li + ri), lh | rh


@triton.jit
def _nnz_local_write(Y, ROW, RE, IM, MASK, FULL, YS,
                     COMPLEX: tl.constexpr, BOUNDARY: tl.constexpr):
    offset = ROW.to(tl.int64) * tl.cast(YS, tl.int64)
    direct = MASK & FULL if BOUNDARY else MASK & False
    atomic = MASK & ~direct
    if COMPLEX:
        tl.store(Y + 2 * offset, RE, direct)
        tl.store(Y + 2 * offset + 1, IM, direct)
        tl.atomic_add(Y + 2 * offset, RE, atomic, sem="relaxed")
        tl.atomic_add(Y + 2 * offset + 1, IM, atomic, sem="relaxed")
    else:
        tl.store(Y + offset, RE, direct)
        tl.atomic_add(Y + offset, RE, atomic, sem="relaxed")


@triton.jit
def nnz_local_reduce_atomic_kernel(A, I, P, X, Y, M, NNZ, XS, YS,
                                   T: tl.constexpr, ACC: tl.constexpr,
                                   COMPLEX: tl.constexpr, BOUNDED: tl.constexpr,
                                   FAST: tl.constexpr, BOUNDARY: tl.constexpr):
    begin = tl.program_id(0).to(tl.int64) * T
    end = tl.minimum(begin + T, NNZ)
    lane = tl.arange(0, T).to(tl.int64)
    pos = begin + lane
    valid = pos < end
    if FAST or BOUNDED:
        edges = tl.where(tl.arange(0, 2) == 0, begin, end - 1)
        owners = _nnz_local_owner(P, edges, tl.full((2,), True, tl.int1), 0, M - 1, 2)
        first, last = tl.min(owners, 0), tl.max(owners, 0)
    else:
        first, last = tl.full((), 0, tl.int64), tl.cast(M, tl.int64) - 1
    re, im = _short_row_product(A, I, X, pos, valid, XS, COMPLEX, ACC)
    if FAST & (first == last):
        # Runtime branches must not merge scalar and [T] SSA values under the
        # same name (Triton 3.6 rejects that even when they are branch-local).
        single_start = tl.load(P + first).to(tl.int64)
        single_stop = tl.load(P + first + 1).to(tl.int64)
        _nnz_local_write(Y, first, tl.sum(re, 0), tl.sum(im, 0),
                         tl.full((), True, tl.int1), (single_start >= begin) & (single_stop <= end),
                         YS, COMPLEX, BOUNDARY)
    else:
        if BOUNDED:
            row = _nnz_local_owner(P, pos, valid, first, last, T)
        else:
            row = _nnz_local_owner(P, pos, valid, 0, M - 1, T)
        row_starts = tl.load(P + row, valid, 0).to(tl.int64)
        row_stops = tl.load(P + row + 1, valid, 0).to(tl.int64)
        head = (lane == 0) | (pos == row_starts) | ~valid
        sr, si, _ = tl.associative_scan((re, im, head), 0, _nnz_local_scan)
        tail = valid & ((pos + 1 == row_stops) | (pos + 1 == end))
        _nnz_local_write(Y, row, sr, si, tail, (row_starts >= begin) & (row_stops <= end),
                         YS, COMPLEX, BOUNDARY)


def run_nnz_local_reduce(prepared, x, out, timing, phase):
    cfg = prepared.config
    complex_input = prepared.data.is_complex()
    component = torch.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else torch.float32
    dtype = (torch.complex128 if component == torch.float64 else torch.complex64) if complex_input else component
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def compute():
        target = y if y.dtype == dtype else torch.empty(prepared.n_rows, dtype=dtype, device=y.device)
        target.zero_()
        data, vector = prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg()
        view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
        if data.numel():
            nnz_local_reduce_atomic_kernel[(triton.cdiv(data.numel(), cfg["block_nnz"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(target),
                prepared.n_rows, data.numel(), vector.stride(0), target.stride(0),
                T=cfg["block_nnz"], ACC=tl.float64 if component == torch.float64 else tl.float32,
                COMPLEX=complex_input, BOUNDED=cfg["owner_search"] == "bounded",
                FAST=cfg["single_row_fastpath"], BOUNDARY=cfg["atomic_mode"] == "boundary",
                num_warps=cfg["num_warps"], num_stages=1, enable_fp_fusion=False)
        if target is not y:
            y.copy_(target)
        return target is y
    direct, compute_ms = phase(compute, timing)
    return y, dict(process_cpu_ms=0.0, process_gpu_ms=0.0 if timing else None, compute_ms=compute_ms,
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   nnz_tile_count=triton.cdiv(prepared.data.numel(), cfg["block_nnz"]),
                   plan_strategy="none", descriptor_peak_bytes=0,
                   workspace_peak_bytes=0 if direct else prepared.n_rows * (16 if dtype == torch.complex128 else 8 if dtype in (torch.float64, torch.complex64) else 4),
                   output_buffer_identity="output" if direct else "compute_precision_buffer",
                   initialization="compute_target_zero", output_conversion="none" if direct else "final_copy_cast",
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()))


@triton.jit
def short_rowqueue_producer_kernel(A, I, P, X, Y, COUNT, ROWS, M, XS, YS,
                                   R: tl.constexpr, S: tl.constexpr,
                                   COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    starts = tl.load(P + row, valid, 0).to(tl.int64)
    stops = tl.load(P + row + 1, valid, 0).to(tl.int64)
    queued = valid & (stops - starts > S)
    flags = queued.to(tl.int32)
    count = tl.sum(flags, 0)
    if count > 0:
        first_slot = tl.atomic_add(COUNT, count, sem="relaxed").to(tl.int64)
        slots = first_slot + tl.cumsum(flags, 0).to(tl.int64) - 1
        tl.store(ROWS + slots, row, queued)
    ordinary = valid & ~queued
    steps = tl.max(tl.where(ordinary, stops - starts, 0), 0)
    re = tl.zeros((R,), ACC)
    im = tl.zeros((R,), ACC)
    for step in range(steps):
        pos = starts + step
        pr, pi = _short_row_product(A, I, X, pos, ordinary & (pos < stops), XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    target = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * target, re, ordinary)
        tl.store(Y + 2 * target + 1, im, ordinary)
    else:
        tl.store(Y + target, re, ordinary)


@triton.jit
def short_rowqueue_vector_kernel(A, I, P, X, Y, COUNT, ROWS, XS, YS,
                                 K: tl.constexpr, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    slot = tl.program_id(0).to(tl.int64)
    # The producer launch has finished on this stream: no global spin barrier
    # or integer task-acquisition atomic is needed in the consumer.
    if slot < tl.load(COUNT).to(tl.int64):
        row = tl.load(ROWS + slot).to(tl.int64)
        start = tl.load(P + row).to(tl.int64)
        stop = tl.load(P + row + 1).to(tl.int64)
        lane = tl.arange(0, K).to(tl.int64)
        re = tl.zeros((K,), ACC)
        im = tl.zeros((K,), ACC)
        for base in range(start, stop, K):
            pos = base + lane
            pr, pi = _short_row_product(A, I, X, pos, pos < stop, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
        target = row * tl.cast(YS, tl.int64)
        if COMPLEX:
            tl.store(Y + 2 * target, tl.sum(re, 0))
            tl.store(Y + 2 * target + 1, tl.sum(im, 0))
        else:
            tl.store(Y + target, tl.sum(re, 0))


def run_short_rowqueue(prepared, x, out, timing, phase):
    from ._spmv_csr_config import short_rowqueue_capacity

    cfg = prepared.config
    stats = short_rowqueue_capacity(prepared.n_rows, prepared.data.numel(), cfg)
    capacity = stats["row_queue_capacity"]
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.empty(capacity, dtype=torch.int64, device=x.device))
    (data, vector, count, rows), init_ms = phase(initialize, timing)
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            short_rowqueue_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                count, rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], S=cfg["short_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    def consume():
        if capacity:
            short_rowqueue_vector_kernel[(capacity,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                count, rows, vector.stride(0), y.stride(0),
                K=cfg["block_nnz"], num_warps=cfg["num_warps"], **common)
    _, consumer_ms = phase(consume, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + consumer_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms, consumer_ms=consumer_ms,
                   finish_ms=0.0 if timing else None, fused_plan_compute_ms=producer_ms,
                   queue_count_readback=False, plan_strategy="per_run_row_ids_only",
                   task_acquisition="static_capacity_grid", floating_point_atomic=False,
                   initialization="queue_counter_only", output_conversion="final_store",
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def short_rowqueue_strided_vector_kernel(A, I, P, X, Y, COUNT, ROWS, XS, YS,
                                         K: tl.constexpr, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    # Producer completion on the same stream makes COUNT and every queued row
    # visible. Each slot belongs to exactly one program modulo grid_stride.
    slot = tl.program_id(0).to(tl.int64)
    grid_stride = tl.num_programs(0).to(tl.int64)
    count = tl.load(COUNT).to(tl.int64)
    lane = tl.arange(0, K).to(tl.int64)
    while slot < count:
        row = tl.load(ROWS + slot).to(tl.int64)
        start = tl.load(P + row).to(tl.int64)
        stop = tl.load(P + row + 1).to(tl.int64)
        # Reset both accumulators for every whole row, including later rounds.
        re = tl.zeros((K,), ACC)
        im = tl.zeros((K,), ACC)
        for base in range(start, stop, K):
            pos = base + lane
            pr, pi = _short_row_product(A, I, X, pos, pos < stop, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
        target = row * tl.cast(YS, tl.int64)
        if COMPLEX:
            tl.store(Y + 2 * target, tl.sum(re, 0))
            tl.store(Y + 2 * target + 1, tl.sum(im, 0))
        else:
            tl.store(Y + target, tl.sum(re, 0))
        slot += grid_stride


def run_short_rowqueue_strided(prepared, x, out, timing, phase):
    from ._spmv_csr_config import short_rowqueue_capacity

    cfg = prepared.config
    stats = short_rowqueue_capacity(prepared.n_rows, prepared.data.numel(), cfg)
    capacity = stats["row_queue_capacity"]
    grid = min(capacity, cfg["grid_blocks"])
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.empty(capacity, dtype=torch.int64, device=x.device))
    (data, vector, count, rows), init_ms = phase(initialize, timing)
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            short_rowqueue_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                count, rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], S=cfg["short_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    def consume():
        if grid:
            short_rowqueue_strided_vector_kernel[(grid,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                count, rows, vector.stride(0), y.stride(0),
                K=cfg["block_nnz"], num_warps=cfg["num_warps"], **common)
    _, consumer_ms = phase(consume, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + consumer_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms, consumer_ms=consumer_ms,
                   finish_ms=0.0 if timing else None, fused_plan_compute_ms=producer_ms,
                   queue_count_readback=False, plan_strategy="per_run_row_ids_only",
                   task_acquisition="static_grid_stride", floating_point_atomic=False,
                   consumer_grid_blocks=grid, consumer_grid_limit=cfg["grid_blocks"],
                   consumer_rounds_capacity=triton.cdiv(capacity, grid) if grid else 0,
                   initialization="queue_counter_only", output_conversion="final_store",
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def cooperative_row_producer_kernel(A, I, P, X, Y, COUNT, ROWS, M, XS, YS,
                                     R: tl.constexpr, V: tl.constexpr, L: tl.constexpr,
                                     COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    stop = tl.load(P + row + 1, valid, 0).to(tl.int64)
    queued = valid & (stop - start > L)
    flags = queued.to(tl.int32)
    count = tl.sum(flags, 0)
    if count > 0:
        first = tl.atomic_add(COUNT, count, sem="relaxed").to(tl.int64)
        slots = first + tl.cumsum(flags, 0).to(tl.int64) - 1
        tl.store(ROWS + slots, row, queued)
    ordinary = valid & ~queued
    length = tl.max(tl.where(ordinary, stop - start, 0), 0)
    lane = tl.arange(0, V).to(tl.int64)
    re = tl.zeros((R, V), ACC)
    im = tl.zeros((R, V), ACC)
    for base in range(0, length, V):
        pos = start[:, None] + base + lane[None, :]
        pr, pi = _short_row_product(A, I, X, pos, ordinary[:, None] & (pos < stop[:, None]),
                                     XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    target = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * target, tl.sum(re, 1), ordinary)
        tl.store(Y + 2 * target + 1, tl.sum(im, 1), ordinary)
    else:
        tl.store(Y + target, tl.sum(re, 1), ordinary)


@triton.jit
def cooperative_row_consumer_kernel(A, I, P, X, COUNT, ROWS, PARTIAL, XS,
                                     K: tl.constexpr, PARTS: tl.constexpr,
                                     COMPLEX: tl.constexpr, ACC: tl.constexpr):
    slot = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64)
    part = tl.program_id(1).to(tl.int64)
    count = tl.load(COUNT).to(tl.int64)
    lane = tl.arange(0, K).to(tl.int64)
    while slot < count:
        row = tl.load(ROWS + slot).to(tl.int64)
        start = tl.load(P + row).to(tl.int64)
        stop = tl.load(P + row + 1).to(tl.int64)
        # Divide K-sized chunks into balanced, contiguous ranges. Even empty
        # ranges write zero to their exclusive slot before the finish launch.
        chunks = (stop - start + K - 1) // K
        begin = start + ((chunks * part) // PARTS) * K
        end = tl.minimum(stop, start + ((chunks * (part + 1)) // PARTS) * K)
        re = tl.zeros((K,), ACC)
        im = tl.zeros((K,), ACC)
        for base in range(begin, end, K):
            pos = base + lane
            pr, pi = _short_row_product(A, I, X, pos, pos < end, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
        target = slot * PARTS + part
        if COMPLEX:
            tl.store(PARTIAL + 2 * target, tl.sum(re, 0))
            tl.store(PARTIAL + 2 * target + 1, tl.sum(im, 0))
        else:
            tl.store(PARTIAL + target, tl.sum(re, 0))
        slot += stride


@triton.jit
def cooperative_row_finish_kernel(COUNT, ROWS, PARTIAL, Y, YS,
                                   PARTS: tl.constexpr, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    slot = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64)
    count = tl.load(COUNT).to(tl.int64)
    part = tl.arange(0, PARTS).to(tl.int64)
    while slot < count:
        row = tl.load(ROWS + slot).to(tl.int64)
        source = slot * PARTS + part
        target = row * tl.cast(YS, tl.int64)
        if COMPLEX:
            re = tl.load(PARTIAL + 2 * source).to(ACC)
            im = tl.load(PARTIAL + 2 * source + 1).to(ACC)
            tl.store(Y + 2 * target, tl.sum(re, 0))
            tl.store(Y + 2 * target + 1, tl.sum(im, 0))
        else:
            re = tl.load(PARTIAL + source).to(ACC)
            tl.store(Y + target, tl.sum(re, 0))
        slot += stride


def run_longrow_cooperative(prepared, x, out, timing, phase):
    from ._spmv_csr_config import cooperative_queue_capacity

    cfg = prepared.config
    stats = cooperative_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)
    capacity, parts = stats["row_queue_capacity"], cfg["cooperative_parts"]
    grid = min(capacity, cfg["grid_blocks"])
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    component = torch.float64 if fp64 else torch.float32
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.empty(capacity, dtype=torch.int64, device=x.device),
                torch.empty(capacity * parts * (2 if complex_input else 1), dtype=component, device=x.device))
    (data, vector, count, rows, partial), init_ms = phase(initialize, timing)
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            cooperative_row_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                count, rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], V=cfg["lanes_per_row"], L=cfg["long_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    def consume():
        if grid:
            cooperative_row_consumer_kernel[(grid, parts)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), count, rows,
                partial, vector.stride(0), K=cfg["block_nnz"], PARTS=parts,
                num_warps=cfg["num_warps"], **common)
    _, consumer_ms = phase(consume, timing)
    def finish():
        if grid:
            cooperative_row_finish_kernel[(grid,)](
                count, rows, partial, view(y), y.stride(0), PARTS=parts,
                num_warps=cfg["producer_num_warps"], **common)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + consumer_ms + finish_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms, consumer_ms=consumer_ms, finish_ms=finish_ms,
                   fused_plan_compute_ms=producer_ms, queue_count_readback=False,
                   plan_strategy="per_run_row_ids_fixed_parts", task_acquisition="static_grid_stride_parts",
                   floating_point_atomic=False, cooperative_parts=parts, row_grid_blocks=grid,
                   consumer_grid_blocks=grid * parts, consumer_grid_limit=cfg["grid_blocks"] * parts,
                   consumer_rounds_capacity=triton.cdiv(capacity, grid) if grid else 0,
                   initialization="queue_counter_only", output_conversion="final_store",
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def three_tier_producer_kernel(A, I, P, X, Y, COUNTS, MEDIUM, LONG, M, XS, YS,
                              R: tl.constexpr, S: tl.constexpr, L: tl.constexpr,
                              COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    stop = tl.load(P + row + 1, valid, 0).to(tl.int64)
    length = stop - start
    short = valid & (length <= S)
    medium = valid & (length > S) & (length <= L)
    long = valid & (length > L)
    medium_flags = medium.to(tl.int32)
    medium_count = tl.sum(medium_flags, 0)
    if medium_count > 0:
        medium_base = tl.atomic_add(COUNTS, medium_count, sem="relaxed").to(tl.int64)
        medium_slot = medium_base + tl.cumsum(medium_flags, 0).to(tl.int64) - 1
        tl.store(MEDIUM + medium_slot, row, medium)
    long_flags = long.to(tl.int32)
    long_count = tl.sum(long_flags, 0)
    if long_count > 0:
        long_base = tl.atomic_add(COUNTS + 1, long_count, sem="relaxed").to(tl.int64)
        long_slot = long_base + tl.cumsum(long_flags, 0).to(tl.int64) - 1
        tl.store(LONG + long_slot, row, long)
    steps = tl.max(tl.where(short, length, 0), 0)
    re = tl.zeros((R,), ACC)
    im = tl.zeros((R,), ACC)
    for step in range(steps):
        pos = start + step
        pr, pi = _short_row_product(A, I, X, pos, short & (pos < stop), XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    target = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * target, re, short)
        tl.store(Y + 2 * target + 1, im, short)
    else:
        tl.store(Y + target, re, short)


def run_three_tier_queue(prepared, x, out, timing, phase):
    from ._spmv_csr_config import three_tier_queue_capacity

    cfg = prepared.config
    stats = three_tier_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)
    medium_capacity, long_capacity = stats["medium_queue_capacity"], stats["long_queue_capacity"]
    parts = cfg["cooperative_parts"]
    long_grid = min(long_capacity, cfg["long_grid_blocks"])
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    component = torch.float64 if fp64 else torch.float32
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(2, dtype=torch.int32, device=x.device),
                torch.empty(medium_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity * parts * (2 if complex_input else 1), dtype=component, device=x.device))
    (data, vector, counts, medium_rows, long_rows, partial), init_ms = phase(initialize, timing)
    long_count = counts[1:]
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            three_tier_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, long_rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], S=cfg["short_row_threshold"], L=cfg["long_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    # Reuse unchanged whole-row and exclusive-part mathematical kernels. This
    # route never invokes an old run function or inherits another route's plan.
    def medium_compute():
        if medium_capacity:
            short_rowqueue_vector_kernel[(medium_capacity,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, vector.stride(0), y.stride(0), K=cfg["block_nnz"],
                num_warps=cfg["num_warps"], **common)
    _, medium_ms = phase(medium_compute, timing)
    def long_compute():
        if long_grid:
            cooperative_row_consumer_kernel[(long_grid, parts)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), long_count,
                long_rows, partial, vector.stride(0), K=cfg["block_nnz"], PARTS=parts,
                num_warps=cfg["num_warps"], **common)
    _, long_ms = phase(long_compute, timing)
    def finish():
        if long_grid:
            cooperative_row_finish_kernel[(long_grid,)](
                long_count, long_rows, partial, view(y), y.stride(0), PARTS=parts,
                num_warps=cfg["producer_num_warps"], **common)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + medium_ms + long_ms + finish_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms,
                   medium_consumer_ms=medium_ms, long_consumer_ms=long_ms,
                   consumer_ms=medium_ms + long_ms if timing else None, finish_ms=finish_ms,
                   fused_plan_compute_ms=producer_ms, queue_count_readback=False,
                   plan_strategy="per_run_three_tier_row_ids", floating_point_atomic=False,
                   task_acquisition="medium_capacity_grid_long_fixed_parts", cooperative_parts=parts,
                   medium_grid_blocks=medium_capacity, long_row_grid_blocks=long_grid,
                   long_consumer_grid_blocks=long_grid * parts,
                   initialization="two_queue_counters_only", output_conversion="final_store",
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   output_buffer_identity="out" if out is not None else "allocated")


def run_three_tier_bounded(prepared, x, out, timing, phase):
    from ._spmv_csr_config import three_tier_queue_capacity

    # Independent schedule: only medium rows use a bounded grid; long rows
    # retain exclusive fixed-part cooperation. No host count readback.
    cfg = prepared.config
    stats = three_tier_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)
    medium_capacity, long_capacity = stats["medium_queue_capacity"], stats["long_queue_capacity"]
    medium_grid = min(medium_capacity, cfg["medium_grid_blocks"])
    parts = cfg["cooperative_parts"]
    long_grid = min(long_capacity, cfg["long_grid_blocks"])
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    component = torch.float64 if fp64 else torch.float32
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(2, dtype=torch.int32, device=x.device),
                torch.empty(medium_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity * parts * (2 if complex_input else 1), dtype=component, device=x.device))
    (data, vector, counts, medium_rows, long_rows, partial), init_ms = phase(initialize, timing)
    long_count = counts[1:]
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            three_tier_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, long_rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], S=cfg["short_row_threshold"], L=cfg["long_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    # Reuse unchanged whole-row and exclusive-part mathematical kernels. This
    # route never invokes an old run function or inherits another route's plan.
    def medium_compute():
        if medium_grid:
            short_rowqueue_strided_vector_kernel[(medium_grid,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, vector.stride(0), y.stride(0), K=cfg["block_nnz"],
                num_warps=cfg["num_warps"], **common)
    _, medium_ms = phase(medium_compute, timing)
    def long_compute():
        if long_grid:
            cooperative_row_consumer_kernel[(long_grid, parts)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), long_count,
                long_rows, partial, vector.stride(0), K=cfg["block_nnz"], PARTS=parts,
                num_warps=cfg["num_warps"], **common)
    _, long_ms = phase(long_compute, timing)
    def finish():
        if long_grid:
            cooperative_row_finish_kernel[(long_grid,)](
                long_count, long_rows, partial, view(y), y.stride(0), PARTS=parts,
                num_warps=cfg["producer_num_warps"], **common)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + medium_ms + long_ms + finish_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms,
                   medium_consumer_ms=medium_ms, long_consumer_ms=long_ms,
                   consumer_ms=medium_ms + long_ms if timing else None, finish_ms=finish_ms,
                   fused_plan_compute_ms=producer_ms, queue_count_readback=False,
                   plan_strategy="per_run_three_tier_row_ids", floating_point_atomic=False,
                   task_acquisition="medium_bounded_grid_stride_long_fixed_parts", cooperative_parts=parts,
                   medium_grid_blocks=medium_grid, medium_grid_limit=cfg["medium_grid_blocks"],
                   medium_rounds_capacity=triton.cdiv(medium_capacity, medium_grid) if medium_grid else 0,
                   long_row_grid_blocks=long_grid,
                   long_consumer_grid_blocks=long_grid * parts,
                   initialization="two_queue_counters_only", output_conversion="final_store",
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, TASK, XS,
                           K: tl.constexpr, PARTS: tl.constexpr,
                           COMPLEX: tl.constexpr, ACC: tl.constexpr):
    slot = TASK // PARTS
    part = TASK % PARTS
    row = tl.load(ROWS + slot).to(tl.int64)
    start = tl.load(P + row).to(tl.int64)
    stop = tl.load(P + row + 1).to(tl.int64)
    chunks = (stop - start + K - 1) // K
    begin = start + ((chunks * part) // PARTS) * K
    end = tl.minimum(stop, start + ((chunks * (part + 1)) // PARTS) * K)
    lane = tl.arange(0, K).to(tl.int64)
    re = tl.zeros((K,), ACC)
    im = tl.zeros((K,), ACC)
    for base in range(begin, end, K):
        pos = base + lane
        pr, pi = _short_row_product(A, I, X, pos, pos < end, XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    # Empty parts still write zero; finish never reads uninitialized valid slots.
    if COMPLEX:
        tl.store(PARTIAL + 2 * TASK, tl.sum(re, 0))
        tl.store(PARTIAL + 2 * TASK + 1, tl.sum(im, 0))
    else:
        tl.store(PARTIAL + TASK, tl.sum(re, 0))


@triton.jit
def three_tier_ticket_consumer_kernel(A, I, P, X, COUNTS, ROWS, PARTIAL, XS,
                                       WORKERS: tl.constexpr, BATCH: tl.constexpr,
                                       STATIC: tl.constexpr, K: tl.constexpr,
                                       PARTS: tl.constexpr, COMPLEX: tl.constexpr,
                                       ACC: tl.constexpr):
    total = tl.load(COUNTS + 1).to(tl.int64) * PARTS
    task = tl.program_id(0).to(tl.int64)
    if STATIC:
        while task < total:
            _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, task, XS, K, PARTS, COMPLEX, ACC)
            task += WORKERS
    else:
        # Static first wave avoids initial contention. T <= W needs no atomic.
        if task < total:
            _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, task, XS, K, PARTS, COMPLEX, ACC)
        if total > WORKERS:
            first = WORKERS + tl.atomic_add(COUNTS + 2, BATCH, sem="relaxed", scope="gpu").to(tl.int64)
            while first < total:
                for offset in tl.static_range(0, BATCH):
                    current = first + offset
                    if current < total:
                        _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, current, XS, K, PARTS, COMPLEX, ACC)
                first = WORKERS + tl.atomic_add(COUNTS + 2, BATCH, sem="relaxed", scope="gpu").to(tl.int64)


@triton.jit
def three_tier_fused_static_consumer_kernel(A, I, P, X, Y, COUNTS, MEDIUM, LONG, PARTIAL,
                                             XS, YS, GM: tl.constexpr, WORKERS: tl.constexpr,
                                             K: tl.constexpr, PARTS: tl.constexpr,
                                             COMPLEX: tl.constexpr, ACC: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid < GM:
        slot = pid
        count = tl.load(COUNTS).to(tl.int64)
        lane = tl.arange(0, K).to(tl.int64)
        while slot < count:
            row = tl.load(MEDIUM + slot).to(tl.int64)
            start = tl.load(P + row).to(tl.int64)
            stop = tl.load(P + row + 1).to(tl.int64)
            re = tl.zeros((K,), ACC)
            im = tl.zeros((K,), ACC)
            for base in range(start, stop, K):
                pos = base + lane
                pr, pi = _short_row_product(A, I, X, pos, pos < stop, XS, COMPLEX, ACC)
                re += pr
                if COMPLEX:
                    im += pi
            target = row * tl.cast(YS, tl.int64)
            if COMPLEX:
                tl.store(Y + 2 * target, tl.sum(re, 0))
                tl.store(Y + 2 * target + 1, tl.sum(im, 0))
            else:
                tl.store(Y + target, tl.sum(re, 0))
            # Each domain uses its own stride, never the combined grid size.
            slot += GM
    else:
        task = pid - GM
        total = tl.load(COUNTS + 1).to(tl.int64) * PARTS
        while task < total:
            _ticket_exclusive_part(A, I, P, X, LONG, PARTIAL, task, XS, K, PARTS, COMPLEX, ACC)
            task += WORKERS


def run_three_tier_fused_static(prepared, x, out, timing, phase):
    from ._spmv_csr_config import fused_static_queue_capacity

    cfg = prepared.config
    stats = fused_static_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)
    medium_capacity, long_capacity = stats["medium_queue_capacity"], stats["long_queue_capacity"]
    medium_grid = min(medium_capacity, cfg["medium_grid_blocks"])
    parts = cfg["cooperative_parts"]
    long_grid = min(long_capacity, cfg["long_grid_blocks"])
    workers = stats["long_worker_programs"]
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    component = torch.float64 if fp64 else torch.float32
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(3, dtype=torch.int32, device=x.device),
                torch.empty(medium_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity * parts * (2 if complex_input else 1), dtype=component, device=x.device))
    (data, vector, counts, medium_rows, long_rows, partial), init_ms = phase(initialize, timing)
    long_count = counts[1:2]
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            three_tier_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, long_rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], S=cfg["short_row_threshold"], L=cfg["long_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    def consume():
        if medium_grid + workers:
            three_tier_fused_static_consumer_kernel[(medium_grid + workers,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, long_rows, partial, vector.stride(0), y.stride(0),
                GM=medium_grid, WORKERS=workers, K=cfg["block_nnz"], PARTS=parts,
                num_warps=cfg["num_warps"], **common)
    _, consumer_ms = phase(consume, timing)
    def finish():
        if long_grid:
            cooperative_row_finish_kernel[(long_grid,)](
                long_count, long_rows, partial, view(y), y.stride(0), PARTS=parts,
                num_warps=cfg["producer_num_warps"], **common)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + consumer_ms + finish_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms, consumer_ms=consumer_ms,
                   consumer_fused_ms=consumer_ms, medium_consumer_ms=None, long_consumer_ms=None,
                   consumer_timing_scope="joint_medium_long", finish_ms=finish_ms,
                   fused_plan_compute_ms=producer_ms, queue_count_readback=False,
                   plan_strategy="per_run_three_tier_row_ids", floating_point_atomic=False,
                   task_acquisition="partitioned_static_domains", long_schedule="static",
                   cooperative_parts=parts, medium_grid_blocks=medium_grid,
                   medium_grid_limit=cfg["medium_grid_blocks"],
                   medium_rounds_capacity=triton.cdiv(medium_capacity, medium_grid) if medium_grid else 0,
                   long_row_grid_blocks=long_grid, long_consumer_grid_blocks=workers,
                   initialization="two_queue_counters_and_unused_control_slot", output_conversion="final_store",
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def _diagnostic_part_chunks(P, ROWS, TASK, K: tl.constexpr, PARTS: tl.constexpr):
    row = tl.load(ROWS + TASK // PARTS).to(tl.int64)
    length = tl.load(P + row + 1).to(tl.int64) - tl.load(P + row).to(tl.int64)
    chunks = (length + K - 1) // K
    part = TASK % PARTS
    return chunks * (part + 1) // PARTS - chunks * part // PARTS


@triton.jit
def three_tier_ticket_trace_kernel(A, I, P, X, COUNTS, ROWS, PARTIAL, XS, TRACE,
                                    WORKERS: tl.constexpr, BATCH: tl.constexpr,
                                    STATIC: tl.constexpr, K: tl.constexpr, PARTS: tl.constexpr,
                                    COMPLEX: tl.constexpr, ACC: tl.constexpr):
    """Instrumented diagnostic only; never launched by the normal run route."""
    total = tl.load(COUNTS + 1).to(tl.int64) * PARTS
    pid = tl.program_id(0).to(tl.int64)
    task = pid
    done = tl.full((), 0, tl.int64)
    work = tl.full((), 0, tl.int64)
    maximum = tl.full((), 0, tl.int64)
    claims = tl.full((), 0, tl.int64)
    first_work = tl.full((), 0, tl.int64)
    if STATIC:
        while task < total:
            _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, task, XS, K, PARTS, COMPLEX, ACC)
            chunks = _diagnostic_part_chunks(P, ROWS, task, K, PARTS)
            first_work += tl.where(task == pid, chunks, 0)
            done += 1
            work += chunks
            maximum = tl.maximum(maximum, chunks)
            task += WORKERS
    else:
        if task < total:
            _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, task, XS, K, PARTS, COMPLEX, ACC)
            chunks = _diagnostic_part_chunks(P, ROWS, task, K, PARTS)
            first_work = chunks
            done += 1
            work += chunks
            maximum = tl.maximum(maximum, chunks)
        if total > WORKERS:
            first = WORKERS + tl.atomic_add(COUNTS + 2, BATCH, sem="relaxed", scope="gpu").to(tl.int64)
            claims += 1
            while first < total:
                for offset in tl.static_range(0, BATCH):
                    current = first + offset
                    if current < total:
                        _ticket_exclusive_part(A, I, P, X, ROWS, PARTIAL, current, XS, K, PARTS, COMPLEX, ACC)
                        chunks = _diagnostic_part_chunks(P, ROWS, current, K, PARTS)
                        done += 1
                        work += chunks
                        maximum = tl.maximum(maximum, chunks)
                first = WORKERS + tl.atomic_add(COUNTS + 2, BATCH, sem="relaxed", scope="gpu").to(tl.int64)
                claims += 1
    tl.store(TRACE + 6 * pid, done)
    tl.store(TRACE + 6 * pid + 1, work)
    tl.store(TRACE + 6 * pid + 2, maximum)
    tl.store(TRACE + 6 * pid + 3, claims)
    tl.store(TRACE + 6 * pid + 4, first_work)
    tl.store(TRACE + 6 * pid + 5, work - first_work)


def run_three_tier_ticket(prepared, x, out, timing, phase):
    from ._spmv_csr_config import ticket_queue_capacity

    cfg = prepared.config
    stats = ticket_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg, prepared.data.dtype)
    medium_capacity, long_capacity = stats["medium_queue_capacity"], stats["long_queue_capacity"]
    medium_grid = min(medium_capacity, cfg["medium_grid_blocks"])
    parts = cfg["cooperative_parts"]
    long_grid = min(long_capacity, cfg["long_grid_blocks"])
    workers = stats["long_worker_programs"]
    complex_input = prepared.data.is_complex()
    fp64 = prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
    component = torch.float64 if fp64 else torch.float32
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        return (prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(),
                torch.zeros(3, dtype=torch.int32, device=x.device),
                torch.empty(medium_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity, dtype=torch.int64, device=x.device),
                torch.empty(long_capacity * parts * (2 if complex_input else 1), dtype=component, device=x.device))
    (data, vector, counts, medium_rows, long_rows, partial), init_ms = phase(initialize, timing)
    long_count = counts[1:2]
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if fp64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            three_tier_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, long_rows, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], S=cfg["short_row_threshold"], L=cfg["long_row_threshold"],
                num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    def medium_compute():
        if medium_grid:
            short_rowqueue_strided_vector_kernel[(medium_grid,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counts, medium_rows, vector.stride(0), y.stride(0), K=cfg["block_nnz"],
                num_warps=cfg["num_warps"], **common)
    _, medium_ms = phase(medium_compute, timing)
    def long_compute():
        if workers:
            three_tier_ticket_consumer_kernel[(workers,)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), counts,
                long_rows, partial, vector.stride(0), WORKERS=workers, BATCH=cfg["ticket_batch"],
                STATIC=cfg["long_schedule"] == "static", K=cfg["block_nnz"], PARTS=parts,
                num_warps=cfg["num_warps"], **common)
    _, long_ms = phase(long_compute, timing)
    def finish():
        if long_grid:
            cooperative_row_finish_kernel[(long_grid,)](
                long_count, long_rows, partial, view(y), y.stride(0), PARTS=parts,
                num_warps=cfg["producer_num_warps"], **common)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + medium_ms + long_ms + finish_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms,
                   medium_consumer_ms=medium_ms, long_consumer_ms=long_ms,
                   consumer_ms=medium_ms + long_ms if timing else None, finish_ms=finish_ms,
                   fused_plan_compute_ms=producer_ms, queue_count_readback=False,
                   plan_strategy="per_run_three_tier_row_ids", floating_point_atomic=False,
                   task_acquisition="static_flat_parts" if cfg["long_schedule"] == "static" else "static_first_wave_ticket_remaining",
                   long_schedule=cfg["long_schedule"], ticket_batch=cfg["ticket_batch"], cooperative_parts=parts,
                   medium_grid_blocks=medium_grid, medium_grid_limit=cfg["medium_grid_blocks"],
                   medium_rounds_capacity=triton.cdiv(medium_capacity, medium_grid) if medium_grid else 0,
                   long_row_grid_blocks=long_grid, long_consumer_grid_blocks=workers,
                   initialization="two_queue_counters_and_ticket", output_conversion="final_store",
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def longrow_grid_consumer_kernel(A, I, X, COUNTERS, JOB_SLOT, JOB_START, JOB_END,
                                 PARTIAL, XS, K: tl.constexpr, COMPLEX: tl.constexpr,
                                 ACC: tl.constexpr):
    job = tl.program_id(0).to(tl.int64)
    count = tl.load(COUNTERS + 1).to(tl.int64)
    if job < count:
        slot = tl.load(JOB_SLOT + job)
        start, end = tl.load(JOB_START + job), tl.load(JOB_END + job)
        lane = tl.arange(0, K).to(tl.int64)
        re, im = tl.zeros((K,), ACC), tl.zeros((K,), ACC)
        for base in range(start, end, K):
            pos = base + lane
            pr, pi = _short_row_product(A, I, X, pos, pos < end, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
        if COMPLEX:
            tl.atomic_add(PARTIAL + 2 * slot, tl.sum(re, 0), sem="relaxed")
            tl.atomic_add(PARTIAL + 2 * slot + 1, tl.sum(im, 0), sem="relaxed")
        else:
            tl.atomic_add(PARTIAL + slot, tl.sum(re, 0), sem="relaxed")


def run_longrow_grid(prepared, x, out, timing, phase):
    from ._spmv_csr_config import longrow_queue_capacity

    cfg = prepared.config
    complex_input = prepared.data.is_complex()
    dtype = torch.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else torch.float32
    channels = 2 if complex_input else 1
    # Reuse the queue's conservative capacity validation, without a worker knob.
    stats = longrow_queue_capacity(prepared.n_rows, prepared.data.numel(), dict(cfg, worker_programs=1),
                                  8 if dtype == torch.float64 else 4, channels)
    capacity, jobs = stats["long_row_capacity"], stats["job_capacity"]
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        counters = torch.zeros(3, dtype=torch.int32, device=x.device)
        owners = torch.empty(capacity, dtype=torch.int64, device=x.device)
        descriptors = tuple(torch.empty(jobs, dtype=torch.int64, device=x.device) for _ in range(3))
        partial = torch.zeros(capacity * channels, dtype=dtype, device=x.device)
        return prepared.data.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg(), counters, owners, descriptors, partial
    (data, vector, counters, owners, descriptors, partial), init_ms = phase(initialize, timing)
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if dtype == torch.float64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            longrow_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counters, owners, *descriptors, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], K=cfg["short_block_nnz"], L=cfg["long_row_threshold"],
                S=cfg["segment_nnz"], num_warps=cfg["producer_num_warps"], **common)
    _, producer_ms = phase(produce, timing)
    def consume():
        if jobs:
            longrow_grid_consumer_kernel[(jobs,)](
                view(data), prepared.kernel_indices, view(vector), counters, *descriptors, partial,
                vector.stride(0), K=cfg["block_nnz"], num_warps=cfg["num_warps"], **common)
    _, consumer_ms = phase(consume, timing)
    def finish():
        if capacity:
            longrow_finish_kernel[(triton.cdiv(capacity, 128),)](
                counters, owners, partial, view(y), y.stride(0), B=128, COMPLEX=complex_input,
                num_warps=cfg["num_warps"], num_stages=1)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=init_ms,
                   compute_ms=producer_ms + consumer_ms + finish_ms if timing else None,
                   init_ms=init_ms, producer_ms=producer_ms, consumer_ms=consumer_ms, finish_ms=finish_ms,
                   fused_plan_compute_ms=producer_ms, queue_count_readback=False,
                   grid_capacity=jobs, task_acquisition="static_program_id", plan_strategy="per_run_gpu_longrow_descriptors",
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()), output_buffer_identity="output")

from . import _common


@triton.jit
def row_scalar_tail_reduce_kernel(A, I, P, X, Y, M, XS, YS,
                                  R: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
                                  COMPLEX: tl.constexpr, ACC: tl.constexpr):
    # One program owns all outputs. Prefix and tail partition each row's NNZ.
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    longest = tl.max(end - start, 0)
    re = tl.zeros((R,), ACC)
    im = tl.zeros((R,), ACC)
    for step in range(tl.minimum(longest, S)):
        pos = start + step
        pr, pi = _short_row_product(A, I, X, pos, valid & (pos < end), XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    if longest > S:
        lane = tl.arange(0, K).to(tl.int64)
        tail_re = tl.zeros((R, K), ACC)
        tail_im = tl.zeros((R, K), ACC)
        for base in range(S, longest, K):
            pos = start[:, None] + base + lane[None, :]
            active = valid[:, None] & (pos < end[:, None])
            pr, pi = _short_row_product(A, I, X, pos, active, XS, COMPLEX, ACC)
            tail_re += pr
            if COMPLEX:
                tail_im += pi
        re += tl.sum(tail_re, 1)
        if COMPLEX:
            im += tl.sum(tail_im, 1)
    yp = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * yp, re, valid)
        tl.store(Y + 2 * yp + 1, im, valid)
    else:
        tl.store(Y + yp, re, valid)


@triton.jit
def row_scalar_bound_kernel(A, I, P, X, Y, M, XS, YS,
                            R: tl.constexpr, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    steps = tl.max(end - start, 0)
    re = tl.zeros((R,), ACC)
    im = tl.zeros((R,), ACC)
    for step in range(steps):
        pos = start + step
        pr, pi = _short_row_product(A, I, X, pos, valid & (pos < end), XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    yp = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * yp, re, valid)
        tl.store(Y + 2 * yp + 1, im, valid)
    else:
        tl.store(Y + yp, re, valid)


@triton.jit
def longrow_producer_kernel(A, I, P, X, Y, COUNTERS, OWNERS, JOB_SLOT, JOB_START,
                            JOB_END, M, XS, YS, R: tl.constexpr, K: tl.constexpr,
                            L: tl.constexpr, S: tl.constexpr,
                            COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    long = valid & (end - start > L)
    long_flags = long.to(tl.int32)
    count = tl.sum(long_flags, 0)
    if count > 0:
        first_slot = tl.atomic_add(COUNTERS, count, sem="relaxed").to(tl.int64)
        slots = first_slot + tl.cumsum(long_flags, 0).to(tl.int64) - 1
        tl.store(OWNERS + slots, row, long)
        jobs = tl.where(long, tl.cdiv(end - start, S), 0)
        job_prefix = tl.cumsum(jobs, 0)
        first_job = tl.atomic_add(COUNTERS + 1, tl.sum(jobs, 0).to(tl.int32), sem="relaxed").to(tl.int64)
        job_base = first_job + job_prefix - jobs
        # Bounded descriptor tiles: no storage proportional to the longest row in registers.
        lane = tl.arange(0, 32).to(tl.int64)
        for base in range(0, tl.max(jobs, 0), 32):
            j = base + lane
            active = long[:, None] & (j[None, :] < jobs[:, None])
            offset = job_base[:, None] + j[None, :]
            begin = start[:, None] + j[None, :] * S
            tl.store(JOB_SLOT + offset, tl.broadcast_to(slots[:, None], (R, 32)), active)
            tl.store(JOB_START + offset, begin, active)
            tl.store(JOB_END + offset, tl.minimum(begin + S, end[:, None]), active)
    ordinary = valid & ~long
    steps = tl.max(tl.where(ordinary, tl.cdiv(end - start, K), 0), 0)
    lane = tl.arange(0, K).to(tl.int64)
    re = tl.zeros((R, K), ACC)
    im = tl.zeros((R, K), ACC)
    for step in range(steps):
        pos = start[:, None] + step * K + lane[None, :]
        mask = ordinary[:, None] & (pos < end[:, None])
        pr, pi = _short_row_product(A, I, X, pos, mask, XS, COMPLEX, ACC)
        re += pr
        if COMPLEX:
            im += pi
    yp = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * yp, tl.sum(re, 1), ordinary)
        tl.store(Y + 2 * yp + 1, tl.sum(im, 1), ordinary)
    else:
        tl.store(Y + yp, tl.sum(re, 1), ordinary)


@triton.jit
def longrow_consumer_kernel(A, I, X, COUNTERS, JOB_SLOT, JOB_START, JOB_END,
                            PARTIAL, XS, K: tl.constexpr, COMPLEX: tl.constexpr,
                            ACC: tl.constexpr):
    count = tl.load(COUNTERS + 1)
    job = tl.atomic_add(COUNTERS + 2, 1, sem="relaxed")
    lane = tl.arange(0, K).to(tl.int64)
    while job < count:
        slot = tl.load(JOB_SLOT + job.to(tl.int64))
        start = tl.load(JOB_START + job.to(tl.int64))
        end = tl.load(JOB_END + job.to(tl.int64))
        re = tl.zeros((K,), ACC)
        im = tl.zeros((K,), ACC)
        for base in range(start, end, K):
            pos = base + lane
            pr, pi = _short_row_product(A, I, X, pos, pos < end, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
        if COMPLEX:
            tl.atomic_add(PARTIAL + 2 * slot, tl.sum(re, 0), sem="relaxed")
            tl.atomic_add(PARTIAL + 2 * slot + 1, tl.sum(im, 0), sem="relaxed")
        else:
            tl.atomic_add(PARTIAL + slot, tl.sum(re, 0), sem="relaxed")
        job = tl.atomic_add(COUNTERS + 2, 1, sem="relaxed")


@triton.jit
def longrow_finish_kernel(COUNTERS, OWNERS, PARTIAL, Y, YS,
                          B: tl.constexpr, COMPLEX: tl.constexpr):
    slot = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    valid = slot < tl.load(COUNTERS)
    row = tl.load(OWNERS + slot, valid, 0)
    yp = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        re = tl.load(PARTIAL + 2 * slot, valid, 0)
        im = tl.load(PARTIAL + 2 * slot + 1, valid, 0)
        tl.store(Y + 2 * yp, re, valid)
        tl.store(Y + 2 * yp + 1, im, valid)
    else:
        tl.store(Y + yp, tl.load(PARTIAL + slot, valid, 0), valid)


def run_longrow_queue(prepared, x, out, timing, phase):
    from ._spmv_csr_config import longrow_queue_capacity

    cfg = prepared.config
    complex_input = prepared.data.is_complex()
    dtype = torch.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else torch.float32
    channels = 2 if complex_input else 1
    stats = longrow_queue_capacity(prepared.n_rows, prepared.data.numel(), cfg,
                                  8 if dtype == torch.float64 else 4, channels)
    capacity, jobs = stats["long_row_capacity"], stats["job_capacity"]
    y = out if out is not None else torch.empty(prepared.n_rows, dtype=prepared.data.dtype, device=x.device)
    def initialize():
        data = prepared.data.resolve_conj().resolve_neg()
        vector = x.resolve_conj().resolve_neg()
        counters = torch.zeros(3, dtype=torch.int32, device=x.device)
        owners = torch.empty(capacity, dtype=torch.int64, device=x.device)
        descriptors = tuple(torch.empty(jobs, dtype=torch.int64, device=x.device) for _ in range(3))
        partial = torch.zeros(capacity * channels, dtype=dtype, device=x.device)
        return data, vector, counters, owners, descriptors, partial
    (data, vector, counters, owners, descriptors, partial), process_ms = phase(initialize, timing)
    view = lambda tensor: torch.view_as_real(tensor) if complex_input else tensor
    common = dict(COMPLEX=complex_input, ACC=tl.float64 if dtype == torch.float64 else tl.float32,
                  num_stages=cfg["num_stages"], enable_fp_fusion=False)
    def produce():
        if prepared.n_rows:
            longrow_producer_kernel[(triton.cdiv(prepared.n_rows, cfg["rows_per_program"]),)](
                view(data), prepared.kernel_indices, prepared.kernel_indptr, view(vector), view(y),
                counters, owners, *descriptors, prepared.n_rows, vector.stride(0), y.stride(0),
                R=cfg["rows_per_program"], K=cfg["short_block_nnz"],
                L=cfg["long_row_threshold"], S=cfg["segment_nnz"],
                num_warps=cfg["producer_num_warps"], **common)
    _, fused_ms = phase(produce, timing)
    def consume():
        if capacity:
            longrow_consumer_kernel[(cfg["worker_programs"],)](
                view(data), prepared.kernel_indices, view(vector), counters, *descriptors,
                partial, vector.stride(0), K=cfg["block_nnz"], num_warps=cfg["num_warps"], **common)
    def finish():
        if capacity:
            longrow_finish_kernel[(triton.cdiv(capacity, 128),)](
                counters, owners, partial, view(y), y.stride(0), B=128, COMPLEX=complex_input,
                num_warps=cfg["num_warps"], num_stages=1)
    _, consumer_ms = phase(consume, timing)
    _, finish_ms = phase(finish, timing)
    return y, dict(stats, process_cpu_ms=0.0, process_gpu_ms=process_ms,
                   compute_ms=(fused_ms + consumer_ms + finish_ms) if timing else None,
                   init_ms=process_ms, producer_ms=fused_ms, consumer_ms=consumer_ms, finish_ms=finish_ms,
                   fused_plan_compute_ms=fused_ms, queue_count_readback=False,
                   execution_indices_dtype=str(prepared.kernel_indices.dtype).removeprefix("torch."),
                   execution_indptr_dtype=str(prepared.kernel_indptr.dtype).removeprefix("torch."),
                   x_stride=tuple(x.stride()), y_stride=tuple(y.stride()),
                   plan_strategy="per_run_long_rows_only", output_buffer_identity="out" if out is not None else "allocated")


@triton.jit
def row_scalar_strip_kernel(A, I, P, X, Y, M, XS, YS,
                            R: tl.constexpr, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    pos = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    re = tl.zeros((R,), ACC)
    im = tl.zeros((R,), ACC)
    while tl.sum((valid & (pos < end)).to(tl.int32), 0) > 0:
        active = valid & (pos < end)
        col = tl.load(I + pos, active, 0).to(tl.int64)
        xp = col * tl.cast(XS, tl.int64)
        if COMPLEX:
            ar = tl.load(A + 2 * pos, active, 0).to(ACC)
            ai = tl.load(A + 2 * pos + 1, active, 0).to(ACC)
            xr = tl.load(X + 2 * xp, active, 0).to(ACC)
            xi = tl.load(X + 2 * xp + 1, active, 0).to(ACC)
            re += ar * xr - ai * xi
            im += ar * xi + ai * xr
        else:
            a = tl.load(A + pos, active, 0).to(ACC)
            x = tl.load(X + xp, active, 0).to(ACC)
            re += a * x
        pos = tl.minimum(pos + 1, end)
    yp = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * yp, re, valid)
        tl.store(Y + 2 * yp + 1, im, valid)
    else:
        tl.store(Y + yp, re, valid)


@triton.jit
def _short_row_product(A, I, X, pos, mask, XS, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    col = tl.load(I + pos, mask, 0).to(tl.int64)
    xp = col * tl.cast(XS, tl.int64)
    if COMPLEX:
        ar = tl.load(A + 2 * pos, mask, 0).to(ACC)
        ai = tl.load(A + 2 * pos + 1, mask, 0).to(ACC)
        xr = tl.load(X + 2 * xp, mask, 0).to(ACC)
        xi = tl.load(X + 2 * xp + 1, mask, 0).to(ACC)
        return ar * xr - ai * xi, ar * xi + ai * xr
    a = tl.load(A + pos, mask, 0).to(ACC)
    x = tl.load(X + xp, mask, 0).to(ACC)
    return a * x, tl.full(pos.shape, 0, ACC)


@triton.jit
def row_static_tile_kernel(A, I, P, X, Y, M, XS, YS,
                           R: tl.constexpr, K: tl.constexpr,
                           COMPLEX: tl.constexpr, ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R).to(tl.int64)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    lane = tl.arange(0, K).to(tl.int64)
    longest = tl.max(end - start, 0)
    re = tl.zeros((R, K), ACC)
    im = tl.zeros((R, K), ACC)
    if longest <= K:
        pos = start[:, None] + lane[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        re, im = _short_row_product(A, I, X, pos, mask, XS, COMPLEX, ACC)
    else:
        for base in range(0, longest, K):
            pos = start[:, None] + base + lane[None, :]
            mask = valid[:, None] & (pos < end[:, None])
            pr, pi = _short_row_product(A, I, X, pos, mask, XS, COMPLEX, ACC)
            re += pr
            if COMPLEX:
                im += pi
    yp = row * tl.cast(YS, tl.int64)
    if COMPLEX:
        tl.store(Y + 2 * yp, tl.sum(re, 1), valid)
        tl.store(Y + 2 * yp + 1, tl.sum(im, 1), valid)
    else:
        tl.store(Y + yp, tl.sum(re, 1), valid)


@triton.jit
def row_refill_kernel(A, I, P, X, Y, M, XS, YS,
                      R: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
                      COMPLEX: tl.constexpr, ACC: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * W
    limit = tl.minimum(base + W, M)
    row = base + tl.arange(0, R).to(tl.int64)
    lane = tl.arange(0, V).to(tl.int64)
    active = row < limit
    pos = tl.load(P + row, active, 0).to(tl.int64)
    end = tl.load(P + row + 1, active, 0).to(tl.int64)
    cursor = tl.full((), R, tl.int64)
    re = tl.zeros((R, V), ACC)
    im = tl.zeros((R, V), ACC)
    while tl.sum(active.to(tl.int32), 0) > 0:
        p = pos[:, None] + lane[None, :]
        valid = active[:, None] & (p < end[:, None])
        col = tl.load(I + p, valid, 0).to(tl.int64)
        xp = col * tl.cast(XS, tl.int64)
        if COMPLEX:
            ar = tl.load(A + 2 * p, valid, 0).to(ACC)
            ai = tl.load(A + 2 * p + 1, valid, 0).to(ACC)
            xr = tl.load(X + 2 * xp, valid, 0).to(ACC)
            xi = tl.load(X + 2 * xp + 1, valid, 0).to(ACC)
            re += ar * xr - ai * xi
            im += ar * xi + ai * xr
        else:
            a = tl.load(A + p, valid, 0).to(ACC)
            x = tl.load(X + xp, valid, 0).to(ACC)
            re += a * x
        pos = tl.minimum(pos + V, end)
        done = active & (pos >= end)
        count = tl.sum(done.to(tl.int32), 0)
        if count > 0:
            yp = row * tl.cast(YS, tl.int64)
            if COMPLEX:
                tl.store(Y + 2 * yp, tl.sum(re, 1), done)
                tl.store(Y + 2 * yp + 1, tl.sum(im, 1), done)
            else:
                tl.store(Y + yp, tl.sum(re, 1), done)
            rank = tl.cumsum(done.to(tl.int32), 0).to(tl.int64) - 1
            candidate = base + cursor + rank
            assign = done & (candidate < limit)
            fresh_pos = tl.load(P + candidate, assign, 0).to(tl.int64)
            fresh_end = tl.load(P + candidate + 1, assign, 0).to(tl.int64)
            row = tl.where(done, candidate, row)
            pos = tl.where(done, fresh_pos, pos)
            end = tl.where(done, fresh_end, end)
            re = tl.where(done[:, None], tl.full((R, V), 0, ACC), re)
            if COMPLEX:
                im = tl.where(done[:, None], tl.full((R, V), 0, ACC), im)
            active = (active & ~done) | assign
            cursor += count.to(tl.int64)


@triton.jit
def _rowgroup_segment_add(a, ha, b, hb):
    return tl.where(hb, b, a + b), ha | hb


@triton.jit
def rowgroup_stream_reduce_kernel(A, I, P, X, Y, M,
                                  XS: tl.constexpr, YS: tl.constexpr,
                                  R: tl.constexpr, T: tl.constexpr,
                                  COMPLEX: tl.constexpr, ACC: tl.constexpr):
    first = tl.program_id(0).to(tl.int64) * R
    rows = first + tl.arange(0, R)
    last = tl.minimum(first + R, M)
    starts = tl.load(P + rows, rows < M, 0).to(tl.int64)
    ends = tl.load(P + rows + 1, rows < M, 0).to(tl.int64)
    begin = tl.load(P + first).to(tl.int64)
    stop = tl.load(P + last).to(tl.int64)
    lane = tl.arange(0, T)
    sr = tl.zeros((R,), ACC)
    si = tl.zeros((R,), ACC)
    for base in range(begin, stop, T):
        pos = base + lane
        valid = pos < stop
        # upper_bound over this row group, including repeated pointers.
        lo = tl.full((T,), first, tl.int64)
        hi = tl.full((T,), last + 1, tl.int64)
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            pointer = tl.load(P + mid, (lo < hi) & (mid <= last), 0)
            right = (lo < hi) & (pointer <= pos)
            hi = tl.where((lo < hi) & ~right, mid, hi)
            lo = tl.where(right, mid + 1, lo)
        owner = tl.where(valid, lo - 1, -1)
        prev = tl.gather(owner, tl.maximum(lane - 1, 0), 0)
        head = (lane == 0) | (owner != prev)
        col = tl.load(I + pos, valid, 0).to(tl.int64)
        if COMPLEX:
            ar = tl.load(A + 2 * pos, valid, 0).to(ACC)
            ai = tl.load(A + 2 * pos + 1, valid, 0).to(ACC)
            xr = tl.load(X + 2 * col * XS, valid, 0).to(ACC)
            xi = tl.load(X + 2 * col * XS + 1, valid, 0).to(ACC)
            re, _ = tl.associative_scan((ar * xr - ai * xi, head), 0, _rowgroup_segment_add)
            im, _ = tl.associative_scan((ar * xi + ai * xr, head), 0, _rowgroup_segment_add)
        else:
            a = tl.load(A + pos, valid, 0).to(ACC)
            x = tl.load(X + col * XS, valid, 0).to(ACC)
            re, _ = tl.associative_scan((a * x, head), 0, _rowgroup_segment_add)
        end = tl.minimum(ends, tl.minimum(base + T, stop))
        active = (rows < M) & (tl.maximum(starts, base) < end)
        pick = tl.minimum(tl.maximum(end - base - 1, 0), T - 1).to(tl.int32)
        sr += tl.where(active, tl.gather(re, pick, 0), 0)
        if COMPLEX:
            si += tl.where(active, tl.gather(im, pick, 0), 0)
    if COMPLEX:
        tl.store(Y + 2 * rows * YS, sr, rows < M)
        tl.store(Y + 2 * rows * YS + 1, si, rows < M)
    else:
        tl.store(Y + rows * YS, sr, rows < M)


def uses_native_rocm_fp32_accumulation(prepared):
    """Return whether a real ROCm FP32 route can safely accumulate in FP32.

    A row_tile lane accumulates consecutive products before the lane reduction.
    On gfx936, rows crossing two 16-wide chunks can exceed the public FP32
    tolerance for cancellation-heavy inputs.  Keep those rows in FP64 while
    retaining the native FP32 fast path for the short-row delivery routes.
    """
    return (
        prepared.data.dtype == torch.float32
        and getattr(prepared.backend_caps, "backend", None) == "rocm"
        and prepared.max_row_nnz <= 32
    )


@triton.jit
def _product(A, X, pos, col, mask, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    if COMPLEX:
        ar = tl.load(A + 2 * pos, mask, 0).to(ACC)
        ai = tl.load(A + 2 * pos + 1, mask, 0).to(ACC)
        xr = tl.load(X + 2 * col, mask, 0).to(ACC)
        xi = tl.load(X + 2 * col + 1, mask, 0).to(ACC)
        return ar * xr - ai * xi, ar * xi + ai * xr
    else:
        a = tl.load(A + pos, mask, 0).to(ACC)
        x = tl.load(X + col, mask, 0).to(ACC)
        return a * x, tl.full(pos.shape, 0, ACC)


@triton.jit
def _store_result(Y, row, real, imag, mask, COMPLEX: tl.constexpr):
    if COMPLEX:
        tl.store(Y + row * 2, real, mask)
        tl.store(Y + row * 2 + 1, imag, mask)
    else:
        tl.store(Y + row, real, mask)


@triton.jit
def row_tile_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    INDEXED: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    STAGES: tl.constexpr,
    COMPLEX: tl.constexpr = False,
    ACC: tl.constexpr = tl.float64,
):
    ridx = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    valid = ridx < N
    if INDEXED:
        row = tl.load(ROWS + ridx, valid, 0)
    else:
        row = ridx
    start = tl.load(RP + row, valid, 0).to(tl.int64)
    end = tl.load(RP + row + 1, valid, 0).to(tl.int64)
    lane = tl.arange(0, V)
    acc = tl.zeros((R, V), ACC)
    imag = tl.zeros((R, V), ACC)
    steps = tl.max(tl.cdiv(end - start, V), 0)
    for step in tl.range(0, steps, num_stages=STAGES):
        pos = start[:, None] + step * V + lane[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
        acc = acc + pr
        if COMPLEX:
            imag = imag + pi
    _store_result(Y, row, tl.sum(acc, 1), tl.sum(imag, 1), valid, COMPLEX)


@triton.jit
def row_subgroup_stream_kernel(
    A, CI, RP, X, Y, M, R: tl.constexpr, V: tl.constexpr,
    GROUPS: tl.constexpr, STAGES: tl.constexpr,
    COMPLEX: tl.constexpr, ACC: tl.constexpr,
):
    first = tl.program_id(0).to(tl.int64) * R * GROUPS
    lane = tl.arange(0, V)
    for group in range(GROUPS):
        row = first + group * R + tl.arange(0, R)
        valid = row < M
        start = tl.load(RP + row, valid, 0).to(tl.int64)
        end = tl.load(RP + row + 1, valid, 0).to(tl.int64)
        real = tl.zeros((R, V), ACC)
        imag = tl.zeros((R, V), ACC)
        steps = tl.max(tl.cdiv(end - start, V), 0)
        for step in tl.range(0, steps, num_stages=STAGES):
            pos = start[:, None] + step * V + lane[None, :]
            mask = valid[:, None] & (pos < end[:, None])
            col = tl.load(CI + pos, mask, 0).to(tl.int64)
            pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
            real += pr
            if COMPLEX:
                imag += pi
        _store_result(Y, row, tl.sum(real, 1), tl.sum(imag, 1), valid, COMPLEX)




@triton.jit
def row_tile_real_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    INDEXED: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    STAGES: tl.constexpr,
    ACC: tl.constexpr,
    POSITION_64: tl.constexpr,
    FIXED_STEPS: tl.constexpr,
):
    """Real-only row tile without complex component bookkeeping."""
    ridx = tl.program_id(0) * R + tl.arange(0, R)
    if POSITION_64:
        ridx = ridx.to(tl.int64)
    valid = ridx < N
    if INDEXED:
        row = tl.load(ROWS + ridx, valid, 0)
    else:
        row = ridx
    start = tl.load(RP + row, valid, 0)
    end = tl.load(RP + row + 1, valid, 0)
    if POSITION_64:
        start = start.to(tl.int64)
        end = end.to(tl.int64)
    lane = tl.arange(0, V)
    acc = tl.zeros((R, V), ACC)
    if FIXED_STEPS > 0:
        steps = FIXED_STEPS
    else:
        steps = tl.max(tl.cdiv(end - start, V), 0)
    for step in tl.range(0, steps, num_stages=STAGES):
        pos = start[:, None] + step * V + lane[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        value = tl.load(A + pos, mask, 0).to(ACC)
        vector = tl.load(X + col, mask, 0).to(ACC)
        acc += value * vector
    tl.store(Y + row, tl.sum(acc, 1), valid)


@triton.jit
def row_vector_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    INDEXED: tl.constexpr,
    B: tl.constexpr,
    STAGES: tl.constexpr,
    COMPLEX: tl.constexpr = False,
    ACC: tl.constexpr = tl.float64,
):
    pid = tl.program_id(0).to(tl.int64)
    if INDEXED:
        row = tl.load(ROWS + pid)
    else:
        row = pid
    start = tl.load(RP + row).to(tl.int64)
    end = tl.load(RP + row + 1).to(tl.int64)
    lane = tl.arange(0, B)
    acc = tl.zeros((B,), ACC)
    imag = tl.zeros((B,), ACC)
    for step in tl.range(0, tl.cdiv(end - start, B), num_stages=STAGES):
        pos = start + step * B + lane
        mask = pos < end
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
        acc = acc + pr
        if COMPLEX:
            imag = imag + pi
    _store_result(Y, row, tl.sum(acc, 0), tl.sum(imag, 0), True, COMPLEX)


@triton.jit
def bucket_rows_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    BATCH: tl.constexpr,
    B: tl.constexpr,
    MAX_SEGS: tl.constexpr,
    COMPLEX: tl.constexpr,
    ACC: tl.constexpr,
):
    """Legacy bucket scheduling, including sequential short-row batches."""
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    for batch in range(BATCH):
        index = pid * BATCH + batch
        active = index < N
        row = tl.load(ROWS + index, active, 0).to(tl.int64)
        start = tl.load(RP + row, active, 0).to(tl.int64)
        end = tl.load(RP + row + 1, active, 0).to(tl.int64)
        real = tl.zeros((B,), ACC)
        imag = tl.zeros((B,), ACC)
        for step in range(MAX_SEGS):
            pos = start + step * B + lane
            mask = active & (pos < end)
            col = tl.load(CI + pos, mask, 0).to(tl.int64)
            pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
            real += pr
            if COMPLEX:
                imag += pi
        _store_result(Y, row, tl.sum(real, 0), tl.sum(imag, 0), active, COMPLEX)


@triton.jit
def classify_kernel(
    RP,
    SHORT,
    MID,
    COUNTS,
    M,
    S: tl.constexpr,
    L: tl.constexpr,
    T: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = r < M
    start = tl.load(RP + r, valid, 0).to(tl.int64)
    end = tl.load(RP + r + 1, valid, 0).to(tl.int64)
    length = end - start
    if ADAPTIVE:
        short = length <= S
        mid = (length > S) & (length <= L)
        counts = tl.where(length > L, tl.cdiv(length, T), 0)
    else:
        short = tl.full((B,), False, tl.int1)
        mid = tl.full((B,), False, tl.int1)
        counts = tl.cdiv(length, T)
    tl.store(SHORT + r, short.to(tl.int64), valid)
    tl.store(MID + r, mid.to(tl.int64), valid)
    tl.store(COUNTS + r, counts, valid)


@triton.jit
def compact_rows_kernel(SHORT, MID, SP, MP, SR, MR, M, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = r < M
    s = tl.load(SHORT + r, valid, 0) != 0
    m = tl.load(MID + r, valid, 0) != 0
    sp = tl.load(SP + r, valid, 0) - 1
    mp = tl.load(MP + r, valid, 0) - 1
    tl.store(SR + sp, r, valid & s)
    tl.store(MR + mp, r, valid & m)


@triton.jit
def _owner(PREFIX, index, M, valid):
    # First row whose inclusive segment offset is greater than index.
    lo = tl.full(index.shape, 0, tl.int64)
    hi = tl.full(index.shape, M, tl.int64)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        value = tl.load(PREFIX + mid, valid & (mid < M), 0)
        take = value <= index
        active = lo < hi
        lo = tl.where(active & take, mid + 1, lo)
        hi = tl.where(active & ~take, mid, hi)
    return lo


@triton.jit
def descriptors_kernel(
    PREFIX, RP, ROWS, STARTS, LENGTHS, TOTAL, M, T: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = i < TOTAL
    row = _owner(PREFIX, i, M, valid)
    before = tl.load(PREFIX + row - 1, valid & (row > 0), 0)
    start = tl.load(RP + row, valid, 0).to(tl.int64) + (i - before) * T
    end = tl.load(RP + row + 1, valid, 0).to(tl.int64)
    tl.store(ROWS + i, row, valid)
    tl.store(STARTS + i, start, valid)
    tl.store(LENGTHS + i, tl.minimum(end - start, T), valid)


@triton.jit
def segment_kernel(
    A,
    CI,
    X,
    STARTS,
    LENGTHS,
    PARTIAL,
    B: tl.constexpr,
    STAGES: tl.constexpr,
    COMPLEX: tl.constexpr = False,
    ACC: tl.constexpr = tl.float64,
):
    pid = tl.program_id(0).to(tl.int64)
    start = tl.load(STARTS + pid)
    length = tl.load(LENGTHS + pid)
    lane = tl.arange(0, B)
    acc = tl.zeros((B,), ACC)
    imag = tl.zeros((B,), ACC)
    for step in tl.range(0, tl.cdiv(length, B), num_stages=STAGES):
        offset = step * B + lane
        mask = offset < length
        pos = start + offset
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
        acc = acc + pr
        if COMPLEX:
            imag = imag + pi
    _store_result(PARTIAL, pid, tl.sum(acc, 0), tl.sum(imag, 0), True, COMPLEX)


@triton.jit
def reduce_level_kernel(
    PARTIAL,
    PREV_PREFIX,
    NEXT_PREFIX,
    OUTPUT,
    M,
    B: tl.constexpr,
    COMPLEX: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    # A one-element block lets the shared owner search serve scalar programs.
    ids = tl.full((1,), pid, tl.int64)
    row = tl.sum(_owner(NEXT_PREFIX, ids, M, tl.full((1,), True, tl.int1)), 0)
    group_base = tl.load(NEXT_PREFIX + row - 1, row > 0, 0)
    prev_base = tl.load(PREV_PREFIX + row - 1, row > 0, 0)
    prev_end = tl.load(PREV_PREFIX + row)
    pos = prev_base + (pid - group_base) * B + tl.arange(0, B)
    stride: tl.constexpr = 2 if COMPLEX else 1
    for component in tl.static_range(stride):
        value = tl.load(PARTIAL + pos * stride + component, pos < prev_end, 0)
        tl.store(OUTPUT + pid * stride + component, tl.sum(value, 0))


@triton.jit
def finish_kernel(
    PARTIAL,
    PREFIX,
    COUNTS,
    Y,
    M,
    WRITE_EMPTY: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    B: tl.constexpr,
    COMPLEX: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = row < M
    count = tl.load(COUNTS + row, valid, 0)
    stride: tl.constexpr = 2 if COMPLEX else 1
    for component in tl.static_range(stride):
        if HAS_PARTIAL:
            pos = tl.load(PREFIX + row, valid, 0) - 1
            value = tl.load(PARTIAL + pos * stride + component, valid & (count > 0), 0)
        else:
            value = tl.full((B,), 0, PARTIAL.dtype.element_ty)
        mask = valid if WRITE_EMPTY else valid & (count > 0)
        tl.store(Y + row * stride + component, value, mask)


def build_plan(prepared, config, adaptive):
    """GPU membership/segment construction; no Python row classification or cache."""
    m, device = prepared.n_rows, prepared.data.device
    alloc = lambda n: torch.empty(n, dtype=torch.int64, device=device)
    short, mid, counts = alloc(m), alloc(m), alloc(m)
    split = config["row_split_reduce"]
    process = config["process"]
    block = process["block_size"]
    classify_kernel[(triton.cdiv(m, block),)](
        prepared.kernel_indptr,
        short,
        mid,
        counts,
        m,
        S=config["short_row_threshold"],
        L=config["split_row_threshold"],
        T=split["segment_nnz"],
        ADAPTIVE=adaptive,
        B=block,
        num_warps=process["num_warps"],
    )
    prefix = counts.cumsum(0)
    if adaptive:
        sp, mp = short.cumsum(0), mid.cumsum(0)
        ns, nm, total, maximum = torch.stack(
            (sp[-1], mp[-1], prefix[-1], counts.max())
        ).tolist()
        sr, mr = alloc(ns), alloc(nm)
        compact_rows_kernel[(triton.cdiv(m, block),)](
            short, mid, sp, mp, sr, mr, m, B=block, num_warps=process["num_warps"]
        )
    else:
        total, maximum = torch.stack((prefix[-1], counts.max())).tolist()
        sr = mr = None
    rows, starts, lengths = alloc(total), alloc(total), alloc(total)
    if total:
        descriptors_kernel[(triton.cdiv(total, block),)](
            prefix,
            prepared.kernel_indptr,
            rows,
            starts,
            lengths,
            total,
            m,
            T=split["segment_nnz"],
            B=block,
            num_warps=process["num_warps"],
        )
    levels = []
    next_counts = counts
    while maximum > 1:
        block = split["reduce_block_size"]
        next_counts = torch.div(next_counts + block - 1, block, rounding_mode="floor")
        next_prefix = next_counts.cumsum(0)
        next_total = int(next_prefix[-1].item())
        levels.append((next_prefix, next_total))
        maximum = triton.cdiv(maximum, block)
    return {
        "short_rows": sr,
        "mid_rows": mr,
        "counts": counts,
        "prefix": prefix,
        "segment_rows": rows,
        "starts": starts,
        "lengths": lengths,
        "total": total,
        "levels": levels,
    }


def compute(prepared, x, y, alg, config, plan=None):
    m = prepared.n_rows
    if not m:
        return y
    if alg in ("csr_row_scalar_strip", "csr_row_static_tile", "csr_row_scalar_bound", "csr_row_scalar_tail_reduce"):
        complex_input = prepared.data.is_complex()
        view = lambda t: torch.view_as_real(t) if complex_input else t
        acc = tl.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else tl.float32
        kernel = {"csr_row_scalar_strip": row_scalar_strip_kernel, "csr_row_static_tile": row_static_tile_kernel,
                  "csr_row_scalar_bound": row_scalar_bound_kernel,
                  "csr_row_scalar_tail_reduce": row_scalar_tail_reduce_kernel}[alg]
        extra = {"K": config["block_nnz"]} if alg == "csr_row_static_tile" else {}
        if alg == "csr_row_scalar_tail_reduce":
            extra = dict(K=config["block_nnz"], S=config["scalar_steps"])
        kernel[(triton.cdiv(m, config["rows_per_program"]),)](
            view(prepared.data), prepared.kernel_indices, prepared.kernel_indptr,
            view(x), view(y), m, x.stride(0), y.stride(0),
            R=config["rows_per_program"], COMPLEX=complex_input, ACC=acc, **extra,
            num_warps=config["num_warps"], num_stages=config["num_stages"], enable_fp_fusion=False)
        return y
    if alg == "csr_row_refill_reduce":
        complex_input = prepared.data.is_complex()
        view = lambda t: torch.view_as_real(t) if complex_input else t
        acc = tl.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else tl.float32
        row_refill_kernel[(triton.cdiv(m, config["window_rows"]),)](
            view(prepared.data), prepared.kernel_indices, prepared.kernel_indptr,
            view(x), view(y), m, x.stride(0), y.stride(0),
            config["rows_per_program"], config["lanes_per_row"], config["window_rows"],
            complex_input, acc, num_warps=config["num_warps"],
            num_stages=config["num_stages"], enable_fp_fusion=False)
        return y
    if alg == "csr_rowgroup_stream_reduce":
        complex_input = prepared.data.is_complex()
        # Preserve component strides; reshape(-1) could copy a strided out.
        view = lambda t: torch.view_as_real(t) if complex_input else t
        acc = tl.float64 if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128) else tl.float32
        rowgroup_stream_reduce_kernel[(triton.cdiv(m, config["rows_per_program"]),)](
            view(prepared.data), prepared.kernel_indices, prepared.kernel_indptr,
            view(x), view(y), m, x.stride(0), y.stride(0),
            config["rows_per_program"], config["block_nnz"], complex_input, acc,
            num_warps=config["num_warps"], num_stages=config["num_stages"], enable_fp_fusion=False)
        return y
    if _common._is_rocm_runtime() and alg == "row_tile" and prepared.rocm_nnz_plan is not None:
        from . import _spmv_csr_nnz

        return _spmv_csr_nnz.bind(prepared, x, y)()
    complex_input = prepared.data.is_complex()
    # gfx936's FP64 accumulation doubles value traffic and register pressure,
    # but a row spanning more than two 16-wide chunks can exceed the public
    # FP32 tolerance.  The predicate is shared with the direct benchmark path.
    native_rocm_fp32 = alg != "row_subgroup_stream" and uses_native_rocm_fp32_accumulation(prepared)
    acc_dtype = (
        torch.float32
        if native_rocm_fp32
        else (
            torch.float64
            if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
            else torch.float32
        )
    )
    acc = tl.float64 if acc_dtype == torch.float64 else tl.float32
    view = lambda t: (
        torch.view_as_real(t.resolve_conj()).reshape(-1) if complex_input else t
    )
    data, vector, output_view = view(prepared.data), view(x), view(y)
    args = (data, prepared.kernel_indices, prepared.kernel_indptr, vector, output_view)
    channels = 2 if complex_input else 1
    if alg == "csr_subgroup_pair_mixed":
        c = config[alg]
        subgroup_pair_mixed_kernel[(triton.cdiv(m, c["rows_per_program"] * c["row_tiles_per_program"]),)](
            *args, m, x.stride(0), y.stride(0), R=c["rows_per_program"], V=c["lanes_per_row"],
            GROUPS=c["row_tiles_per_program"], CHUNKS=c["prefetch_chunks"],
            num_warps=c["num_warps"], num_stages=1, enable_fp_fusion=False)
        return y
    if alg == "row_subgroup_stream":
        c = config[alg]
        row_subgroup_stream_kernel[(triton.cdiv(m, c["rows_per_program"] * c["row_tiles_per_program"]),)](
            *args, m, R=c["rows_per_program"], V=c["lanes_per_row"],
            GROUPS=c["row_tiles_per_program"], STAGES=c["loop_num_stages"],
            COMPLEX=complex_input, ACC=acc, num_warps=c["num_warps"], enable_fp_fusion=False,
        )
        return y
    if alg in ("row_tile", "row_adaptive_split"):
        rows = None if plan is None else plan["short_rows"]
        n = m if rows is None else rows.numel()
        c = config["row_tile"]
        if n:
            grid = (triton.cdiv(n, c["rows_per_program"]),)
            launch_args = (
                *args,
                prepared.kernel_indptr if rows is None else rows,
                n,
            )
            if not _common._is_rocm_runtime():
                # Keep the pre-existing kernel for non-ROCm backends.  The
                # real-only specialization below is tuned for DCU wavefronts.
                row_tile_kernel[grid](
                    *launch_args,
                    INDEXED=rows is not None,
                    R=c["rows_per_program"],
                    V=c["lanes_per_row"],
                    STAGES=c["loop_num_stages"],
                    COMPLEX=complex_input,
                    ACC=acc,
                    num_warps=c["num_warps"],
                    enable_fp_fusion=False,
                )
            elif complex_input:
                row_tile_kernel[grid](
                    *launch_args,
                    INDEXED=rows is not None,
                    R=c["rows_per_program"],
                    V=c["lanes_per_row"],
                    STAGES=c["loop_num_stages"],
                    COMPLEX=True,
                    ACC=acc,
                    num_warps=c["num_warps"],
                    enable_fp_fusion=False,
                )
            else:
                avg_nnz = prepared.data.numel() / m
                fixed_steps = 0
                if alg == "row_tile" and rows is None:
                    # On gfx936, short FP32 rows spanning several vector
                    # loads benefit from a tile-local bound: a matrix-wide
                    # maximum makes every tile execute the same masked tail.
                    # Retain fixed bounds for one-load rows and 25-32 NNZ rows.
                    if (
                        prepared.data.dtype == torch.float32
                        and prepared.max_row_nnz <= 32
                        and (
                            prepared.max_row_nnz <= c["lanes_per_row"]
                            or prepared.max_row_nnz > 24
                        )
                    ) or (
                        prepared.data.dtype == torch.float64
                        and avg_nnz > 7.0
                        and prepared.max_row_nnz <= 32
                    ):
                        fixed_steps = triton.cdiv(
                            prepared.max_row_nnz, c["lanes_per_row"]
                        )
                row_tile_real_kernel[grid](
                    *launch_args,
                    INDEXED=rows is not None,
                    R=c["rows_per_program"],
                    V=c["lanes_per_row"],
                    STAGES=c["loop_num_stages"],
                    ACC=acc,
                    POSITION_64=not (
                        prepared.data.dtype == torch.float32
                        and prepared.kernel_indptr.dtype == torch.int32
                    ),
                    FIXED_STEPS=fixed_steps,
                    num_warps=c["num_warps"],
                    enable_fp_fusion=(prepared.data.dtype == torch.float32),
                )
    if alg in ("row_vector", "row_adaptive_split"):
        rows = None if plan is None else plan["mid_rows"]
        n = m if rows is None else rows.numel()
        c = config["row_vector"]
        if n:
            row_vector_kernel[(n,)](
                *args,
                prepared.kernel_indptr if rows is None else rows,
                n,
                INDEXED=rows is not None,
                B=c["block_nnz"],
                STAGES=c["loop_num_stages"],
                COMPLEX=complex_input,
                ACC=acc,
                num_warps=c["num_warps"],
                enable_fp_fusion=False,
            )
    if plan is not None:
        c = config["row_split_reduce"]
        partial = torch.empty(
            plan["total"] * channels, dtype=acc_dtype, device=y.device
        )
        prefix = plan["prefix"]
        if plan["total"]:
            segment_kernel[(plan["total"],)](
                data,
                prepared.kernel_indices,
                vector,
                plan["starts"],
                plan["lengths"],
                partial,
                B=c["block_nnz"],
                STAGES=c["loop_num_stages"],
                COMPLEX=complex_input,
                ACC=acc,
                num_warps=c["num_warps"],
                enable_fp_fusion=False,
            )
            for next_prefix, total in plan["levels"]:
                output = torch.empty(total * channels, dtype=acc_dtype, device=y.device)
                reduce_level_kernel[(total,)](
                    partial,
                    prefix,
                    next_prefix,
                    output,
                    m,
                    B=c["reduce_block_size"],
                    COMPLEX=complex_input,
                    num_warps=c["reduce_num_warps"],
                )
                partial, prefix = output, next_prefix
        finish_block = config["process"]["block_size"]
        finish_kernel[(triton.cdiv(m, finish_block),)](
            partial,
            prefix,
            plan["counts"],
            output_view,
            m,
            WRITE_EMPTY=alg == "row_split_reduce",
            HAS_PARTIAL=plan["total"] > 0,
            B=finish_block,
            COMPLEX=complex_input,
            num_warps=c["reduce_num_warps"],
        )
    return y


@triton.jit
def subgroup_pair_mixed_kernel(A, I, P, X, Y, M, XS, YS,
         R: tl.constexpr, V: tl.constexpr, GROUPS: tl.constexpr,
         CHUNKS: tl.constexpr):
    """Native compressed-axis mixed reduction with independent accumulation chains."""
    first = tl.program_id(0).to(tl.int64) * R * GROUPS
    lane = tl.arange(0, V)
    for group in range(GROUPS):
        row = first + group * R + tl.arange(0, R)
        valid = row < M
        start = tl.load(P + row, valid, 0).to(tl.int64)
        end = tl.load(P + row + 1, valid, 0).to(tl.int64)
        acc0 = tl.zeros((R, V), tl.float32)
        acc1 = tl.zeros((R, V), tl.float32)
        steps = tl.max(tl.cdiv(end - start, V * CHUNKS), 0)
        for step in range(steps):
            p0 = start[:, None] + step * V * CHUNKS + lane[None, :]
            m0 = valid[:, None] & (p0 < end[:, None])
            c0 = tl.load(I + p0, m0, 0).to(tl.int64)
            a0 = tl.load(A + p0, m0, 0)
            x0 = tl.load(X + c0 * XS, m0, 0)
            if CHUNKS == 2:
                p1 = p0 + V
                m1 = valid[:, None] & (p1 < end[:, None])
                c1 = tl.load(I + p1, m1, 0).to(tl.int64)
                a1 = tl.load(A + p1, m1, 0)
                x1 = tl.load(X + c1 * XS, m1, 0)
            acc0 += a0.to(tl.float32) * x0.to(tl.float32)
            if CHUNKS == 2:
                acc1 += a1.to(tl.float32) * x1.to(tl.float32)
        if CHUNKS == 2:
            acc0 += acc1
        tl.store(Y + row * YS, tl.sum(acc0, 1), valid)
