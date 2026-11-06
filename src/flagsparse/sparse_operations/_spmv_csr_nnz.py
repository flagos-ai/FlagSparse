"""CSR NNZ execution: per-call mixed tiles and the existing gfx936 prepared path."""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _flagged_sum(lv, lf, rv, rf):
    return tl.where(rf, rv, lv + rv), lf | rf


@triton.jit
def _mixed_nnz_panel(A, I, P, B, C, Partial, Owners, Out,
                     M, NNZ, FIRST_TILE, ROW_FIRST, ROW_END, COL_FIRST, WIDTH,
                     BS0, BS1, CS0, CS1, OS0, OS1,
                     TILE: tl.constexpr, PANEL: tl.constexpr,
                     Bounds=None, FIXED: tl.constexpr = False):
    tile = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, TILE)
    pos = (FIRST_TILE + tile).to(tl.int64) * TILE + lane
    valid = pos < NNZ
    # upper_bound(P, pos) - 1 skips every empty row, including leading empties.
    if FIXED:
        first_owner = tl.load(Bounds + 2 * tile)
        last_owner = tl.load(Bounds + 2 * tile + 1)
        lo = tl.full((TILE,), 0, tl.int64) + first_owner + 1
        hi = tl.full((TILE,), 0, tl.int64) + last_owner + 1
    else:
        lo = tl.full((TILE,), 0, tl.int64)
        hi = tl.full((TILE,), M + 1, tl.int64)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        value = tl.load(P + mid, (lo < hi) & (mid <= M), NNZ + 1)
        right = value <= pos
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    row = lo - 1
    valid &= (row >= ROW_FIRST) & (row < ROW_END)
    start = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    ci = tl.load(I + pos, valid, 0).to(tl.int64)
    a = tl.load(A + pos, valid, 0).to(tl.float32)
    col = COL_FIRST + tl.arange(0, PANEL)
    b = tl.load(B + ci[:, None] * BS0 + col[None, :] * BS1,
                valid[:, None] & (col[None, :] < COL_FIRST + WIDTH), 0).to(tl.float32)
    head = (lane == 0) | (pos == start)
    total, _ = tl.associative_scan(
        (a[:, None] * b, tl.broadcast_to(head[:, None], (TILE, PANEL))),
        0, _flagged_sum)
    tail = valid & ((pos + 1 == end) | (lane == TILE - 1) | (pos + 1 == NNZ))
    crossing = (start // TILE) != ((end - 1) // TILE)
    tl.store(Out + row[:, None] * OS0 + col[None, :] * OS1,
             total, tail[:, None] & ~crossing[:, None] & (col[None, :] < COL_FIRST + WIDTH))
    first_row = tl.min(tl.where(lane == 0, row, 0x7fffffffffffffff), 0)
    slot = 2 * tile + (row != first_row).to(tl.int64)
    boundary = tail & crossing
    tl.store(Owners + slot, row, boundary)
    tl.store(Partial + slot[:, None] * PANEL + tl.arange(0, PANEL)[None, :],
             total, boundary[:, None])


@triton.jit
def _mixed_tile_bounds(P, Bounds, M, NNZ, FIRST, TILE: tl.constexpr):
    tile = tl.program_id(0).to(tl.int64)
    edge = tl.arange(0, 2)
    pos = tl.minimum((FIRST + tile) * TILE + edge * (TILE - 1), NNZ - 1)
    lo = tl.full((2,), 0, tl.int64)
    hi = tl.full((2,), M + 1, tl.int64)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        value = tl.load(P + mid, lo < hi, NNZ + 1)
        active = lo < hi
        lo = tl.where(active & (value <= pos), mid + 1, lo)
        hi = tl.where(active & (value > pos), mid, hi)
    tl.store(Bounds + 2 * tile + edge, lo - 1)


@triton.jit
def _fixed_boundary_reduce(Input, Owners, Output, NextOwners, P, C,
                           COUNT, M, ROW_FIRST, ROW_END,
                           TILE: tl.constexpr, WIDTH: tl.constexpr,
                           BLOCK: tl.constexpr, FINAL: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    pos = group * BLOCK + lane
    row = tl.load(Owners + pos, pos < COUNT, M).to(tl.int64)
    previous = tl.load(Owners + pos - 1, (pos > 0) & (pos < COUNT), -1)
    following = tl.load(Owners + pos + 1, pos + 1 < COUNT, M)
    col = tl.arange(0, WIDTH)
    values = tl.load(Input + pos[:, None] * WIDTH + col[None, :],
                     (pos < COUNT)[:, None], 0)
    head = (lane == 0) | (row != previous)
    sums, _ = tl.associative_scan(
        (values, tl.broadcast_to(head[:, None], (BLOCK, WIDTH))), 0, _flagged_sum)
    first = tl.load(Owners + group * BLOCK)
    last_pos = tl.minimum((group + 1) * BLOCK, COUNT) - 1
    last = tl.load(Owners + last_pos)
    before = tl.load(Owners + group * BLOCK - 1, group > 0, -1)
    after = tl.load(Owners + last_pos + 1, last_pos + 1 < COUNT, M)
    spill = ((row == first) & (first == before)) | ((row == last) & (last == after))
    tail = (pos < COUNT) & ((row != following) | (lane == BLOCK - 1))
    valid = tail & (row >= ROW_FIRST) & (row < ROW_END)
    begin = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    crossing = (begin < end) & (begin // TILE != (end - 1) // TILE)
    complete = valid & crossing & ~spill
    address = C + (row[:, None] - ROW_FIRST) * WIDTH + col[None, :]
    prior = tl.load(address, complete[:, None], 0)
    tl.store(address, prior + sums, complete[:, None])
    if not FINAL:
        # Keep sorted owners, even for zero-valued slots; no compaction or readback.
        edge = tl.arange(0, 2)
        tl.store(NextOwners + 2 * group + edge, tl.where(edge == 0, first, last))
        first_sum = tl.sum(tl.where((tail & spill & (row == first))[:, None], sums, 0), 0)
        last_sum = tl.sum(tl.where((tail & spill & (row == last) & (last != first))[:, None], sums, 0), 0)
        tl.store(Output + (2 * group) * WIDTH + col, first_sum)
        tl.store(Output + (2 * group + 1) * WIDTH + col, last_sum)


def _run_mixed_fixed(prepared, rhs, out, config, phases):
    """Fixed-capacity boundary tree; all execution data is local to this run."""
    block, panel = config["block_nnz"], config.get("block_n", 1)
    limit, reduction = config["workspace_bytes"], config["reduce_block_size"]
    if reduction < 4 or limit < 36 * panel:
        raise ValueError("fixed plan requires reduction >= 4 and 36 * panel workspace bytes")
    m, n = out.shape
    nnz = prepared.data.numel()
    stats = dict(plan_mode="fixed", boundary_slot_capacity=0, reduction_levels=0,
                 tile_batches=0, workspace_peak_bytes=0, descriptor_peak_bytes=0,
                 nnz_tile_count=triton.cdiv(nnz, block), plan_builds=1)
    if not m or not n:
        return stats
    panels_per_wave = min(4, triton.cdiv(n, panel), max(1, limit // (36 * panel)))
    row_wave = max(1, min(m, (limit - 32 * panel) // (4 * panel * panels_per_wave),
                          limit // (16 * panel * panels_per_wave)))
    launch = dict(num_warps=config["num_warps"], num_stages=1)
    for row_first in range(0, m, row_wave):
        row_end = min(m, row_first + row_wave)
        rows = row_end - row_first
        with phases.measure("process_gpu_ms"):
            begin, end = prepared.kernel_indptr[[row_first, row_end]].tolist()
        first, last = begin // block, triton.cdiv(end, block)
        for column_first in range(0, n, panels_per_wave * panel):
            panels = min(panels_per_wave, triton.cdiv(n - column_first, panel))
            accumulator_bytes = panels * rows * panel * 4
            batch_tiles = max(1, (limit - accumulator_bytes) // (32 * panel))
            with phases.measure("compute_ms"):
                accumulators = torch.zeros((panels, rows, panel), device=out.device, dtype=torch.float32)
            stats["workspace_peak_bytes"] = max(stats["workspace_peak_bytes"], accumulator_bytes)
            for tile_first in range(first, last, batch_tiles):
                count = min(batch_tiles, last - tile_first)
                slots = 2 * count
                with phases.measure("process_gpu_ms"):
                    bounds = torch.empty((slots,), device=out.device, dtype=torch.int64)
                    _mixed_tile_bounds[(count,)](prepared.kernel_indptr, bounds, m, nnz,
                                                tile_first, block, **launch)
                for pi in range(panels):
                    column = column_first + pi * panel
                    width = min(panel, n - column)
                    accumulator = accumulators[pi]
                    with phases.measure("compute_ms"):
                        current = torch.zeros((slots, panel), device=out.device, dtype=torch.float32)
                        owners = bounds.clone()
                        _mixed_nnz_panel[(count,)](
                            prepared.data, prepared.kernel_indices, prepared.kernel_indptr,
                            rhs, accumulator, current, owners, out, m, nnz, tile_first,
                            row_first, row_end, column, width, *rhs.stride(), *accumulator.stride(),
                            *out.stride(), block, panel, Bounds=bounds, FIXED=True, **launch)
                    stats["tile_batches"] += 1
                    stats["boundary_slot_capacity"] = max(stats["boundary_slot_capacity"], slots)
                    level, capacity = 0, slots
                    while capacity:
                        groups = triton.cdiv(capacity, reduction)
                        final = groups == 1
                        next_capacity = 0 if final else 2 * groups
                        with phases.measure("compute_ms"):
                            target = torch.empty((next_capacity, panel), device=out.device, dtype=torch.float32)
                            target_rows = torch.empty((next_capacity,), device=out.device, dtype=torch.int64)
                            _fixed_boundary_reduce[(groups,)](
                                current, owners, target, target_rows, prepared.kernel_indptr,
                                accumulator, capacity, m, row_first, row_end, block, panel,
                                reduction, final, **launch)
                        stats["workspace_peak_bytes"] = max(stats["workspace_peak_bytes"],
                            accumulator_bytes + (capacity + next_capacity) * panel * 4)
                        stats["descriptor_peak_bytes"] = max(stats["descriptor_peak_bytes"],
                            (slots + capacity + next_capacity) * 8)
                        current, owners = target, target_rows
                        capacity = next_capacity
                        level += 1
                    del current, owners, target, target_rows, accumulator
                    stats["reduction_levels"] = max(stats["reduction_levels"], level)
                del bounds
            with phases.measure("compute_ms"):
                for pi in range(panels):
                    column = column_first + pi * panel
                    _mixed_nnz_output[(rows,)](
                        accumulators[pi], prepared.kernel_indptr, out, row_first, row_end,
                        column, min(panel, n - column), *out.stride(), block, panel, **launch)
            del accumulators
    return stats


@triton.jit
def _mixed_nnz_output(C, P, Out, ROW_FIRST, ROW_END, COLUMN, WIDTH,
                      OS0, OS1, TILE: tl.constexpr, PANEL: tl.constexpr):
    row = ROW_FIRST + tl.program_id(0).to(tl.int64)
    begin = tl.load(P + row).to(tl.int64)
    end = tl.load(P + row + 1).to(tl.int64)
    owned = (begin == end) | (begin // TILE != (end - 1) // TILE)
    col = tl.arange(0, PANEL)
    value = tl.load(C + (row - ROW_FIRST) * PANEL + col)
    tl.store(Out + row * OS0 + (COLUMN + col) * OS1, value, owned & (col < WIDTH))


@triton.jit
def _compact_boundaries(Owners, Offsets, Selected, Rows, COUNT, BLOCK: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Owners + pos, pos < COUNT, -1)
    slot = tl.load(Offsets + pos, pos < COUNT, 0) - 1
    tl.store(Selected + slot, pos, (pos < COUNT) & (row >= 0))
    tl.store(Rows + slot, row, (pos < COUNT) & (row >= 0))


@triton.jit
def _mixed_boundary_reduce(Input, Output, Starts, Ends, WIDTH: tl.constexpr,
                           BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    begin = tl.load(Starts + group).to(tl.int64)
    end = tl.load(Ends + group).to(tl.int64)
    pos = begin + tl.arange(0, BLOCK)
    col = tl.arange(0, WIDTH)
    value = tl.load(Input + pos[:, None] * WIDTH + col[None, :],
                    (pos < end)[:, None], 0)
    tl.store(Output + group * WIDTH + col, tl.sum(value, 0))


@triton.jit
def _mixed_boundary_finish(Input, Rows, C, ROW_FIRST, WIDTH: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    row = tl.load(Rows + group).to(tl.int64) - ROW_FIRST
    col = tl.arange(0, WIDTH)
    address = C + row * WIDTH + col
    tl.store(address, tl.load(address) + tl.load(Input + group * WIDTH + col))


def run_mixed(prepared, rhs, out, config, phases):
    """Per-call native CSR NNZ tiles; shared SpMV/SpMM FP32 boundary reduction.

    Numeric workspace includes the row-wave accumulator and simultaneous partial,
    compacted and reduction buffers. Output storage is not workspace. Row waves
    also bound half-output accumulation when even a full column exceeds the cap.
    """
    if config.get("plan_mode", "compact") == "fixed":
        return _run_mixed_fixed(prepared, rhs, out, config, phases)
    block = config["block_nnz"]
    panel = config.get("block_n", 1)
    limit = config["workspace_bytes"]
    reduce_block = config["reduce_block_size"]
    if limit < 4 * panel * 9:
        raise ValueError("workspace_bytes cannot hold one row and one NNZ tile")
    m, n = out.shape
    nnz = prepared.data.numel()
    stats = dict(plan_mode="compact", boundary_record_count=0, reduction_levels=0,
                 tile_batches=0, workspace_peak_bytes=0, descriptor_peak_bytes=0,
                 nnz_tile_count=triton.cdiv(nnz, block), plan_builds=1)
    row_wave = max(1, min(m, limit // (16 * panel), (limit - 32 * panel) // (4 * panel)))
    for column in range(0, n, panel):
        width = min(panel, n - column)
        for row_first in range(0, m, row_wave):
            row_end = min(m, row_first + row_wave)
            rows = row_end - row_first
            accumulator_bytes = rows * panel * 4
            stats["workspace_peak_bytes"] = max(stats["workspace_peak_bytes"], accumulator_bytes)
            # At most 2 records/tile, with four simultaneous numeric buffers.
            batch_tiles = max(1, (limit - accumulator_bytes) // (32 * panel))
            with phases.measure("process_gpu_ms"):
                begin, end = prepared.kernel_indptr[[row_first, row_end]].tolist()
            with phases.measure("compute_ms"):
                accumulator = torch.zeros((rows, panel), device=out.device, dtype=torch.float32)
            first, last = begin // block, triton.cdiv(end, block)
            for tile_first in range(first, last, batch_tiles):
                count = min(batch_tiles, last - tile_first)
                slots = 2 * count
                with phases.measure("process_gpu_ms"):
                    owners = torch.full((slots,), -1, device=out.device, dtype=torch.int64)
                with phases.measure("compute_ms"):
                    partial = torch.empty((slots, panel), device=out.device, dtype=torch.float32)
                    _mixed_nnz_panel[(count,)](
                        prepared.data, prepared.kernel_indices, prepared.kernel_indptr,
                        rhs, accumulator, partial, owners, out, m, nnz, tile_first,
                        row_first, row_end, column, width, *rhs.stride(), *accumulator.stride(), *out.stride(),
                        block, panel, num_warps=config["num_warps"], num_stages=1)
                with phases.measure("process_gpu_ms"):
                    valid = owners >= 0
                    offsets = valid.to(torch.int64).cumsum(0)
                    records = int(offsets[-1].item())
                    selected = torch.empty(records, device=out.device, dtype=torch.int64)
                    active_rows = torch.empty_like(selected)
                    _compact_boundaries[(triton.cdiv(slots, 256),)](
                        owners, offsets, selected, active_rows, slots, 256,
                        num_warps=config["num_warps"])
                stats["boundary_record_count"] += records
                stats["tile_batches"] += 1
                stats["workspace_peak_bytes"] = max(stats["workspace_peak_bytes"], accumulator_bytes + 4 * slots * panel * 4)
                stats["descriptor_peak_bytes"] = max(stats["descriptor_peak_bytes"], slots * 128)
                with phases.measure("compute_ms"):
                    current = partial[selected]
                    del partial
                level = 0
                while records:
                    with phases.measure("process_gpu_ms"):
                        unique, lengths = torch.unique_consecutive(active_rows, return_counts=True)
                        if unique.numel() == records:
                            break
                        ends = lengths.cumsum(0)
                        starts = ends - lengths
                        groups = (lengths + reduce_block - 1) // reduce_block
                        group_ends = groups.cumsum(0)
                        group_first = group_ends - groups
                        mapping = torch.repeat_interleave(torch.arange(unique.numel(), device=out.device), groups)
                        rank = torch.arange(mapping.numel(), device=out.device) - group_first[mapping]
                        starts = starts[mapping] + rank * reduce_block
                        ends = torch.minimum(starts + reduce_block, ends[mapping])
                        next_rows = unique[mapping]
                    with phases.measure("compute_ms"):
                        reduced = torch.empty((mapping.numel(), panel), device=out.device, dtype=torch.float32)
                        _mixed_boundary_reduce[(mapping.numel(),)](
                            current, reduced, starts, ends, panel, reduce_block,
                            num_warps=config["num_warps"])
                        current = reduced
                        del reduced
                    active_rows = next_rows
                    records = mapping.numel()
                    level += 1
                with phases.measure("compute_ms"):
                    if records:
                        _mixed_boundary_finish[(records,)](
                            current, active_rows, accumulator, row_first, panel,
                            num_warps=config["num_warps"])
                    del current
                stats["reduction_levels"] = max(stats["reduction_levels"], level)
            with phases.measure("compute_ms"):
                _mixed_nnz_output[(rows,)](
                    accumulator, prepared.kernel_indptr, out, row_first, row_end,
                    column, width, *out.stride(), block, panel,
                    num_warps=config["num_warps"])
                del accumulator
    return stats


@triton.jit
def _segmented_add(left_row, left_value, right_row, right_value):
    return right_row, right_value + tl.where(left_row == right_row, left_value, 0.)


@triton.jit
def _nnz_kernel(A, CI, RP, ROW, X, Y, EMPTY, NE: tl.constexpr,
                NNZ: tl.constexpr, B: tl.constexpr, ACC: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    pos = pid * B + lane
    empty_row = tl.load(EMPTY + pos, pos < NE, 0)
    tl.store(Y + empty_row, 0., pos < NE)
    valid = pos < NNZ
    row = tl.load(ROW + pos, valid, -1)
    col = tl.load(CI + pos, valid, 0)
    value = tl.load(A + pos, valid, 0).to(ACC)
    vector = tl.load(X + col, valid, 0).to(ACC)
    _, total = tl.associative_scan((row, value * vector), 0, _segmented_add)
    next_row = tl.load(ROW + pos + 1, pos + 1 < NNZ, -1)
    # Only the block containing a row's last nonzero owns its output.
    boundary = valid & (row != next_row)
    first_row = tl.load(ROW + pid * B, pid * B < NNZ, 0)
    start = tl.load(RP + first_row)
    end = tl.load(RP + first_row + 1)
    prefix = tl.full((), 0., ACC)
    if (start < pid * B) & (end <= (pid + 1) * B) & (pid * B < NNZ):
        # Recompute the short prefix crossing a block boundary, avoiding an
        # atomic update or a second launch to merge a partial-sum buffer.
        for offset in range(start, pid * B, 128):
            index = offset + tl.arange(0, 128)
            active = index < pid * B
            column = tl.load(CI + index, active, 0)
            a = tl.load(A + index, active, 0).to(ACC)
            x = tl.load(X + column, active, 0).to(ACC)
            prefix += tl.sum(a * x, 0)
    total += tl.where(row == first_row, prefix, 0.)
    tl.store(Y + row, total, boundary)


def prepare(prepared):
    """Build index-only metadata once; values and vectors remain live inputs."""
    device = prepared.data.device
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    rows = torch.repeat_interleave(
        torch.arange(prepared.n_rows, device=device, dtype=torch.int32),
        prepared.row_lengths,
        output_size=prepared.data.numel(),
    )
    empty = torch.nonzero(prepared.row_lengths == 0).flatten().to(torch.int32)
    torch.cuda.synchronize(device)
    return {
        "rows": rows,
        "empty": empty,
        "prepare_ms": (time.perf_counter() - start) * 1000,
        "metadata_bytes": (rows.numel() + empty.numel()) * 4,
    }


def bind(prepared, x, out):
    """Bind the same native launch used by the public API and benchmarks."""
    plan = prepared.rocm_nnz_plan
    nnz = prepared.data.numel()
    empty_count = plan["empty"].numel()
    block = 256
    grid = (triton.cdiv(max(nnz, empty_count), block),)
    acc = tl.float32 if prepared.data.dtype == torch.float32 else tl.float64

    def run():
        if grid[0]:
            _nnz_kernel[grid](
                prepared.data, prepared.kernel_indices, prepared.kernel_indptr,
                plan["rows"], x, out, plan["empty"], empty_count, nnz, block, acc,
                num_warps=2,
            )
        return out

    return run
