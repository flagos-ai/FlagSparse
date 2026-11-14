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

"""Native CSR kernels; complex pointers are interleaved real component views."""

import triton
import triton.language as tl


@triton.jit
def _nnz_owner(P, pos, low, high, WIDTH: tl.constexpr):
    # Always use a rank-one block, including endpoint searches (WIDTH=1).
    # Do not infer a block shape from a scalar tensor or reshape a scalar loop
    # predicate. Both bounds stay int64 across the loop's SSA back-edge.
    lo = tl.full((WIDTH,), 0, tl.int64) + low
    hi = tl.full((WIDTH,), 0, tl.int64) + high
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        ptr = tl.load(P + mid, lo < hi, 0)
        right = (lo < hi) & (ptr <= pos)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    return lo - 1


@triton.jit
def _nnz_panel_add(a, ha, b, hb):
    return tl.where(hb, b, a + b), ha | hb


@triton.jit
def nnz_segment_panel_atomic_kernel(A, I, P, B, C, M, N, NNZ,
                                    B0: tl.constexpr, B1: tl.constexpr,
                                    C0: tl.constexpr, C1: tl.constexpr,
                                    T: tl.constexpr, BN: tl.constexpr,
                                    COMPLEX: tl.constexpr, ACC: tl.constexpr,
                                    ALL_ATOMIC: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * T
    stop = tl.minimum(base + T, NNZ)
    lane = tl.arange(0, T)
    pos = base + lane
    valid = pos < stop
    upper = tl.cast(M, tl.int64) + 1
    first = tl.sum(_nnz_owner(P, base, 0, upper, 1), 0)
    last = tl.sum(_nnz_owner(P, stop - 1, first, upper, 1), 0)
    if first == last:
        owner = tl.full((T,), 0, tl.int64) + first
    else:
        owner = _nnz_owner(P, tl.minimum(pos, stop - 1), first, last + 2, T)
    owner = tl.where(valid, owner, -1)
    prev = tl.gather(owner, tl.maximum(lane - 1, 0), 0)
    following = tl.gather(owner, tl.minimum(lane + 1, T - 1), 0)
    head = (lane == 0) | (owner != prev)
    tail = valid & ((pos == stop - 1) | (owner != following))
    start_row = tl.load(P + owner, tail, 0).to(tl.int64)
    end_row = tl.load(P + owner + 1, tail, 0).to(tl.int64)
    complete = (start_row >= base) & (end_row <= stop)
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    col = tl.load(I + pos, valid, 0).to(tl.int64)
    bo = col[:, None] * B0 + ns[None, :] * B1
    mask = valid[:, None] & (ns[None, :] < N)
    flags = tl.broadcast_to(head[:, None], (T, BN))
    if COMPLEX:
        ar = tl.load(A + 2 * pos, valid, 0).to(ACC)
        ai = tl.load(A + 2 * pos + 1, valid, 0).to(ACC)
        br = tl.load(B + 2 * bo, mask, 0).to(ACC)
        bi = tl.load(B + 2 * bo + 1, mask, 0).to(ACC)
        re, _ = tl.associative_scan((ar[:, None] * br - ai[:, None] * bi, flags), 0, _nnz_panel_add)
        im, _ = tl.associative_scan((ar[:, None] * bi + ai[:, None] * br, flags), 0, _nnz_panel_add)
    else:
        a = tl.load(A + pos, valid, 0).to(ACC)
        b = tl.load(B + bo, mask, 0).to(ACC)
        re, _ = tl.associative_scan((a[:, None] * b, flags), 0, _nnz_panel_add)
    out_mask = tail[:, None] & (ns[None, :] < N)
    direct = out_mask & complete[:, None]
    atom = out_mask & ~complete[:, None]
    if ALL_ATOMIC:
        atom = out_mask
    co = owner[:, None] * C0 + ns[None, :] * C1
    if COMPLEX:
        if not ALL_ATOMIC:
            tl.store(C + 2 * co, re, direct)
            tl.store(C + 2 * co + 1, im, direct)
        tl.atomic_add(C + 2 * co, re, atom, sem="relaxed")
        tl.atomic_add(C + 2 * co + 1, im, atom, sem="relaxed")
    else:
        if not ALL_ATOMIC:
            tl.store(C + co, re, direct)
        tl.atomic_add(C + co, re, atom, sem="relaxed")


@triton.jit
def rows_kernel(A, I, P, B, C, Rows, Starts, Ends,
                M, N, B0: tl.constexpr, B1: tl.constexpr,
                C0: tl.constexpr, C1: tl.constexpr, COL_START,
                R: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                LIST: tl.constexpr, SEGMENT: tl.constexpr,
                COMPLEX: tl.constexpr, ACC: tl.constexpr):
    slots = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    valid = slots < M
    if LIST:
        rows = tl.load(Rows + slots, valid, 0).to(tl.int64)
    else:
        rows = slots
    if SEGMENT:
        starts = tl.load(Starts + slots, valid, 0).to(tl.int64)
        ends = tl.load(Ends + slots, valid, 0).to(tl.int64)
        out_rows = slots
    else:
        starts = tl.load(P + rows, valid, 0).to(tl.int64)
        ends = tl.load(P + rows + 1, valid, 0).to(tl.int64)
        out_rows = rows
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    ar = tl.full((R, BK, BN), 0, ACC)
    ai = tl.full((R, BK, BN), 0, ACC)
    longest = tl.max(ends - starts, 0)
    for base in range(0, longest, BK):
        pos = starts[:, None] + base + ks[None, :]
        mask = valid[:, None] & (pos < ends[:, None])
        cols = tl.load(I + pos, mask, 0).to(tl.int64)
        bo = cols[:, :, None] * B0 + (ns[None, None, :] + COL_START) * B1
        bm = mask[:, :, None] & (ns[None, None, :] < N)
        if COMPLEX:
            vr = tl.load(A + 2 * pos, mask, 0).to(ACC)
            vi = tl.load(A + 2 * pos + 1, mask, 0).to(ACC)
            br = tl.load(B + 2 * bo, bm, 0).to(ACC)
            bi = tl.load(B + 2 * bo + 1, bm, 0).to(ACC)
            ar += vr[:, :, None] * br - vi[:, :, None] * bi
            ai += vr[:, :, None] * bi + vi[:, :, None] * br
        else:
            v = tl.load(A + pos, mask, 0).to(ACC)
            b = tl.load(B + bo, bm, 0).to(ACC)
            ar += v[:, :, None] * b
    rr = tl.sum(ar, 1)
    co = out_rows[:, None] * C0 + ns[None, :] * C1
    cm = valid[:, None] & (ns[None, :] < N)
    if COMPLEX:
        ri = tl.sum(ai, 1)
        tl.store(C + 2 * co, rr, cm)
        tl.store(C + 2 * co + 1, ri, cm)
    else:
        tl.store(C + co, rr, cm)


@triton.jit
def row_panel_kernel(A, I, P, B, C, M, N,
                     B0: tl.constexpr, B1: tl.constexpr,
                     C0: tl.constexpr, C1: tl.constexpr,
                     R: tl.constexpr, BN: tl.constexpr, ACCUMULATORS: tl.constexpr,
                     COMPLEX: tl.constexpr, ACC: tl.constexpr, LOAD_NNZ: tl.constexpr = 1):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    valid = rows < M
    starts = tl.load(P + rows, valid, 0).to(tl.int64)
    ends = tl.load(P + rows + 1, valid, 0).to(tl.int64)
    longest = tl.max(ends - starts, 0)
    r0 = tl.zeros((R, BN), ACC)
    r1 = tl.zeros((R, BN), ACC)
    r2 = tl.zeros((R, BN), ACC)
    r3 = tl.zeros((R, BN), ACC)
    i0 = tl.zeros((R, BN), ACC)
    i1 = tl.zeros((R, BN), ACC)
    i2 = tl.zeros((R, BN), ACC)
    i3 = tl.zeros((R, BN), ACC)
    CHUNK: tl.constexpr = max(LOAD_NNZ, ACCUMULATORS)
    for base in range(0, longest, CHUNK):
        if LOAD_NNZ > 1:
            offsets = starts[:, None] + base + tl.arange(0, CHUNK)[None, :]
            active = valid[:, None] & (offsets < ends[:, None])
            indices = tl.load(I + offsets, active, 0).to(tl.int64)
            if COMPLEX:
                values_r = tl.load(A + 2 * offsets, active, 0).to(ACC)
                values_i = tl.load(A + 2 * offsets + 1, active, 0).to(ACC)
            else:
                values_r = tl.load(A + offsets, active, 0).to(ACC)
        for k in tl.static_range(CHUNK):
            pos = starts + base + k
            mask = valid & (pos < ends)
            if LOAD_NNZ > 1:
                pick = tl.full((R, 1), k, tl.int32)
                cols = tl.gather(indices, pick, 1).reshape((R,))
                vr = tl.gather(values_r, pick, 1).reshape((R,))
                if COMPLEX:
                    vi = tl.gather(values_i, pick, 1).reshape((R,))
            else:
                cols = tl.load(I + pos, mask, 0).to(tl.int64)
            bo = cols[:, None] * B0 + ns[None, :] * B1
            bm = mask[:, None] & (ns[None, :] < N)
            if COMPLEX:
                if LOAD_NNZ == 1:
                    vr = tl.load(A + 2 * pos, mask, 0).to(ACC)
                    vi = tl.load(A + 2 * pos + 1, mask, 0).to(ACC)
                br = tl.load(B + 2 * bo, bm, 0).to(ACC)
                bi = tl.load(B + 2 * bo + 1, bm, 0).to(ACC)
                real = vr[:, None] * br - vi[:, None] * bi
                imag = vr[:, None] * bi + vi[:, None] * br
            else:
                if LOAD_NNZ > 1:
                    v = vr
                else:
                    v = tl.load(A + pos, mask, 0).to(ACC)
                real = v[:, None] * tl.load(B + bo, bm, 0).to(ACC)
                imag = tl.zeros((R, BN), ACC)
            # Evaluate the static loop expression directly: annotating a local
            # constexpr here redefines it on each unrolled iteration.
            if k % ACCUMULATORS == 0:
                r0 += real
                i0 += imag
            elif k % ACCUMULATORS == 1:
                r1 += real
                i1 += imag
            elif k % ACCUMULATORS == 2:
                r2 += real
                i2 += imag
            else:
                r3 += real
                i3 += imag
    real = r0
    imag = i0
    if ACCUMULATORS >= 2:
        real += r1
        imag += i1
    if ACCUMULATORS == 4:
        real += r2 + r3
        imag += i2 + i3
    co = rows[:, None] * C0 + ns[None, :] * C1
    cm = valid[:, None] & (ns[None, :] < N)
    if COMPLEX:
        tl.store(C + 2 * co, real, cm)
        tl.store(C + 2 * co + 1, imag, cm)
    else:
        tl.store(C + co, real, cm)


@triton.jit
def reduce_kernel(X, Y, Starts, Ends, N, K: tl.constexpr, BN: tl.constexpr,
                  COMPLEX: tl.constexpr, ACC: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    start = tl.load(Starts + group).to(tl.int64)
    end = tl.load(Ends + group).to(tl.int64)
    ks = start + tl.arange(0, K)
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    off = ks[:, None] * N + ns[None, :]
    mask = (ks[:, None] < end) & (ns[None, :] < N)
    dst = group * N + ns
    if COMPLEX:
        re = tl.sum(tl.load(X + 2 * off, mask, 0).to(ACC), 0)
        im = tl.sum(tl.load(X + 2 * off + 1, mask, 0).to(ACC), 0)
        tl.store(Y + 2 * dst, re, ns < N)
        tl.store(Y + 2 * dst + 1, im, ns < N)
    else:
        re = tl.sum(tl.load(X + off, mask, 0).to(ACC), 0)
        tl.store(Y + dst, re, ns < N)


@triton.jit
def finish_kernel(X, C, Rows, Offsets, Counts, N, COL_START,
                  C0: tl.constexpr, C1: tl.constexpr, BN: tl.constexpr,
                  COMPLEX: tl.constexpr):
    slot = tl.program_id(0).to(tl.int64)
    row = tl.load(Rows + slot).to(tl.int64)
    count = tl.load(Counts + slot)
    src = tl.load(Offsets + slot).to(tl.int64)
    ns = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    xo = src * N + ns
    co = row * C0 + (ns + COL_START) * C1
    mask = (count > 0) & (ns < N)
    # Batches execute on one stream; each row is unique within a batch.
    if COMPLEX:
        re = tl.load(X + 2 * xo, mask, 0) + tl.load(C + 2 * co, mask, 0)
        im = tl.load(X + 2 * xo + 1, mask, 0) + tl.load(C + 2 * co + 1, mask, 0)
        tl.store(C + 2 * co, re, mask)
        tl.store(C + 2 * co + 1, im, mask)
    else:
        re = tl.load(X + xo, mask, 0) + tl.load(C + co, mask, 0)
        tl.store(C + co, re, mask)


@triton.jit
def panel_staged_mixed_kernel(A, I, P, B, C, M, N,
         B0: tl.constexpr, B1: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr,
         R: tl.constexpr, BN: tl.constexpr, LOAD: tl.constexpr,
         ACCS: tl.constexpr, COMPUTE_STAGE: tl.constexpr):
    """Mixed panel: staged storage loads, FP32 accumulation, one final store."""
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    col = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    valid = row < M
    start = tl.load(P + row, valid, 0).to(tl.int64)
    end = tl.load(P + row + 1, valid, 0).to(tl.int64)
    longest = tl.max(end - start, 0)
    a0 = tl.zeros((R, BN), tl.float32)
    a1 = tl.zeros((R, BN), tl.float32)
    a2 = tl.zeros((R, BN), tl.float32)
    a3 = tl.zeros((R, BN), tl.float32)
    # LOAD and ACCS are powers of two, so max is their least common multiple.
    U: tl.constexpr = max(LOAD, ACCS)
    for base in range(0, longest, U):
        for group in tl.static_range(U // LOAD):
            pos = start[:, None] + base + group * LOAD + tl.arange(0, LOAD)[None, :]
            mask = valid[:, None] & (pos < end[:, None])
            indices = tl.load(I + pos, mask, 0)
            values = tl.load(A + pos, mask, 0)
            if COMPUTE_STAGE:
                values = values.to(tl.float32)
            for k in tl.static_range(LOAD):
                pick = tl.full((R, 1), k, tl.int32)
                idx = tl.gather(indices, pick, 1).reshape((R,)).to(tl.int64)
                value = tl.gather(values, pick, 1).reshape((R,)).to(tl.float32)
                active = valid & (start + base + group * LOAD + k < end)
                rhs = tl.load(B + idx[:, None] * B0 + col[None, :] * B1,
                              active[:, None] & (col[None, :] < N), 0).to(tl.float32)
                product = value[:, None] * rhs
                if (group * LOAD + k) % ACCS == 0:
                    a0 += product
                elif (group * LOAD + k) % ACCS == 1:
                    a1 += product
                elif (group * LOAD + k) % ACCS == 2:
                    a2 += product
                else:
                    a3 += product
    if ACCS >= 2:
        a0 += a1
    if ACCS == 4:
        a0 += a2 + a3
    tl.store(C + row[:, None] * C0 + col[None, :] * C1, a0,
             valid[:, None] & (col[None, :] < N))
