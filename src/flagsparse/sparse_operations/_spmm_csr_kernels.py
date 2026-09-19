"""Native CSR kernels; complex pointers are interleaved real component views."""

import triton
import triton.language as tl


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
