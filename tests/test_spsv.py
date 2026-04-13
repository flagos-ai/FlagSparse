"""SpSV tests: synthetic triangular systems and optional .mtx (CSR/COO).

与 CSR 相同的计时列、PyTorch 稠密参考、CSV 字段与 PASS 判定；COO 测试时 CuPy 基线使用
``coo_matrix``（与 FlagSparse COO 输入同构），CSR 测试时仍用 ``csr_matrix``。
"""

import argparse
import csv
import glob
import os

import torch

import flagsparse as fs

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cpx_spsolve_triangular
except Exception:
    cp = None
    cpx_sparse = None
    cpx_spsolve_triangular = None

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
TEST_SIZES = [256, 512, 1024, 2048]
WARMUP = 5
ITERS = 20

DENSE_REF_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
FLOAT16_LIMIT = 65504.0
COMPLEX32_DTYPE = getattr(torch, "complex32", None)
if COMPLEX32_DTYPE is None:
    COMPLEX32_DTYPE = getattr(torch, "chalf", None)

# CSR 完整组合覆盖（在原 csv-csr 逻辑外新增，不影响原入口）
CSR_FULL_VALUE_DTYPES = [
    torch.float32,
    torch.float64,
]
if COMPLEX32_DTYPE is not None:
    CSR_FULL_VALUE_DTYPES.append(COMPLEX32_DTYPE)
CSR_FULL_VALUE_DTYPES.append(torch.complex64)
CSR_FULL_INDEX_DTYPES = [torch.int32, torch.int64]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _tol_for_dtype(dtype):
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        return 5e-3, 5e-3
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-2
    return 1e-12, 1e-10


def _randn_by_dtype(n, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(n, dtype=dtype, device=device)
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        pair = torch.randn((n, 2), dtype=torch.float16, device=device) * 0.1
        return torch.view_as_complex(pair)
    base = torch.float32
    real = torch.randn(n, dtype=base, device=device)
    imag = torch.randn(n, dtype=base, device=device)
    return torch.complex(real, imag)


def _dense_ref_dtype(dtype):
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        return torch.complex64
    return dtype


def _tensor_from_scalar_values(values, dtype, device):
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        real = torch.clamp(
            torch.tensor(values, dtype=torch.float32, device=device),
            min=-FLOAT16_LIMIT,
            max=FLOAT16_LIMIT,
        ).to(torch.float16)
        imag = torch.zeros_like(real)
        return torch.view_as_complex(torch.stack([real, imag], dim=-1).contiguous())
    return torch.tensor(values, dtype=dtype, device=device)


def _safe_cast_tensor(tensor, dtype):
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        real = torch.clamp(tensor.real, min=-FLOAT16_LIMIT, max=FLOAT16_LIMIT).to(torch.float16)
        imag = torch.clamp(tensor.imag, min=-FLOAT16_LIMIT, max=FLOAT16_LIMIT).to(torch.float16)
        return torch.view_as_complex(torch.stack([real, imag], dim=-1).contiguous())
    return tensor.to(dtype)


def _cast_real_tensor_to_value_dtype(values, value_dtype):
    if COMPLEX32_DTYPE is not None and value_dtype == COMPLEX32_DTYPE:
        real = torch.clamp(values, min=-FLOAT16_LIMIT, max=FLOAT16_LIMIT).to(torch.float16)
        imag = torch.zeros_like(real)
        return torch.view_as_complex(torch.stack([real, imag], dim=-1).contiguous())
    return values.to(value_dtype)


def _cupy_ref_inputs(data, b):
    if COMPLEX32_DTYPE is not None and data.dtype == COMPLEX32_DTYPE:
        return data.to(torch.complex64), b.to(torch.complex64)
    return data, b


def _compare_view(tensor, value_dtype):
    if COMPLEX32_DTYPE is not None and value_dtype == COMPLEX32_DTYPE:
        return tensor.to(torch.complex64)
    return tensor


def _supported_csr_full_ops(value_dtype, index_dtype):
    if value_dtype not in CSR_FULL_VALUE_DTYPES:
        return []
    if index_dtype == torch.int32:
        return ["NON", "TRANS"]
    if index_dtype == torch.int64:
        return ["NON"]
    return []


def _allow_dense_pytorch_ref(shape, dtype):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    dense_bytes = n_rows * n_cols * elem_bytes
    return dense_bytes <= DENSE_REF_MAX_BYTES


def _build_random_triangular_csr(n, value_dtype, index_dtype, device, lower=True):
    """Build a well-conditioned triangular CSR (float32/float64)."""
    max_bandwidth = max(4, min(n, 16))
    rows_host = []
    cols_host = []
    vals_host = []
    if value_dtype == torch.float32:
        base_real_dtype = torch.float32
    elif value_dtype == torch.float64:
        base_real_dtype = torch.float64
    elif COMPLEX32_DTYPE is not None and value_dtype == COMPLEX32_DTYPE:
        base_real_dtype = torch.float16
    else:
        base_real_dtype = torch.float32

    for i in range(n):
        if lower:
            cand_cols = list(range(0, i + 1))
        else:
            cand_cols = list(range(i, n))
        if not cand_cols:
            cand_cols = [i]
        diag_col = i
        off_cand = [c for c in cand_cols if c != diag_col]
        k_off = min(len(off_cand), max_bandwidth - 1)
        if k_off > 0:
            perm = torch.randperm(len(off_cand))[:k_off].tolist()
            off_cols = [off_cand[j] for j in perm]
        else:
            off_cols = []
        off_vals_real = torch.randn(len(off_cols), dtype=base_real_dtype).mul_(0.01)
        sum_abs = float(torch.sum(torch.abs(off_vals_real)).item()) if off_vals_real.numel() else 0.0
        diag_val = sum_abs + 1.0
        rows_host.append(i)
        cols_host.append(diag_col)
        vals_host.append(diag_val)
        for c, v in zip(off_cols, off_vals_real.tolist()):
            rows_host.append(i)
            cols_host.append(int(c))
            vals_host.append(v)

    rows_t = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols_t = torch.tensor(cols_host, dtype=torch.int64, device=device)
    vals_t = _cast_real_tensor_to_value_dtype(
        torch.tensor(vals_host, dtype=base_real_dtype, device=device),
        value_dtype,
    )
    order = torch.argsort(rows_t * max(1, n) + cols_t)
    rows_t = rows_t[order]
    cols_t = cols_t[order]
    vals_t = vals_t[order]
    nnz_per_row = torch.bincount(rows_t, minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = cols_t.to(index_dtype)
    return vals_t, indices, indptr, (n, n)


def _csr_to_dense(data, indices, indptr, shape):
    n_rows, n_cols = shape
    coo_data = data.to(torch.complex64) if COMPLEX32_DTYPE is not None and data.dtype == COMPLEX32_DTYPE else data
    row_ind = torch.repeat_interleave(
        torch.arange(n_rows, device=coo_data.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    coo = torch.sparse_coo_tensor(
        torch.stack([row_ind, indices.to(torch.int64)]),
        coo_data,
        (n_rows, n_cols),
        device=coo_data.device,
    ).coalesce()
    return coo.to_dense()


def _csr_to_coo(data, indices, indptr, shape):
    n_rows = int(shape[0])
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    col = indices.to(torch.int64)
    return data, row, col


def _csr_transpose(data, indices, indptr, shape):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_cols + 1, dtype=torch.int64, device=data.device),
        )

    row, col = _csr_to_coo(data, indices, indptr, shape)[1:]
    row_t = col
    col_t = row
    key = row_t * max(1, n_rows) + col_t
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)

    row_t = row_t[order]
    col_t = col_t[order]
    data_t = data[order]
    nnz_per_row = torch.bincount(row_t, minlength=n_cols)
    indptr_t = torch.zeros(n_cols + 1, dtype=torch.int64, device=data.device)
    indptr_t[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_t, col_t.to(torch.int64), indptr_t


def _load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = []
    header_info = None
    mm_field = "real"
    mm_symmetry = "general"
    for line in lines:
        line = line.strip()
        if line.startswith("%%MatrixMarket"):
            parts = line.split()
            if len(parts) >= 5:
                mm_field = parts[3].lower()
                mm_symmetry = parts[4].lower()
            continue
        if line.startswith("%"):
            continue
        if not header_info and line:
            parts = line.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if line:
            data_lines.append(line)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info
    if n_rows != n_cols:
        raise ValueError("SpSV requires square matrices")
    row_maps = [dict() for _ in range(n_rows)]

    def _accum(r, c, v):
        row = row_maps[r]
        row[c] = row.get(c, 0.0) + v

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if len(parts) >= 3:
            v = float(parts[2])
        elif mm_field == "pattern":
            v = 1.0
        else:
            continue
        _accum(r, c, v)
        if mm_symmetry in ("symmetric", "hermitian") and r != c:
            _accum(c, r, v)
        elif mm_symmetry == "skew-symmetric" and r != c:
            _accum(c, r, -v)

    for r in range(n_rows):
        row = row_maps[r]
        lower_row = {}
        off_abs_sum = 0.0
        for c, v in row.items():
            if c < r:
                lower_row[c] = lower_row.get(c, 0.0) + v
                off_abs_sum += abs(v)
        lower_row[r] = off_abs_sum + 1.0
        row_maps[r] = lower_row

    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))
    data = _tensor_from_scalar_values(vals_s, dtype, device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _coo_inputs_for_csv(data, indices, indptr, shape, coo_mode):
    """Sorted COO from CSR; optional shuffle/duplicate for csr|auto (与原先 CSV 行为一致)."""
    data_c, row_c, col_c = _csr_to_coo(data, indices, indptr, shape)
    if coo_mode in ("csr", "auto"):
        if data_c.numel() == 0:
            return data_c, row_c, col_c
        if data_c.numel() <= 2_000_000:
            left = data_c * 0.25
            right = data_c - left
            data_dup = torch.cat([left, right], dim=0)
            row_dup = torch.cat([row_c, row_c], dim=0)
            col_dup = torch.cat([col_c, col_c], dim=0)
            perm = torch.randperm(data_dup.numel(), device=data_c.device)
            return data_dup[perm], row_dup[perm], col_dup[perm]
        perm = torch.randperm(data_c.numel(), device=data_c.device)
        return data_c[perm], row_c[perm], col_c[perm]
    return data_c, row_c, col_c


def _build_rhs_for_csr_op(data, indices, indptr, x_true, shape, op_mode):
    if COMPLEX32_DTYPE is not None and data.dtype == COMPLEX32_DTYPE:
        data_ref = data.to(torch.complex64)
        x_ref = x_true.to(torch.complex64)
        if op_mode == "NON":
            b_ref, _ = fs.flagsparse_spmv_csr(
                data_ref, indices, indptr, x_ref, shape, return_time=True
            )
            return _safe_cast_tensor(b_ref, x_true.dtype)
        if op_mode == "TRANS":
            data_t, indices_t, indptr_t = _csr_transpose(data_ref, indices, indptr, shape)
            b_ref, _ = fs.flagsparse_spmv_csr(
                data_t,
                indices_t.to(indices.dtype),
                indptr_t.to(indptr.dtype),
                x_ref,
                (shape[1], shape[0]),
                return_time=True,
            )
            return _safe_cast_tensor(b_ref, x_true.dtype)
        raise ValueError("op_mode must be 'NON' or 'TRANS'")
    if op_mode == "NON":
        b, _ = fs.flagsparse_spmv_csr(
            data, indices, indptr, x_true, shape, return_time=True
        )
        return b
    if op_mode == "TRANS":
        data_t, indices_t, indptr_t = _csr_transpose(data, indices, indptr, shape)
        b, _ = fs.flagsparse_spmv_csr(
            data_t,
            indices_t.to(indices.dtype),
            indptr_t.to(indptr.dtype),
            x_true,
            (shape[1], shape[0]),
            return_time=True,
        )
        return b
    raise ValueError("op_mode must be 'NON' or 'TRANS'")


def _cupy_spsolve_lower_csr_or_coo(
    fmt,
    data,
    indices,
    indptr,
    shape,
    b,
    warmup,
    iters,
):
    """Triangular solve via CuPy: CSR or COO storage. Returns (ms, x_torch) or (None, None)."""
    if (
        cp is None
        or cpx_sparse is None
        or cpx_spsolve_triangular is None
    ):
        return None, None
    try:
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b.contiguous()))
        if fmt == "COO":
            dc, rr, cc = _csr_to_coo(data, indices, indptr, shape)
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(dc.contiguous()))
            row_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(rr.to(torch.int64).contiguous())
            )
            col_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(cc.to(torch.int64).contiguous())
            )
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
            idx_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indices.to(torch.int64).contiguous())
            )
            ptr_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indptr.contiguous())
            )
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        for _ in range(warmup):
            _ = cpx_spsolve_triangular(
                A_cp, b_cp, lower=True, unit_diagonal=False
            )
        cp.cuda.runtime.deviceSynchronize()
        t0 = cp.cuda.Event()
        t1 = cp.cuda.Event()
        t0.record()
        for _ in range(iters):
            x_cu = cpx_spsolve_triangular(
                A_cp, b_cp, lower=True, unit_diagonal=False
            )
        t1.record()
        t1.synchronize()
        cupy_ms = cp.cuda.get_elapsed_time(t0, t1) / iters
        x_cu_t = torch.utils.dlpack.from_dlpack(x_cu.toDlpack())
        if COMPLEX32_DTYPE is not None and b.dtype == COMPLEX32_DTYPE:
            x_cu_t = x_cu_t.to(torch.complex64)
        else:
            x_cu_t = x_cu_t.to(b.dtype)
        return cupy_ms, x_cu_t
    except Exception:
        return None, None


def _cupy_spsolve_csr_with_op(data, indices, indptr, shape, b, op_mode):
    if (
        cp is None
        or cpx_sparse is None
        or cpx_spsolve_triangular is None
    ):
        return None, None
    try:
        data_ref, b_ref = _cupy_ref_inputs(data, b)
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref.contiguous()))
        idx_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indices.to(torch.int64).contiguous())
        )
        ptr_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indptr.to(torch.int64).contiguous())
        )
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_ref.contiguous()))
        A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        if op_mode == "TRANS":
            A_eff = A_cp.transpose().tocsr()
            lower_eff = False
        else:
            A_eff = A_cp
            lower_eff = True

        for _ in range(WARMUP):
            _ = cpx_spsolve_triangular(
                A_eff, b_cp, lower=lower_eff, unit_diagonal=False
            )
        cp.cuda.runtime.deviceSynchronize()
        c0 = cp.cuda.Event()
        c1 = cp.cuda.Event()
        c0.record()
        for _ in range(ITERS):
            x_cp = cpx_spsolve_triangular(
                A_eff, b_cp, lower=lower_eff, unit_diagonal=False
            )
        c1.record()
        c1.synchronize()
        ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
        x_t = torch.utils.dlpack.from_dlpack(x_cp.toDlpack()).to(b.dtype)
        return ms, x_t
    except Exception:
        return None, None


def run_spsv_synthetic_all():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    sep = "=" * 110
    print(sep)
    print("FLAGSPARSE SpSV BENCHMARK (synthetic triangular systems, CSR + COO)")
    print(sep)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iters: {ITERS}")
    print()

    hdr = (
        f"{'Fmt':>5} {'N':>6} {'FlagSparse(ms)':>14} {'PyTorch(ms)':>12} {'CuPy(ms)':>10} "
        f"{'FS/PT':>8} {'FS/CU':>8} {'Status':>8} {'Err(PT)':>12} {'Err(CU)':>12}"
    )

    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 110)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12} | "
                f"Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 110)
            print(hdr)
            print("-" * 110)
            for n in TEST_SIZES:
                for fmt in ("CSR", "COO"):
                    data, indices, indptr, shape = _build_random_triangular_csr(
                        n, value_dtype, index_dtype, device, lower=True
                    )
                    A_dense = _csr_to_dense(
                        data, indices.to(torch.int64), indptr, shape
                    )
                    x_true = torch.randn(n, dtype=value_dtype, device=device)
                    b = A_dense @ x_true

                    torch.cuda.synchronize()
                    if fmt == "CSR":
                        x, t_ms = fs.flagsparse_spsv_csr(
                            data,
                            indices,
                            indptr,
                            b,
                            shape,
                            lower=True,
                            return_time=True,
                        )
                    else:
                        dc, rr, cc = _csr_to_coo(data, indices, indptr, shape)
                        x, t_ms = fs.flagsparse_spsv_coo(
                            dc,
                            rr,
                            cc,
                            b,
                            shape,
                            lower=True,
                            coo_mode="auto",
                            return_time=True,
                        )
                    torch.cuda.synchronize()

                    A_ref = A_dense
                    b_ref = b
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(True)
                    e1 = torch.cuda.Event(True)
                    e0.record()
                    x_pt = torch.linalg.solve(
                        A_ref, b_ref.unsqueeze(1)
                    ).squeeze(1)
                    e1.record()
                    torch.cuda.synchronize()
                    pytorch_ms = e0.elapsed_time(e1)
                    err_pt = float(torch.max(torch.abs(x - x_pt)).item()) if n > 0 else 0.0

                    cupy_ms = None
                    err_cu = None
                    x_cu_t = None
                    if value_dtype in (torch.float32, torch.float64):
                        cupy_ms, x_cu_t = _cupy_spsolve_lower_csr_or_coo(
                            fmt,
                            data,
                            indices,
                            indptr,
                            shape,
                            b,
                            WARMUP,
                            ITERS,
                        )
                        if x_cu_t is not None and n > 0:
                            err_cu = float(
                                torch.max(torch.abs(x - x_cu_t)).item()
                            )

                    atol, rtol = _tol_for_dtype(value_dtype)
                    ok_pt = torch.allclose(x, x_pt, atol=atol, rtol=rtol)
                    ok_cu = (
                        True
                        if x_cu_t is None
                        else torch.allclose(x, x_cu_t, atol=atol, rtol=rtol)
                    )
                    ok = ok_pt or ok_cu
                    status = "PASS" if ok else "FAIL"
                    if not ok:
                        failed += 1
                    total += 1

                    fs_vs_pt = (
                        (pytorch_ms / t_ms) if (t_ms and t_ms > 0) else None
                    )
                    fs_vs_cu = (
                        (cupy_ms / t_ms)
                        if (cupy_ms is not None and t_ms and t_ms > 0)
                        else None
                    )
                    print(
                        f"{fmt:>5} {n:>6} {_fmt_ms(t_ms):>14} {_fmt_ms(pytorch_ms):>12} "
                        f"{_fmt_ms(cupy_ms):>10} "
                        f"{(f'{fs_vs_pt:.2f}x' if fs_vs_pt is not None else 'N/A'):>8} "
                        f"{(f'{fs_vs_cu:.2f}x' if fs_vs_cu is not None else 'N/A'):>8} "
                        f"{status:>8} {_fmt_err(err_pt):>12} {_fmt_err(err_cu):>12}"
                    )
            print("-" * 110)
            print()

    print(sep)
    print(f"Total cases: {total}  Failed: {failed}")
    print(sep)


def _run_one_csv_row_csr(path, value_dtype, index_dtype, device):
    data, indices, indptr, shape = _load_mtx_to_csr_torch(
        path, dtype=value_dtype, device=device
    )
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    x_true = torch.randn(n_rows, dtype=value_dtype, device=device)
    b, _ = fs.flagsparse_spmv_csr(
        data, indices, indptr, x_true, shape, return_time=True
    )
    x, t_ms = fs.flagsparse_spsv_csr(
        data, indices, indptr, b, shape, lower=True, return_time=True
    )
    return _finalize_csv_row(
        path, value_dtype, index_dtype, data, indices, indptr, shape,
        x, t_ms, b, n_rows, n_cols,
    )


def _run_one_csv_row_coo(path, value_dtype, index_dtype, device, coo_mode):
    data, indices, indptr, shape = _load_mtx_to_csr_torch(
        path, dtype=value_dtype, device=device
    )
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    x_true = torch.randn(n_rows, dtype=value_dtype, device=device)
    b, _ = fs.flagsparse_spmv_csr(
        data, indices, indptr, x_true, shape, return_time=True
    )
    d_in, r_in, c_in = _coo_inputs_for_csv(
        data, indices, indptr, shape, coo_mode
    )
    x, t_ms = fs.flagsparse_spsv_coo(
        d_in,
        r_in,
        c_in,
        b,
        shape,
        lower=True,
        coo_mode=coo_mode,
        return_time=True,
    )
    return _finalize_csv_row(
        path,
        value_dtype,
        index_dtype,
        data,
        indices,
        indptr,
        shape,
        x,
        t_ms,
        b,
        n_rows,
        n_cols,
        nnz_display=int(d_in.numel()),
        cupy_coo_data=d_in,
        cupy_coo_row=r_in,
        cupy_coo_col=c_in,
    )


def _finalize_csv_row(
    path,
    value_dtype,
    index_dtype,
    data,
    indices,
    indptr,
    shape,
    x,
    t_ms,
    b,
    n_rows,
    n_cols,
    *,
    nnz_display=None,
    cupy_coo_data=None,
    cupy_coo_row=None,
    cupy_coo_col=None,
):
    atol, rtol = _tol_for_dtype(value_dtype)
    pytorch_ms = None
    err_pt = None
    ok_pt = False
    pt_skip_reason = None
    if _allow_dense_pytorch_ref(shape, value_dtype):
        try:
            A_dense = _csr_to_dense(
                data, indices.to(torch.int64), indptr, shape
            )
            ref_dtype = _dense_ref_dtype(value_dtype)
            A_ref = A_dense.to(ref_dtype)
            b_ref = b.to(ref_dtype)
            e0 = torch.cuda.Event(True)
            e1 = torch.cuda.Event(True)
            torch.cuda.synchronize()
            e0.record()
            x_ref = torch.linalg.solve(A_ref, b_ref.unsqueeze(1)).squeeze(1)
            x_cmp = _compare_view(x, value_dtype)
            x_ref_cmp = _compare_view(x_ref, value_dtype)
            e1.record()
            torch.cuda.synchronize()
            pytorch_ms = e0.elapsed_time(e1)
            err_pt = (
                float(torch.max(torch.abs(x_cmp - x_ref_cmp)).item())
                if n_rows > 0
                else 0.0
            )
            ok_pt = torch.allclose(x_cmp, x_ref_cmp, atol=atol, rtol=rtol)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pt_skip_reason = "PyTorch dense ref OOM; skipped"
            else:
                raise
    else:
        pt_skip_reason = (
            f"PyTorch dense ref skipped (> {DENSE_REF_MAX_BYTES // (1024**3)} GiB dense matrix)"
        )

    cupy_ms = None
    err_cu = None
    ok_cu = False
    x_cu_t = None
    if (
        cp is not None
        and cpx_sparse is not None
        and cpx_spsolve_triangular is not None
    ):
        try:
            data_ref, b_ref = _cupy_ref_inputs(data, b)
            b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_ref.contiguous()))
            if (
                cupy_coo_data is not None
                and cupy_coo_row is not None
                and cupy_coo_col is not None
            ):
                coo_data_ref, _ = _cupy_ref_inputs(cupy_coo_data, b)
                data_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(coo_data_ref.contiguous())
                )
                row_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        cupy_coo_row.to(torch.int64).contiguous()
                    )
                )
                col_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        cupy_coo_col.to(torch.int64).contiguous()
                    )
                )
                A_cp = cpx_sparse.coo_matrix(
                    (data_cp, (row_cp, col_cp)), shape=shape
                )
            else:
                data_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(data_ref.contiguous())
                )
                idx_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        indices.to(torch.int64).contiguous()
                    )
                )
                ptr_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(indptr.contiguous())
                )
                A_cp = cpx_sparse.csr_matrix(
                    (data_cp, idx_cp, ptr_cp), shape=shape
                )
            for _ in range(WARMUP):
                _ = cpx_spsolve_triangular(
                    A_cp, b_cp, lower=True, unit_diagonal=False
                )
            cp.cuda.runtime.deviceSynchronize()
            c0 = cp.cuda.Event()
            c1 = cp.cuda.Event()
            c0.record()
            for _ in range(ITERS):
                x_cu = cpx_spsolve_triangular(
                    A_cp, b_cp, lower=True, unit_diagonal=False
                )
            c1.record()
            c1.synchronize()
            cupy_ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
            x_cu_t = torch.utils.dlpack.from_dlpack(x_cu.toDlpack())
            x_cmp = _compare_view(x, value_dtype)
            x_cu_cmp = _compare_view(x_cu_t, value_dtype)
            err_cu = (
                float(torch.max(torch.abs(x_cmp - x_cu_cmp)).item())
                if n_rows > 0
                else 0.0
            )
            ok_cu = torch.allclose(x_cmp, x_cu_cmp, atol=atol, rtol=rtol)
        except Exception:
            cupy_ms = None
            err_cu = None

    status = "PASS" if (ok_pt or ok_cu) else "FAIL"
    if (not ok_pt) and (not ok_cu) and (err_pt is None and err_cu is None):
        status = "REF_FAIL"

    nnz_out = (
        int(data.numel()) if nnz_display is None else int(nnz_display)
    )
    row = {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz_out,
        "triton_ms": t_ms,
        "pytorch_ms": pytorch_ms,
        "cusparse_ms": cupy_ms,
        "csc_ms": None,
        "status": status,
        "err_pt": err_pt,
        "err_cu": err_cu,
    }
    return row, pt_skip_reason


def _run_one_csv_row_csr_full(path, value_dtype, index_dtype, op_mode, device):
    data, indices, indptr, shape = _load_mtx_to_csr_torch(
        path, dtype=value_dtype, device=device
    )
    indices = indices.to(index_dtype)
    indptr = indptr.to(index_dtype)
    n_rows, n_cols = shape
    x_true = _randn_by_dtype(n_rows, value_dtype, device)
    b = _build_rhs_for_csr_op(data, indices, indptr, x_true, shape, op_mode)
    x, t_ms = fs.flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        shape,
        lower=True,
        transpose=(op_mode == "TRANS"),
        return_time=True,
    )
    return _finalize_csv_row_csr_full(
        path,
        value_dtype,
        index_dtype,
        op_mode,
        data,
        indices,
        indptr,
        shape,
        x,
        t_ms,
        b,
        n_rows,
        n_cols,
    )


def _finalize_csv_row_csr_full(
    path,
    value_dtype,
    index_dtype,
    op_mode,
    data,
    indices,
    indptr,
    shape,
    x,
    t_ms,
    b,
    n_rows,
    n_cols,
):
    atol, rtol = _tol_for_dtype(value_dtype)

    pytorch_ms = None
    err_pt = None
    ok_pt = False
    pt_skip_reason = None
    if _allow_dense_pytorch_ref(shape, value_dtype):
        try:
            A_dense = _csr_to_dense(
                data, indices.to(torch.int64), indptr.to(torch.int64), shape
            ).to(_dense_ref_dtype(value_dtype))
            A_ref = A_dense.transpose(0, 1) if op_mode == "TRANS" else A_dense
            e0 = torch.cuda.Event(True)
            e1 = torch.cuda.Event(True)
            torch.cuda.synchronize()
            e0.record()
            x_ref = torch.linalg.solve(A_ref, b.to(A_ref.dtype).unsqueeze(1)).squeeze(1)
            x_cmp = _compare_view(x, value_dtype)
            x_ref_cmp = _compare_view(x_ref, value_dtype)
            e1.record()
            torch.cuda.synchronize()
            pytorch_ms = e0.elapsed_time(e1)
            err_pt = (
                float(torch.max(torch.abs(x_cmp - x_ref_cmp)).item())
                if n_rows > 0
                else 0.0
            )
            ok_pt = torch.allclose(x_cmp, x_ref_cmp, atol=atol, rtol=rtol)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pt_skip_reason = "PyTorch dense ref OOM; skipped"
            else:
                raise
    else:
        pt_skip_reason = (
            f"PyTorch dense ref skipped (> {DENSE_REF_MAX_BYTES // (1024**3)} GiB dense matrix)"
        )

    cupy_ms = None
    err_cu = None
    ok_cu = False
    x_cu_t = None
    cupy_ms, x_cu_t = _cupy_spsolve_csr_with_op(
        data, indices, indptr, shape, b, op_mode
    )
    if x_cu_t is not None:
        x_cmp = _compare_view(x, value_dtype)
        x_cu_cmp = _compare_view(x_cu_t, value_dtype)
        err_cu = (
            float(torch.max(torch.abs(x_cmp - x_cu_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_cu = torch.allclose(x_cmp, x_cu_cmp, atol=atol, rtol=rtol)

    status = "PASS" if (ok_pt or ok_cu) else "FAIL"
    if (not ok_pt) and (not ok_cu) and (err_pt is None and err_cu is None):
        status = "REF_FAIL"

    row = {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "opA": op_mode,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "triton_ms": t_ms,
        "pytorch_ms": pytorch_ms,
        "cusparse_ms": cupy_ms,
        "csc_ms": None,
        "status": status,
        "err_pt": err_pt,
        "err_cu": err_cu,
    }
    return row, pt_skip_reason


def run_all_supported_spsv_csr_csv(mtx_paths, csv_path):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    for value_dtype in CSR_FULL_VALUE_DTYPES:
        for index_dtype in CSR_FULL_INDEX_DTYPES:
            op_modes = _supported_csr_full_ops(value_dtype, index_dtype)
            for op_mode in op_modes:
                print("=" * 150)
                print(
                    f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  CSR  |  opA={op_mode}"
                )
                print(
                    "Formats: FlagSparse=CSR, cuSPARSE=CSR ref, PyTorch=Dense solve."
                )
                print(
                    "Err(PT)=|FlagSparse-PyTorch|, Err(CU)=|FlagSparse-cuSPARSE|. "
                    "PASS if either error within tolerance."
                )
                print("-" * 150)
                print(
                    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                    f"{'FlagSparse(ms)':>10} {'CSR(ms)':>10} {'CSC(ms)':>10} {'PyTorch(ms)':>11} "
                    f"{'FS/CSR':>7} {'FS/PT':>7} {'Status':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
                )
                print("-" * 150)
                for path in mtx_paths:
                    try:
                        row, pt_skip = _run_one_csv_row_csr_full(
                            path, value_dtype, index_dtype, op_mode, device
                        )
                        rows_out.append(row)
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        n_rows, n_cols = row["n_rows"], row["n_cols"]
                        nnz = row["nnz"]
                        t_ms = row["triton_ms"]
                        cupy_ms = row["cusparse_ms"]
                        pytorch_ms = row["pytorch_ms"]
                        err_pt, err_cu = row["err_pt"], row["err_cu"]
                        status = row["status"]
                        print(
                            f"{name:<28} {n_rows:>7} {n_cols:>7} {nnz:>10} "
                            f"{_fmt_ms(t_ms):>10} {_fmt_ms(cupy_ms):>10} {_fmt_ms(None):>10} {_fmt_ms(pytorch_ms):>11} "
                            f"{_fmt_speedup(cupy_ms, t_ms):>7} {_fmt_speedup(pytorch_ms, t_ms):>7} "
                            f"{status:>6} {_fmt_err(err_pt):>10} {_fmt_err(err_cu):>10}"
                        )
                        if pt_skip:
                            print(f"  NOTE: {pt_skip}")
                    except Exception as e:
                        err_msg = str(e)
                        status = "SKIP" if "SpSV requires square matrices" in err_msg else "ERROR"
                        rows_out.append(
                            {
                                "matrix": os.path.basename(path),
                                "value_dtype": _dtype_name(value_dtype),
                                "index_dtype": _dtype_name(index_dtype),
                                "opA": op_mode,
                                "n_rows": "ERR",
                                "n_cols": "ERR",
                                "nnz": "ERR",
                                "triton_ms": None,
                                "pytorch_ms": None,
                                "cusparse_ms": None,
                                "csc_ms": None,
                                "status": status,
                                "err_pt": None,
                                "err_cu": None,
                            }
                        )
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        print(
                            f"{name:<28} {'ERR':>7} {'ERR':>7} {'ERR':>10} "
                            f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>11} "
                            f"{'N/A':>7} {'N/A':>7} "
                            f"{status:>6} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  {status}: {e}")
                print("-" * 150)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "opA",
        "n_rows",
        "n_cols",
        "nnz",
        "triton_ms",
        "pytorch_ms",
        "cusparse_ms",
        "csc_ms",
        "status",
        "err_pt",
        "err_cu",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def run_all_dtypes_spsv_csv(mtx_paths, csv_path, use_coo=False, coo_mode="auto"):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    label = "COO" if use_coo else "CSR"
    cu_col = "COO(ms)" if use_coo else "CSR(ms)"
    fs_cu_hdr = "FS/COO" if use_coo else "FS/CSR"
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            atol, rtol = _tol_for_dtype(value_dtype)
            print("=" * 150)
            print(
                f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  {label}"
                + (f"  coo_mode={coo_mode}" if use_coo else "")
            )
            if use_coo:
                print(
                    "Formats: FlagSparse=COO SpSV, cuSPARSE=COO ref, PyTorch=Dense solve. "
                    "b 由 CSR SpMV 构造，与 CSR 测试一致。"
                )
            else:
                print(
                    "Formats: FlagSparse=CSR, cuSPARSE=CSR ref, PyTorch=Dense solve."
                )
            print(
                "Err(PT)=|FlagSparse-PyTorch|, Err(CU)=|FlagSparse-cuSPARSE|. "
                "PASS if either error within tolerance."
            )
            print("-" * 150)
            print(
                f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                f"{'FlagSparse(ms)':>10} {cu_col:>10} {'CSC(ms)':>10} {'PyTorch(ms)':>11} "
                f"{fs_cu_hdr:>7} {'FS/PT':>7} {'Status':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
            )
            print("-" * 150)
            for path in mtx_paths:
                try:
                    if use_coo:
                        row, pt_skip = _run_one_csv_row_coo(
                            path, value_dtype, index_dtype, device, coo_mode
                        )
                    else:
                        row, pt_skip = _run_one_csv_row_csr(
                            path, value_dtype, index_dtype, device
                        )
                    rows_out.append(row)
                    name = os.path.basename(path)[:27]
                    if len(os.path.basename(path)) > 27:
                        name = name + "…"
                    n_rows, n_cols = row["n_rows"], row["n_cols"]
                    nnz = row["nnz"]
                    t_ms = row["triton_ms"]
                    cupy_ms = row["cusparse_ms"]
                    pytorch_ms = row["pytorch_ms"]
                    err_pt, err_cu = row["err_pt"], row["err_cu"]
                    status = row["status"]
                    print(
                        f"{name:<28} {n_rows:>7} {n_cols:>7} {nnz:>10} "
                        f"{_fmt_ms(t_ms):>10} {_fmt_ms(cupy_ms):>10} {_fmt_ms(None):>10} {_fmt_ms(pytorch_ms):>11} "
                        f"{_fmt_speedup(cupy_ms, t_ms):>7} {_fmt_speedup(pytorch_ms, t_ms):>7} "
                        f"{status:>6} {_fmt_err(err_pt):>10} {_fmt_err(err_cu):>10}"
                    )
                    if pt_skip:
                        print(f"  NOTE: {pt_skip}")
                except Exception as e:
                    err_msg = str(e)
                    status = "SKIP" if "SpSV requires square matrices" in err_msg else "ERROR"
                    rows_out.append(
                        {
                            "matrix": os.path.basename(path),
                            "value_dtype": _dtype_name(value_dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "n_rows": "ERR",
                            "n_cols": "ERR",
                            "nnz": "ERR",
                            "triton_ms": None,
                            "pytorch_ms": None,
                            "cusparse_ms": None,
                            "csc_ms": None,
                            "status": status,
                            "err_pt": None,
                            "err_cu": None,
                        }
                    )
                    name = os.path.basename(path)[:27]
                    if len(os.path.basename(path)) > 27:
                        name = name + "…"
                    print(
                        f"{name:<28} {'ERR':>7} {'ERR':>7} {'ERR':>10} "
                        f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>11} "
                        f"{'N/A':>7} {'N/A':>7} {status:>6} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                    )
                    print(f"  {status}: {e}")
            print("-" * 150)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "n_rows",
        "n_cols",
        "nnz",
        "triton_ms",
        "pytorch_ms",
        "cusparse_ms",
        "csc_ms",
        "status",
        "err_pt",
        "err_cu",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpSV test: synthetic triangular systems and optional .mtx (CSR/COO), same baselines as CSR."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic triangular tests"
    )
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run full supported CSR SpSV combinations (dtype/index/opA) on .mtx and export CSV",
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on .mtx (COO SpSV), same CSV columns as --csv-csr",
    )
    parser.add_argument(
        "--coo-mode",
        type=str,
        default="auto",
        choices=["auto", "direct", "csr"],
        help="COO mode for --csv-coo (default: auto)",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_spsv_synthetic_all()
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.csv_csr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-csr")
            return
        run_all_supported_spsv_csr_csv(paths, args.csv_csr)
        return
    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_spsv_csv(
            paths, args.csv_coo, use_coo=True, coo_mode=args.coo_mode
        )
        return

    print("Use --synthetic, --csv-csr, or --csv-coo to run SpSV tests.")


if __name__ == "__main__":
    main()
