"""SpSM tests: synthetic triangular systems and optional .mtx batch CSV."""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
import flagsparse.sparse_operations.spsm as fs_spsm_impl

try:
    import cupy as cp
    import cupyx.cusparse as cpx_cusparse
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_cusparse = None
    cpx_sparse = None


FORMATS = ("csr", "coo")
VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 20
TRACE_CUSPARSE = False
SPSM_OP_MODES = ["NON", "NON_TRANS"]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _tol(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-3
    return 1e-12, 1e-10


def _reference_check_threshold(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-6
    return 1e-12


def _reference_max_relative_error(answer, result, dtype):
    if answer is None or result is None:
        return None
    if answer.numel() != result.numel():
        return float("inf")
    if answer.numel() == 0:
        return 0.0
    if dtype in (torch.complex64, torch.complex128):
        answer_cmp = torch.abs(answer)
        result_cmp = torch.abs(result)
        diff = torch.abs(answer_cmp - result_cmp)
    else:
        diff = torch.abs(answer - result)
        result_cmp = torch.abs(result)
    if not bool(torch.isfinite(diff).all().item()) or not bool(torch.isfinite(result_cmp).all().item()):
        return float("inf")
    max_error = torch.max(diff)
    max_result = torch.max(result_cmp)
    if float(max_result.item()) == 0.0:
        return 0.0 if float(max_error.item()) == 0.0 else float("inf")
    return float((max_error / max_result).item())


def _is_fatal_cuda_error(exc):
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "device-side assert" in msg
        or "unspecified launch failure" in msg
    )


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_ratio(v):
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _safe_ratio(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return other_ms / triton_ms

def _cusparse_measurement_scope():
    return "cupyx_interface_call"


def _amortized_total_ms(analysis_ms, solve_ms, iters):
    if (
        analysis_ms is None
        or solve_ms is None
        or iters is None
        or int(iters) <= 0
    ):
        return None
    return (float(analysis_ms) / float(iters)) + float(solve_ms)


def _speedup_total_ratio(other_ms, analysis_ms, solve_ms, iters):
    return _safe_ratio(other_ms, _amortized_total_ms(analysis_ms, solve_ms, iters))


def _fmt_trace_times(times):
    if not times:
        return "[]"
    return "[" + ", ".join(f"{float(t):.6f}" for t in times) + "]"


def _emit_cusparse_trace(tag, warmup_times, timed_times, reduced_ms):
    if not TRACE_CUSPARSE:
        return
    print(
        f"[trace][{tag}] scope={_cusparse_measurement_scope()} "
        "(matrix construction and Python-side setup stay outside the timed region)"
    )
    if warmup_times:
        print(f"[trace][{tag}] warmup_ms={_fmt_trace_times(warmup_times)}")
    print(f"[trace][{tag}] timed_ms={_fmt_trace_times(timed_times)}")
    if timed_times:
        print(
            f"[trace][{tag}] first_timed_ms={float(timed_times[0]):.6f} "
            f"reduced_ms={float(reduced_ms):.6f}"
        )


def _spsm_benchmark_schedule(nnz, n_rhs, value_dtype, fmt="csr"):
    del nnz, n_rhs, value_dtype, fmt
    return int(WARMUP), int(ITERS)


def _allinone_filtered_avg_ms(times):
    if not times:
        return None
    times = [float(t) for t in times]
    if len(times) == 1:
        return times[0]
    ordered = sorted(times)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    lo = median * 0.9
    hi = median * 1.1
    kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def _parse_csv_tokens(raw):
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_ops_filter(raw):
    tokens = [tok.strip().upper() for tok in _parse_csv_tokens(raw)]
    if not tokens:
        return ["NON"]
    invalid = [tok for tok in tokens if tok not in SPSM_OP_MODES]
    if invalid:
        raise ValueError(
            f"unsupported spsm ops: {invalid}; current SpSM test only supports NON/NON_TRANS"
        )
    normalized = []
    for tok in tokens:
        normalized.append("NON" if tok == "NON_TRANS" else tok)
    return normalized


def _build_triangular_case(n=512, n_rhs=1024, value_dtype=torch.float32):
    device = torch.device("cuda")
    A = torch.tril(torch.randn((n, n), dtype=value_dtype, device=device) * 0.02)
    diag_base_dtype = torch.float32 if value_dtype == torch.complex64 else torch.float64
    diag = (torch.rand((n,), dtype=diag_base_dtype, device=device) + 2.0).to(value_dtype)
    A = A + torch.diag(diag)
    coo = A.to_sparse().coalesce()
    row = coo.indices()[0].to(torch.int64)
    col = coo.indices()[1].to(torch.int64)
    data = coo.values().to(value_dtype)
    _, order = torch.sort(row * n + col)
    row = row[order]
    col = col[order]
    data = data[order]
    nnz_per_row = torch.bincount(row, minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    B = torch.randn((n, n_rhs), dtype=value_dtype, device=device).contiguous()
    return data, row, col, indptr, B, (n, n)


def _csr_to_coo(indices, indptr, n_rows):
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=indptr.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    return row, indices.to(torch.int64)


def _extract_effective_lower_csr(data, indices, indptr, shape):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    tri_mask = col <= row
    data_tri = data[tri_mask]
    row_tri = row[tri_mask]
    col_tri = col[tri_mask]
    if row_tri.numel() == 0:
        empty_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr.device)
        empty_indices = torch.empty((0,), dtype=torch.int64, device=indices.device)
        empty_data = data.new_empty((0,))
        return empty_data, empty_indices, empty_indptr
    order = torch.argsort(row_tri * n_rows + col_tri)
    row_tri = row_tri[order]
    col_tri = col_tri[order]
    data_tri = data_tri[order]
    counts = torch.bincount(row_tri, minlength=n_rows)
    indptr_tri = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr.device)
    indptr_tri[1:] = torch.cumsum(counts, dim=0)
    return data_tri, col_tri.to(torch.int64), indptr_tri


def _benchmark_pytorch_reference(data, indices, indptr, shape, B):
    try:
        sparse_spsolve = getattr(torch.sparse, "spsolve", None)
        if sparse_spsolve is None:
            raise NotImplementedError("torch.sparse.spsolve is unavailable")
        data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
            data, indices, indptr, shape
        )
        A_csr = torch.sparse_csr_tensor(
            indptr_eff,
            indices_eff,
            data_eff,
            size=shape,
            device=data.device,
        )
        if not A_csr.is_cuda:
            raise RuntimeError("torch.sparse.spsolve CUDA path is unavailable")
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        e0.record()
        cols = []
        for bj in torch.unbind(B, dim=1):
            cols.append(sparse_spsolve(A_csr, bj))
        X_ref = torch.stack(cols, dim=1) if cols else B.new_empty(B.shape)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        return X_ref.to(B.dtype), ms, "gpu_sparse", None
    except Exception as exc:
        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, "unavailable", f"PyTorch sparse solve unavailable ({exc})"


def _benchmark_cusparse_reference(data, row, col, indptr, B, shape, fmt, warmup, iters):
    if cp is None or cpx_sparse is None or cpx_cusparse is None:
        return None, None, "cusparse unavailable"
    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B.contiguous()))
        if fmt == "coo":
            row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.contiguous()))
            col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.contiguous()))
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.contiguous()))
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.contiguous()))
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        A_cp.sum_duplicates()
        warmup_times = []
        for _ in range(warmup):
            if TRACE_CUSPARSE:
                c0 = cp.cuda.Event()
                c1 = cp.cuda.Event()
                c0.record()
                _ = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
                c1.record()
                c1.synchronize()
                warmup_times.append(cp.cuda.get_elapsed_time(c0, c1))
            else:
                _ = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
        cp.cuda.runtime.deviceSynchronize()
        times = []
        for _ in range(iters):
            c0 = cp.cuda.Event()
            c1 = cp.cuda.Event()
            c0.record()
            X_cp = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
            c1.record()
            c1.synchronize()
            times.append(cp.cuda.get_elapsed_time(c0, c1))
        ms = _allinone_filtered_avg_ms(times)
        _emit_cusparse_trace(
            f"SpSM fmt={fmt.upper()} shape={shape} rhs={B.shape[1]}",
            warmup_times,
            times,
            ms,
        )
        X_t = torch.utils.dlpack.from_dlpack(X_cp.toDlpack()).to(B.dtype)
        return X_t, ms, None
    except Exception as exc:
        return None, None, str(exc)


def _apply_csr_to_dense_rhs(data, indices, indptr, X, shape):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    out = torch.zeros((n_rows, X.shape[1]), dtype=X.dtype, device=X.device)
    out.index_add_(0, row, data[:, None] * X[col])
    return out


def _solution_residual_metrics(data, indices, indptr, shape, X, B, value_dtype):
    atol, rtol = _tol(value_dtype)
    data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
        data, indices, indptr, shape
    )
    B_recon = _apply_csr_to_dense_rhs(data_eff, indices_eff, indptr_eff, X, shape)
    err = float(torch.max(torch.abs(B_recon - B)).item()) if B.numel() else 0.0
    ok = torch.allclose(B_recon, B, atol=atol, rtol=rtol)
    return err, ok


def _benchmark_flagsparse(call, warmup, iters):
    X = None
    for _ in range(warmup):
        X = call()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        e0.record()
        X = call()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    return X, _allinone_filtered_avg_ms(times)


def _benchmark_flagsparse_spsm_csr_split(data, indices, indptr, B, shape):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="csr"
    )
    analysis_ms = fs_spsm_impl._analyze_spsm_csr(
        data,
        indices,
        indptr,
        B,
        shape,
        lower=True,
        unit_diagonal=False,
        clear_cache=True,
        return_time=True,
    )
    X, solve_ms = _benchmark_flagsparse(
        lambda: fs.flagsparse_spsm_csr(
            data,
            indices,
            indptr,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        ),
        warmup,
        iters,
    )
    return X, analysis_ms, solve_ms


def _benchmark_flagsparse_spsm_coo_split(data, row, col, B, shape):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="coo"
    )
    analysis_ms = fs_spsm_impl._analyze_spsm_coo(
        data,
        row,
        col,
        B,
        shape,
        lower=True,
        unit_diagonal=False,
        clear_cache=True,
        return_time=True,
    )
    X, solve_ms = _benchmark_flagsparse(
        lambda: fs.flagsparse_spsm_coo(
            data,
            row,
            col,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        ),
        warmup,
        iters,
    )
    return X, analysis_ms, solve_ms


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
        raise ValueError("SpSM requires square matrices")

    row_maps = [dict() for _ in range(n_rows)]

    def _accum(r, c, v):
        row = row_maps[r]
        if c in row:
            row[c] += v
        else:
            row[c] = v

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if mm_field == "complex":
            if len(parts) < 4:
                raise ValueError("MatrixMarket complex entry requires real and imag parts")
            v = complex(float(parts[2]), float(parts[3]))
        elif len(parts) >= 3:
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

    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))

    data = torch.tensor(vals_s, dtype=dtype, device=device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _run_one_spsm_case(data, indices, indptr, shape, value_dtype, index_dtype, n_rhs, fmt):
    n_rows = int(shape[0])
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=data.device).contiguous()
    atol, rtol = _tol(value_dtype)
    data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
        data, indices, indptr, shape
    )
    row, col = _csr_to_coo(indices_eff, indptr_eff, n_rows)
    warmup, iters = _spsm_benchmark_schedule(
        data_eff.numel(), n_rhs, value_dtype, fmt=fmt
    )

    if fmt == "csr":
        X_fs, analysis_ms, solve_ms = _benchmark_flagsparse_spsm_csr_split(
            data_eff,
            indices_eff.to(index_dtype),
            indptr_eff.to(index_dtype),
            B,
            shape,
        )
    else:
        X_fs, analysis_ms, solve_ms = _benchmark_flagsparse_spsm_coo_split(
            data_eff,
            row.to(index_dtype),
            col.to(index_dtype),
            B,
            shape,
        )
    X_cu, cusparse_ms, _cusparse_reason = _benchmark_cusparse_reference(
        data_eff, row, col, indptr_eff, B, shape, fmt, warmup, iters
    )
    X_pt, pytorch_ms, _pt_backend, pytorch_reason = _benchmark_pytorch_reference(
        data_eff, indices_eff, indptr_eff, shape, B
    )

    err_cu = None
    ok_cu = None
    rel_cu = None
    if X_cu is not None:
        err_cu = float(torch.max(torch.abs(X_fs - X_cu)).item()) if X_fs.numel() else 0.0
        rel_cu = _reference_max_relative_error(X_cu, X_fs, value_dtype)
        ok_cu = rel_cu <= _reference_check_threshold(value_dtype)

    err_pt = None
    ok_pt = None
    rel_pt = None
    if X_pt is not None:
        err_pt = float(torch.max(torch.abs(X_fs - X_pt)).item()) if X_fs.numel() else 0.0
        rel_pt = _reference_max_relative_error(X_pt, X_fs, value_dtype)
        ok_pt = rel_pt <= _reference_check_threshold(value_dtype)

    err_res, ok_res = _solution_residual_metrics(
        data_eff, indices_eff, indptr_eff, shape, X_fs, B, value_dtype
    )
    ref_errors = [v for v in (err_pt, err_cu) if v is not None]
    err_ref = min(ref_errors) if ref_errors else None

    if ok_cu is not None:
        status = "PASS" if ok_cu else "FAIL"
    elif ok_pt is not None:
        status = "PASS" if ok_pt else "FAIL"
    elif X_pt is None and X_cu is None:
        status = "REF_FAIL"
    else:
        status = "FAIL"

    return {
        "format": fmt,
        "n_rows": n_rows,
        "n_cols": int(shape[1]),
        "nnz": int(data_eff.numel()),
        "n_rhs": int(n_rhs),
        "FlagSparse_analysis_ms": analysis_ms,
        "FlagSparse_solve_ms": solve_ms,
        "FlagSparse_ms": _amortized_total_ms(analysis_ms, solve_ms, iters),
        "cuSPARSE_ms": cusparse_ms,
        "PyTorch_ms": pytorch_ms,
        "FlagSparse_vs_cuSPARSE_speedup": _speedup_total_ratio(
            cusparse_ms, analysis_ms, solve_ms, iters
        ),
        "FlagSparse_vs_PyTorch_speedup": _speedup_total_ratio(
            pytorch_ms, analysis_ms, solve_ms, iters
        ),
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "pytorch_reason": pytorch_reason,
        "error": None,
    }


def run_spsm_synthetic_all(n=512, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    total = 0
    failed = 0
    print("=" * 144)
    print("FLAGSPARSE SpSM synthetic test")
    print("=" * 144)
    print(
        "Baselines: cuSPARSE full-RHS interface call, PyTorch per-column sparse solve."
    )
    print(
        f"{'Fmt':>5} {'dtype':>9} {'index':>7} {'N':>6} {'RHS':>6} {'NNZ':>10} "
        f"{'FS.an':>11} {'FS.sol':>10} {'FS.ms':>10} "
        f"{'CU.ms':>10} {'PT.ms':>10} {'FS/CU':>10} {'FS/PT':>10} "
        f"{'Status':>10} {'Err(Ref)':>12} {'Err(Res)':>12} {'Err(PT)':>12} {'Err(CU)':>12}"
    )
    print("-" * 144)

    for fmt in FORMATS:
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                data, row, col, indptr, _B, shape = _build_triangular_case(
                    n=n,
                    n_rhs=n_rhs,
                    value_dtype=value_dtype,
                )
                one = _run_one_spsm_case(
                    data,
                    col,
                    indptr,
                    shape,
                    value_dtype,
                    index_dtype,
                    n_rhs,
                    fmt,
                )
                total += 1
                if one["status"] != "PASS":
                    failed += 1
                print(
                    f"{fmt:>5} {_dtype_name(value_dtype):>9} {_dtype_name(index_dtype):>7} "
                    f"{shape[0]:>6} {n_rhs:>6} {one['nnz']:>10} "
                    f"{_fmt_ms(one['FlagSparse_analysis_ms']):>11} {_fmt_ms(one['FlagSparse_solve_ms']):>10} {_fmt_ms(one['FlagSparse_ms']):>10} "
                    f"{_fmt_ms(one['cuSPARSE_ms']):>10} {_fmt_ms(one['PyTorch_ms']):>10} "
                    f"{_fmt_ratio(one['FlagSparse_vs_cuSPARSE_speedup']):>10} {_fmt_ratio(one['FlagSparse_vs_PyTorch_speedup']):>10} "
                    f"{one['status']:>10} {_fmt_err(one['err_ref']):>12} {_fmt_err(one['err_res']):>12} "
                    f"{_fmt_err(one['err_pt']):>12} {_fmt_err(one['err_cu']):>12}"
                )
                if one["status"] in ("FAIL", "REF_FAIL"):
                    if one["pytorch_reason"]:
                        print(f"  NOTE: {one['pytorch_reason']}")
    print("-" * 144)
    print(f"Total cases: {total}  Failed: {failed}")
    print("=" * 144)


def run_all_dtypes_spsm_csv(mtx_paths, csv_path, use_coo=False, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    records_out = []
    fmt = "coo" if use_coo else "csr"

    print("=" * 152)
    print(
        f"FLAGSPARSE SpSM .mtx batch ({fmt.upper()}) | "
        "CU: full-RHS interface call | PT: per-column sparse solve"
    )
    print("=" * 152)
    print(
        f"Benchmark schedule: warmup={WARMUP}, timed_iters={ITERS} "
        "(filtered per-iteration solve averages; override with --warmup/--iters)"
    )
    print(
        "FS.ms = FS.sol + FS.an / timed_iters. "
        "CU.ms/PT.ms are baseline call times. "
        "FS/CU and FS/PT are baseline_ms / FS.ms."
    )
    print(
        f"{'Matrix':<28} {'dtype':>9} {'index':>7} {'N':>7} {'RHS':>6} {'NNZ':>10} "
        f"{'FS.an':>11} {'FS.sol':>10} {'FS.ms':>10} "
        f"{'CU.ms':>10} {'PT.ms':>10} {'FS/CU':>10} {'FS/PT':>10} "
        f"{'Status':>10} {'Eref':>12} {'Eres':>12} {'Ept':>12} {'Ecu':>12}"
    )
    print("-" * 152)

    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            for path in mtx_paths:
                base = {
                    "matrix": os.path.basename(path),
                    "value_dtype": _dtype_name(value_dtype),
                    "index_dtype": _dtype_name(index_dtype),
                }
                try:
                    print(
                        f"RUNNING: {base['matrix']} | dtype={base['value_dtype']} | "
                        f"index={base['index_dtype']} | fmt={fmt}",
                        flush=True,
                    )
                    data, indices, indptr, shape = _load_mtx_to_csr_torch(
                        path,
                        dtype=value_dtype,
                        device=device,
                    )
                    record = _run_one_spsm_case(
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        index_dtype,
                        n_rhs,
                        fmt,
                    )
                    record = {**base, **record}
                    records_out.append(record)
                    short = base["matrix"][:27] + ("…" if len(base["matrix"]) > 27 else "")
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{record['n_rows']:>7} {record['n_rhs']:>6} {record['nnz']:>10} "
                        f"{_fmt_ms(record['FlagSparse_analysis_ms']):>11} {_fmt_ms(record['FlagSparse_solve_ms']):>10} {_fmt_ms(record['FlagSparse_ms']):>10} "
                        f"{_fmt_ms(record['cuSPARSE_ms']):>10} {_fmt_ms(record['PyTorch_ms']):>10} "
                        f"{_fmt_ratio(record['FlagSparse_vs_cuSPARSE_speedup']):>10} {_fmt_ratio(record['FlagSparse_vs_PyTorch_speedup']):>10} "
                        f"{record['status']:>10} {_fmt_err(record['err_ref']):>12} {_fmt_err(record['err_res']):>12} "
                        f"{_fmt_err(record['err_pt']):>12} {_fmt_err(record['err_cu']):>12}"
                    )
                    if record["status"] in ("FAIL", "REF_FAIL"):
                        if record["pytorch_reason"]:
                            print(f"  NOTE: {record['pytorch_reason']}")
                except Exception as exc:
                    err_msg = str(exc)
                    if _is_fatal_cuda_error(exc):
                        print(
                            f"  FATAL CUDA ERROR: {exc}\n"
                            "  CUDA context is no longer reliable. Restart the Python/Singularity session "
                            "and rerun the single failing matrix with CUDA_LAUNCH_BLOCKING=1."
                        )
                        raise
                    status = "SKIP" if "SpSM requires square matrices" in err_msg else "ERROR"
                    record = {
                        **base,
                        "format": fmt,
                        "n_rows": "ERR",
                        "n_cols": "ERR",
                        "nnz": "ERR",
                        "n_rhs": int(n_rhs),
                        "FlagSparse_analysis_ms": None,
                        "FlagSparse_solve_ms": None,
                        "FlagSparse_ms": None,
                        "cuSPARSE_ms": None,
                        "PyTorch_ms": None,
                        "FlagSparse_vs_cuSPARSE_speedup": None,
                        "FlagSparse_vs_PyTorch_speedup": None,
                        "status": status,
                        "err_ref": None,
                        "err_res": None,
                        "err_pt": None,
                        "err_cu": None,
                        "pytorch_reason": None,
                        "error": err_msg,
                    }
                    records_out.append(record)
                    short = base["matrix"][:27] + ("…" if len(base["matrix"]) > 27 else "")
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{'ERR':>7} {int(n_rhs):>6} {'ERR':>10} "
                        f"{_fmt_ms(None):>11} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                        f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {status:>10} "
                        f"{_fmt_err(None):>12} {_fmt_err(None):>12} {_fmt_err(None):>12} {_fmt_err(None):>12}"
                    )
                    print(f"  {status}: {exc}")

    print("-" * 152)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "format",
        "n_rows",
        "n_cols",
        "nnz",
        "n_rhs",
        "FlagSparse_analysis_ms",
        "FlagSparse_solve_ms",
        "FlagSparse_ms",
        "cuSPARSE_ms",
        "PyTorch_ms",
        "FlagSparse_vs_cuSPARSE_speedup",
        "FlagSparse_vs_PyTorch_speedup",
        "status",
        "err_ref",
        "err_res",
        "err_pt",
        "err_cu",
        "pytorch_reason",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for record in records_out:
            w.writerow({k: ("" if v is None else v) for k, v in record.items()})
    print(f"Wrote {len(records_out)} rows to {csv_path}")


def main():
    global WARMUP, ITERS, TRACE_CUSPARSE
    parser = argparse.ArgumentParser(
        description="SpSM test: synthetic triangular systems and optional .mtx batch CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic triangular tests")
    parser.add_argument("--n", type=int, default=512, help="matrix size (synthetic)")
    parser.add_argument(
        "--rhs",
        type=int,
        default=1024,
        help="number of RHS columns (default: 1024, matching all-in-one SpSM)",
    )
    parser.add_argument("--csv-csr", type=str, default=None, metavar="FILE")
    parser.add_argument("--csv-coo", type=str, default=None, metavar="FILE")
    parser.add_argument(
        "--ops",
        type=str,
        default="NON",
        help="comma-separated op(A) modes; currently only NON/NON_TRANS is supported",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="Benchmark warmup solve iterations (default: 10, matching all-in-one cuSPARSE SpSM timing)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=ITERS,
        help="Benchmark timed solve iterations; solve times report the average (default: 20, matching all-in-one cuSPARSE SpSM timing)",
    )
    parser.add_argument(
        "--print-cusparse-times",
        action="store_true",
        help="Print warmup/timed per-round cuSPARSE interface-call times to clarify what cupyx is measuring.",
    )
    args = parser.parse_args()
    WARMUP = max(0, int(args.warmup))
    ITERS = max(1, int(args.iters))
    TRACE_CUSPARSE = bool(args.print_cusparse_times)

    ops = _parse_ops_filter(args.ops)
    if any(op != "NON" for op in ops):
        raise ValueError("SpSM test currently supports only --ops NON/NON_TRANS")

    if args.synthetic:
        run_spsm_synthetic_all(n=args.n, n_rhs=args.rhs)
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
        run_all_dtypes_spsm_csv(paths, args.csv_csr, use_coo=False, n_rhs=args.rhs)
        return

    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_spsm_csv(paths, args.csv_coo, use_coo=True, n_rhs=args.rhs)
        return

    print("Use --synthetic, --csv-csr, or --csv-coo to run SpSM tests.")


if __name__ == "__main__":
    main()
