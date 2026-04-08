import argparse
import csv
import math
import os
import time

import torch

import flagsparse as ast
from flagsparse.sparse_operations._common import cp


DEFAULT_CASES = [
    (32_768, 1_024),
    (131_072, 4_096),
    (524_288, 16_384),
    (1_048_576, 65_536),
]
DEFAULT_VALUE_DTYPES = "float16,bfloat16,float32,float64,complex32,complex64,complex128"
DEFAULT_INDEX_DTYPES = "int32,int64"
WARMUP = 20
ITERS = 200


def _fmt_ms(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _fmt_speedup(value):
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


def _fmt_err(value):
    if value is None:
        return "N/A"
    return f"{value:.2e}"


def _parse_value_dtypes(raw):
    allowed = {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex32",
        "complex64",
        "complex128",
    }
    tokens = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        raise ValueError("value dtypes list is empty")
    invalid = [tok for tok in tokens if tok not in allowed]
    if invalid:
        raise ValueError(f"unsupported value dtypes: {invalid}")
    return tokens


def _parse_index_dtypes(raw):
    mapping = {"int32": torch.int32, "int64": torch.int64}
    tokens = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        raise ValueError("index dtypes list is empty")
    invalid = [tok for tok in tokens if tok not in mapping]
    if invalid:
        raise ValueError(f"unsupported index dtypes: {invalid}")
    return [(tok, mapping[tok]) for tok in tokens]


def _parse_cases(raw):
    if raw is None or not str(raw).strip():
        return list(DEFAULT_CASES)
    pairs = []
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid case '{item}', expected dense:nnz")
        left, right = item.split(":", 1)
        dense_size = int(left)
        nnz = int(right)
        if dense_size < 0 or nnz < 0:
            raise ValueError(f"case values must be non-negative: {item}")
        pairs.append((dense_size, nnz))
    if not pairs:
        raise ValueError("case list is empty")
    return pairs


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _sync_all():
    torch.cuda.synchronize()
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()


def _bench_cuda_op(op, warmup, iters):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    output = None
    for _ in range(warmup):
        output = op()
    _sync_all()
    start = time.perf_counter()
    for _ in range(iters):
        output = op()
    _sync_all()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return output, elapsed_ms


def _to_real_imag(value):
    if hasattr(value, "is_complex") and value.is_complex():
        return float(value.real.item()), float(value.imag.item())
    if getattr(value, "ndim", 0) == 1 and int(value.numel()) == 2:
        return float(value[0].item()), float(value[1].item())
    scalar = float(value.item())
    return scalar, 0.0


def _collect_samples(case_id, expected, flagsparse_out, limit):
    rows = []
    if expected is None or flagsparse_out is None:
        return rows
    max_items = min(int(limit), int(expected.shape[0]), int(flagsparse_out.shape[0]))
    for pos in range(max_items):
        exp_val = expected[pos]
        fs_val = flagsparse_out[pos]
        exp_real, exp_imag = _to_real_imag(exp_val)
        fs_real, fs_imag = _to_real_imag(fs_val)
        abs_error = float(torch.max(torch.abs(fs_val - exp_val)).item())
        rows.append(
            {
                "case_id": case_id,
                "pos": pos,
                "expected_real": exp_real,
                "expected_imag": exp_imag,
                "flagsparse_real": fs_real,
                "flagsparse_imag": fs_imag,
                "abs_error": abs_error,
            }
        )
    return rows


def _dtype_mode(value_dtype_req):
    if value_dtype_req in ("float16", "bfloat16", "complex32", "complex64"):
        return "gather_cupy"
    return "gather_triton"


def _select_mode(value_dtype_req, index_dtype):
    # Keep original gather path for half+int32 while retaining cupy path for new combos.
    if value_dtype_req == "float16" and index_dtype == torch.int32:
        return "gather_triton"
    return _dtype_mode(value_dtype_req)


def _build_dense(value_dtype_req, dense_size, device):
    if value_dtype_req == "float16":
        return torch.randn(dense_size, dtype=torch.float16, device=device)
    if value_dtype_req == "bfloat16":
        return torch.randn(dense_size, dtype=torch.bfloat16, device=device)
    if value_dtype_req == "float32":
        return torch.randn(dense_size, dtype=torch.float32, device=device)
    if value_dtype_req == "float64":
        return torch.randn(dense_size, dtype=torch.float64, device=device)
    if value_dtype_req == "complex64":
        real = torch.randn(dense_size, dtype=torch.float32, device=device)
        imag = torch.randn(dense_size, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if value_dtype_req == "complex128":
        real = torch.randn(dense_size, dtype=torch.float64, device=device)
        imag = torch.randn(dense_size, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    if value_dtype_req == "complex32":
        return torch.randn(dense_size, 2, dtype=torch.float16, device=device)
    raise ValueError(f"Unsupported value dtype request: {value_dtype_req}")


def _effective_dtype_name(value_dtype_req):
    mapping = {
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
        "float64": "float64",
        "complex64": "complex64",
        "complex128": "complex128",
        "complex32": "complex16_pair_f16",
    }
    return mapping[value_dtype_req]


def _tolerance(value_dtype_req):
    if value_dtype_req in ("float16", "complex32"):
        return 5e-3, 5e-3
    if value_dtype_req in ("bfloat16",):
        return 1e-2, 1e-2
    if value_dtype_req in ("float32", "complex64"):
        return 1e-6, 1e-5
    if value_dtype_req in ("float64", "complex128"):
        return 1e-10, 1e-8
    return 1e-6, 1e-5


def _check_dtype_supported(value_dtype_req):
    if value_dtype_req in ("bfloat16",) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 not supported on this GPU")


def _is_supported_extra_gather_combo(value_dtype_req, index_dtype):
    # Required extra gather combos only:
    # Half+Int32/Int64, Bfloat16+Int32/Int64, Complex32+Int32/Int64, Complex64+Int32/Int64
    if value_dtype_req == "float16":
        return index_dtype in (torch.int32, torch.int64)
    if value_dtype_req in ("bfloat16", "complex32", "complex64"):
        return index_dtype in (torch.int32, torch.int64)
    # Original gather path dtypes keep original behavior.
    return True


def _build_indices(dense_size, nnz, index_dtype, device):
    return torch.randint(0, dense_size, (nnz,), dtype=index_dtype, device=device)


def _benchmark_gather_case(
    value_dtype_req,
    index_dtype,
    dense_size,
    nnz,
    warmup,
    iters,
    run_cusparse,
    index_fallback_policy,
):
    device = torch.device("cuda")
    _check_dtype_supported(value_dtype_req)
    dense_vector = _build_dense(value_dtype_req, dense_size, device)
    indices = _build_indices(dense_size, nnz, index_dtype, device)
    expected = dense_vector.index_select(0, indices.to(torch.int64))

    mode = _select_mode(value_dtype_req, index_dtype)
    if mode == "gather_cupy":
        preview_output, gather_meta = ast.flagsparse_gather_cupy(
            dense_vector,
            indices,
            index_fallback_policy=index_fallback_policy,
            return_metadata=True,
        )
        _ = preview_output
        flagsparse_op = lambda: ast.flagsparse_gather_cupy(
            dense_vector,
            indices,
            index_fallback_policy=index_fallback_policy,
        )
        cusparse_op = lambda: ast.cusparse_spmv_gather_cupy(dense_vector, indices)[0]
    else:
        gather_meta = {
            "index_fallback_applied": False,
            "index_fallback_reason": None,
            "kernel_index_dtype": str(index_dtype).replace("torch.", ""),
        }
        flagsparse_op = lambda: ast.flagsparse_gather(dense_vector, indices)
        cusparse_op = lambda: ast.cusparse_spmv_gather(dense_vector, indices)[0]

    pytorch_op = lambda: dense_vector.index_select(0, indices.to(torch.int64))
    pytorch_values, pytorch_ms = _bench_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    flagsparse_values, flagsparse_ms = _bench_cuda_op(flagsparse_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance(value_dtype_req)
    fs_match = torch.allclose(flagsparse_values, expected, atol=atol, rtol=rtol)
    fs_max_error = (
        float(torch.max(torch.abs(flagsparse_values - expected)).item())
        if expected.numel() > 0
        else 0.0
    )

    cusparse_values = None
    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        try:
            cusparse_values, cusparse_ms = _bench_cuda_op(cusparse_op, warmup=warmup, iters=iters)
            cusparse_match = torch.allclose(cusparse_values, expected, atol=atol, rtol=rtol)
            cusparse_max_error = (
                float(torch.max(torch.abs(cusparse_values - expected)).item())
                if expected.numel() > 0
                else 0.0
            )
        except Exception as exc:
            cusparse_reason = str(exc)

    fs_vs_pt = pytorch_ms / flagsparse_ms if flagsparse_ms > 0 else float("inf")
    fs_vs_cs = (
        cusparse_ms / flagsparse_ms
        if (cusparse_ms is not None and flagsparse_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": int(indices.numel()),
            "value_dtype": value_dtype_req,
            "effective_value_dtype": _effective_dtype_name(value_dtype_req),
            "index_dtype": str(index_dtype),
            "mode": mode,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": flagsparse_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": fs_vs_pt,
            "triton_speedup_vs_cusparse": fs_vs_cs,
        },
        "verification": {
            "triton_match_pytorch": fs_match,
            "triton_max_error": fs_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
            "index_fallback_applied": bool(gather_meta.get("index_fallback_applied")),
            "index_fallback_reason": gather_meta.get("index_fallback_reason"),
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": flagsparse_values,
        },
    }


def _status_from_result(verification):
    triton_ok = verification.get("triton_match_pytorch")
    cusparse_ok = verification.get("cusparse_match_pytorch")
    overall_ok = bool(triton_ok) and (cusparse_ok is None or bool(cusparse_ok))
    return "PASS" if overall_ok else "FAIL"


def _print_header():
    print("-" * 196)
    print(
        f"{'ValueReq':>14} {'ValueEff':>18} {'Index':>6} {'Dense':>10} {'NNZ':>10} "
        f"{'IFB':>4} {'PT(ms)':>10} {'FS(ms)':>10} {'CS(ms)':>10} "
        f"{'FS/PT':>8} {'FS/CS':>8} {'Status':>6} {'Err(FS)':>12} {'Err(CS)':>12}"
    )
    print("-" * 196)


def _print_row(row):
    print(
        f"{row['value_dtype_req']:>14} {row['value_dtype_compute']:>18} {row['index_dtype']:>6} "
        f"{row['dense_size']:>10,d} {row['nnz']:>10,d} {str(row['index_fallback_applied']):>4} "
        f"{_fmt_ms(row['pytorch_ms']):>10} {_fmt_ms(row['triton_ms']):>10} {_fmt_ms(row['cusparse_ms']):>10} "
        f"{_fmt_speedup(row['triton_speedup_vs_pytorch']):>8} {_fmt_speedup(row['triton_speedup_vs_cusparse']):>8} "
        f"{row['status']:>6} {_fmt_err(row['triton_max_error']):>12} {_fmt_err(row['cusparse_max_error']):>12}"
    )


def run_cli(args):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return

    value_dtype_tokens = _parse_value_dtypes(args.value_dtypes)
    index_dtype_pairs = _parse_index_dtypes(args.index_dtypes)
    run_cusparse = not args.no_cusparse
    work_cases = _parse_cases(args.cases)

    print("=" * 180)
    print("FLAGSPARSE GATHER BENCHMARK/VALIDATION")
    print("=" * 180)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Warmup: {args.warmup} | Iterations: {args.iters} | "
        f"index_fallback_policy: {args.index_fallback_policy}"
    )
    print()
    _print_header()

    summary_rows = []
    sample_rows = []
    total_cases = 0
    failed_cases = 0

    for value_dtype in value_dtype_tokens:
        for index_name, index_dtype in index_dtype_pairs:
            if not _is_supported_extra_gather_combo(value_dtype, index_dtype):
                continue
            for dense_size, nnz in work_cases:
                dense_size = int(dense_size)
                nnz = int(nnz)
                case_id = (
                    f"{value_dtype}|{index_name}|dense={dense_size}|nnz={nnz}|"
                    f"ifb_policy={args.index_fallback_policy}"
                )
                total_cases += 1
                try:
                    result = _benchmark_gather_case(
                        value_dtype_req=value_dtype,
                        index_dtype=index_dtype,
                        dense_size=dense_size,
                        nnz=nnz,
                        warmup=args.warmup,
                        iters=args.iters,
                        run_cusparse=run_cusparse,
                        index_fallback_policy=args.index_fallback_policy,
                    )
                    perf = result["performance"]
                    verify = result["verification"]
                    params = result["parameters"]
                    backend = result["backend_status"]
                    status = _status_from_result(verify)
                    if status != "PASS":
                        failed_cases += 1
                    row = {
                        "case_id": case_id,
                        "gpu": torch.cuda.get_device_name(0),
                        "value_dtype_req": params.get("value_dtype"),
                        "value_dtype_compute": str(params.get("effective_value_dtype")),
                        "index_dtype": str(params.get("index_dtype")).replace("torch.", ""),
                        "dense_size": int(params.get("dense_size")),
                        "nnz": int(params.get("nnz")),
                        "mode": params.get("mode"),
                        "index_fallback_policy": args.index_fallback_policy,
                        "index_fallback_applied": bool(
                            backend.get("index_fallback_applied")
                        ),
                        "triton_ms": perf.get("triton_ms"),
                        "pytorch_ms": perf.get("pytorch_ms"),
                        "cusparse_ms": perf.get("cusparse_ms"),
                        "triton_speedup_vs_pytorch": perf.get("triton_speedup_vs_pytorch"),
                        "triton_speedup_vs_cusparse": perf.get("triton_speedup_vs_cusparse"),
                        "triton_match_pytorch": verify.get("triton_match_pytorch"),
                        "cusparse_match_pytorch": verify.get("cusparse_match_pytorch"),
                        "triton_max_error": verify.get("triton_max_error"),
                        "cusparse_max_error": verify.get("cusparse_max_error"),
                        "cusparse_unavailable_reason": backend.get("cusparse_unavailable_reason"),
                        "index_fallback_reason": backend.get("index_fallback_reason"),
                        "status": status,
                    }
                    summary_rows.append(row)
                    _print_row(row)

                    if args.csv_samples:
                        sample_rows.extend(
                            _collect_samples(
                                case_id,
                                result["samples"].get("pytorch"),
                                result["samples"].get("triton"),
                                args.sample_limit,
                            )
                        )
                except Exception as exc:
                    failed_cases += 1
                    row = {
                        "case_id": case_id,
                        "gpu": torch.cuda.get_device_name(0),
                        "value_dtype_req": value_dtype,
                        "value_dtype_compute": "N/A",
                        "index_dtype": index_name,
                        "dense_size": dense_size,
                        "nnz": nnz,
                        "mode": _dtype_mode(value_dtype),
                        "index_fallback_policy": args.index_fallback_policy,
                        "index_fallback_applied": False,
                        "triton_ms": None,
                        "pytorch_ms": None,
                        "cusparse_ms": None,
                        "triton_speedup_vs_pytorch": None,
                        "triton_speedup_vs_cusparse": None,
                        "triton_match_pytorch": None,
                        "cusparse_match_pytorch": None,
                        "triton_max_error": None,
                        "cusparse_max_error": None,
                        "cusparse_unavailable_reason": str(exc),
                        "index_fallback_reason": str(exc),
                        "status": "ERROR",
                    }
                    summary_rows.append(row)
                    _print_row(row)

    print("-" * 196)
    print(f"Total cases: {total_cases}")
    print(f"Failed cases: {failed_cases}")
    print(f"Passed cases: {total_cases - failed_cases}")

    if args.csv_summary:
        summary_fields = [
            "case_id",
            "gpu",
            "value_dtype_req",
            "value_dtype_compute",
            "index_dtype",
            "dense_size",
            "nnz",
            "mode",
            "index_fallback_policy",
            "index_fallback_applied",
            "triton_ms",
            "pytorch_ms",
            "cusparse_ms",
            "triton_speedup_vs_pytorch",
            "triton_speedup_vs_cusparse",
            "triton_match_pytorch",
            "cusparse_match_pytorch",
            "triton_max_error",
            "cusparse_max_error",
            "cusparse_unavailable_reason",
            "index_fallback_reason",
            "status",
        ]
        _write_csv(args.csv_summary, summary_rows, summary_fields)
        print(f"Wrote summary CSV: {args.csv_summary}")

    if args.csv_samples:
        sample_fields = [
            "case_id",
            "pos",
            "expected_real",
            "expected_imag",
            "flagsparse_real",
            "flagsparse_imag",
            "abs_error",
        ]
        _write_csv(args.csv_samples, sample_rows, sample_fields)
        print(f"Wrote samples CSV: {args.csv_samples}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Gather benchmark/validation aligned with scatter-style CLI and CSV export."
    )
    parser.add_argument("--value-dtypes", default=DEFAULT_VALUE_DTYPES)
    parser.add_argument("--index-dtypes", default=DEFAULT_INDEX_DTYPES)
    parser.add_argument(
        "--cases",
        default=",".join(f"{dense}:{nnz}" for dense, nnz in DEFAULT_CASES),
        help="Comma-separated dense:nnz pairs, e.g. 32768:1024,131072:4096",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--index-fallback-policy", choices=["auto", "strict"], default="auto")
    parser.add_argument("--csv-summary", default=None)
    parser.add_argument("--csv-samples", default=None)
    parser.add_argument("--sample-limit", type=int, default=32)
    return parser


if __name__ == "__main__":
    cli_parser = build_parser()
    run_cli(cli_parser.parse_args())
