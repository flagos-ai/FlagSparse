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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flagsparse.sparse_operations._common import _ACCEL, _ACCEL_DEVICE_TYPE


ACCEL = _ACCEL


def accelerator_available() -> bool:
    return bool(ACCEL.is_available())


def accelerator_device() -> torch.device:
    return torch.device(_ACCEL_DEVICE_TYPE)


def add_type_combo_arguments(parser, *, spmm=False):
    """Extend the existing CLIs without changing their legacy defaults."""
    parser.add_argument("--type-combos", help="A:RHS:output triples, comma-separated")
    if spmm:
        parser.add_argument("--op-b", default="non", help="non,trans,conj or all (type-combos)")
    for option, kwargs in (
        ("--indptr-dtypes", dict(default=None)),
        ("--config", dict(default=None)),
        ("--fail-fast", dict(action="store_true")),
        ("--synthetic", dict(action="store_true")),
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
    if not args.type_combos:
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
    from tests.utils import filtered_avg_ms

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
        combos = parse_type_combos(args.type_combos, tokens)
        ops = selection(option("ops", "op", default="non") or "non", ("non", "trans", "conj"))
        op_bs = selection(args.op_b, ("non", "trans", "conj")) if matrix else ("non",)
        layouts = selection(option("layout", default="row"), ("row", "col")) if matrix else ("row",)
        indices = selection(option("index_dtypes", "index_dtype", default="int32"), ("int32", "int64"))
        pointers = selection(args.indptr_dtypes, ("int32", "int64")) if args.indptr_dtypes else None
        ns = tuple(int(n) for n in str(getattr(args, "dense_cols", 1)).split(","))
        if any(n < 0 for n in ns) or args.warmup < 0 or args.iters < 1:
            raise ValueError("require N >= 0, warmup >= 0 and iters >= 1")
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
    if not paths and not args.synthetic:
        parser.error("supply MatrixMarket inputs or --synthetic")
    if not accelerator_available():
        raise RuntimeError("an accelerator is required for type-combination benchmarks")
    module = importlib.import_module("flagsparse.sparse_operations." + operator)
    query = getattr(module, "list_" + operator + "_algorithms")
    prepare = getattr(module, "prepare_" + operator + ("_route" if matrix else ""))
    run = getattr(module, "flagsparse_" + operator + "_run")
    backend, device = _backend_name(), accelerator_device()
    csv_path = option("csv_" + fmt, "csv")
    fields = ["matrix", "dtype", "type_combo", "a_dtype", "rhs_dtype", "compute_dtype", "output_dtype",
              "op", "op_a", "op_b", "layout", "dense_cols", "index_dtype", "indptr_dtype",
              "alg", "alg_requested", "alg_resolved", "backend", "m", "n", "nnz", "status", "reason", "max_error",
              "ms", "gpu_ms", "process_cpu_ms", "process_gpu_ms", "compute_ms",
              "vendor_ms", "vendor_backend", "vendor_version", "vendor_alg", "vendor_preprocessed", "vendor_status",
              "vendor_reason", "vendor_max_error", "speedup_vs_vendor", "metadata", "timed", "sample_rule"]
    stream = open(csv_path, "w", newline="", encoding="utf-8") if csv_path else None
    writer = csv.DictWriter(stream, fields, extrasaction="ignore") if stream else None
    if writer:
        writer.writeheader()
        stream.flush()
    print("Registered type combinations; prepare excluded; full run includes required conversions and reductions.")
    print("ms=process_cpu_ms+gpu_ms; median +/-10% mean; phase diagnostics measured separately.")
    print(f"{'Matrix':24} {'A/RHS/Out':23} {'A/B':9} {'Lay/N':9} {'Algorithm':32} {'ms':>9} {'CPU':>8} {'ProcGPU':>8} {'Compute':>8} {'Vendor':>9} {'V/Alg':>8} Status")
    reported_vendor_reasons = set()

    def emit(row):
        if writer:
            writer.writerow({k: json.dumps(v, default=str, sort_keys=True) if isinstance(v, dict) else v
                             for k, v in row.items()})
            stream.flush()
        def number(key):
            value = row.get(key)
            return "N/A" if value is None else f"{value:.4f}"
        combo = "/".join(row[k].replace("float", "f").replace("bf", "bf") for k in ("a_dtype", "rhs_dtype", "output_dtype"))
        print(f"{Path(row['matrix']).name:24} {combo:23} {row['op_a']+'/'+row['op_b']:9} "
              f"{row['layout']+'/'+str(row['dense_cols']):9} {row['alg']:32} "
              + " ".join(f"{number(k):>8}" for k in ("ms", "process_cpu_ms", "process_gpu_ms", "compute_ms", "vendor_ms", "speedup_vs_vendor"))
              + f" {row['status']}")
        if row.get("reason"):
            print(row["reason"])
        reason = row.get("vendor_reason")
        if reason and reason not in reported_vendor_reasons:
            print(f"Vendor {row.get('vendor_status', 'SKIP')}: {reason}")
            reported_vendor_reasons.add(reason)

    def accuracy(result, golden, dtype):
        actual = result.detach().cpu()
        reference = torch.as_tensor(golden).to(dtype)
        error = float((actual.double() - reference.double()).abs().max()) if actual.numel() else 0.0
        rtol, atol = close_tolerances(dtype)
        return ("PASS" if torch.allclose(actual, reference, rtol=rtol, atol=atol) else "FAIL"), error

    try:
        sources = paths + ([None] if args.synthetic else [])
        for source, combo in itertools.product(sources, combos):
            dtype, rhs_dtype, output_dtype = (getattr(torch, item) for item in combo)
            # Construct once per matrix/type; quantize before forming the CPU golden.
            if source is None:
                import scipy.sparse as sparse
                host = sparse.coo_matrix(([1., .00390625, -.125, .5], ([0, 0, 0, 2], [0, 0, 1, 2])), shape=(3, 4)).asformat(fmt)
                a = torch.tensor(host.data, dtype=dtype)
                first = torch.tensor(host.row if fmt == "coo" else host.indices, dtype=torch.int64)
                second = torch.tensor(host.col if fmt == "coo" else host.indptr, dtype=torch.int64)
                shape = host.shape
            else:
                a, first, second, shape = getattr(mtx_fast, "load_" + fmt)(source, dtype=dtype, device="cpu")
            import scipy.sparse as sparse
            values = a.double().numpy()
            host = (sparse.coo_matrix((values, (first.numpy(), second.numpy())), shape=shape)
                    if fmt == "coo" else getattr(sparse, fmt + "_matrix")((values, first.numpy(), second.numpy()), shape=shape))
            for op, op_b, layout, ncols in itertools.product(ops, op_bs, layouts, ns):
                inner = shape[1] if op == "non" else shape[0]
                generator = torch.Generator().manual_seed(2026)
                effective = torch.randn((inner, ncols) if matrix else (inner,), generator=generator).to(rhs_dtype)
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
                    arrays = (a.to(device), first.to(device=device, dtype=getattr(torch, index)),
                              second.to(device=device, dtype=getattr(torch, ptr)))
                    for alg in selected:
                        row = dict(matrix=str(source or "synthetic"), dtype=combo[0], type_combo=":".join(combo), a_dtype=combo[0],
                                   rhs_dtype=combo[1], compute_dtype="float32", output_dtype=combo[2], op=op,
                                   op_a=op, op_b=op_b, layout=layout, dense_cols=ncols, index_dtype=index,
                                   indptr_dtype=ptr, alg=alg, alg_requested=alg, backend=backend, m=shape[0], n=shape[1], nnz=a.numel(),
                                   timed="prepared_run", sample_rule="median +/-10% mean", status="ERROR")
                        try:
                            prepared = prepare(*arrays, shape, op=op, alg=alg, config=config)
                            kwargs = dict(out_dtype=output_dtype, return_time=True, return_meta=True)
                            if matrix:
                                kwargs["op_b"] = op_b
                                if fmt in ("csr", "coo"):
                                    kwargs["dense_layout"] = layout
                            # Compile and validate once even when --warmup=0.
                            run(prepared, rhs, **kwargs)
                            for _ in range(args.warmup):
                                run(prepared, rhs, **kwargs)
                            gpu, cpu = [], []
                            for _ in range(args.iters):
                                result, elapsed, meta = run(prepared, rhs, **kwargs)
                                gpu.append(float(meta.get("gpu_ms", elapsed)))
                                cpu.append(float(meta.get("process_cpu_ms", 0.0)))
                            row.update(gpu_ms=filtered_avg_ms(gpu), process_cpu_ms=filtered_avg_ms(cpu))
                            row["ms"] = row["gpu_ms"] + row["process_cpu_ms"]
                            row["status"], row["max_error"] = accuracy(result, golden, output_dtype)
                            if args.timing:
                                _, _, diagnostics = run(prepared, rhs, timing=True, **kwargs)
                                row.update({key: diagnostics.get(key) for key in ("process_gpu_ms", "compute_ms")})
                            row["metadata"] = meta
                            row["alg_resolved"] = meta.get("alg_resolved", meta.get("alg", alg))
                            row["vendor_status"] = "SKIP"
                            row["vendor_reason"] = "disabled"
                            if not (getattr(args, "no_vendor", False) or getattr(args, "no_cusparse", False)):
                                plan = None
                                try:
                                    target = torch.empty_like(result)
                                    plan = (vendor.prepare_spmm(fmt, arrays, shape, rhs, target, op, op_b, torch.float32, backend=backend)
                                            if matrix else vendor.prepare_spmv(fmt, arrays, shape, rhs, target, op, torch.float32, backend=backend))
                                    from tests.utils import cuda_event_benchmark_filtered
                                    plan.run()
                                    actual, vendor_ms = cuda_event_benchmark_filtered(plan.run, args.warmup, args.iters)
                                    row.update({key: value for key, value in plan.metadata.items() if key.startswith("vendor_")},
                                               vendor_ms=vendor_ms, vendor_reason="")
                                    row["vendor_status"], row["vendor_max_error"] = accuracy(actual, golden, output_dtype)
                                    row["speedup_vs_vendor"] = vendor_ms / row["ms"] if row["ms"] > 0 else None
                                except NotImplementedError as exc:
                                    row["vendor_reason"] = str(exc)
                                except Exception as exc:
                                    row.update(vendor_status="ERROR", vendor_reason=str(exc))
                                    traceback.print_exc()
                                    if args.fail_fast:
                                        raise
                                finally:
                                    if plan is not None:
                                        plan.close()
                        except BackendCapabilityUnavailable as exc:
                            row.update(status="SKIP", reason=str(exc))
                        except Exception as exc:
                            row.update(status="ERROR", reason=f"{operator}/{alg}/{combo}/{op}/{op_b}: {exc}")
                            traceback.print_exc()
                            if args.fail_fast:
                                emit(row)
                                raise
                        emit(row)
    finally:
        if stream:
            stream.close()
    return True
