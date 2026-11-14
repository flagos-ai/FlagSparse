"""CSR route policy tests runnable with stdlib unittest, without GPU imports."""

import ast
import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/flagsparse/sparse_operations"


def test_fixed_nnz_plan_has_no_dynamic_boundary_compaction():
    tree = ast.parse((SOURCE / "_spmv_csr_nnz.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_run_mixed_fixed")
    calls = {n.func.attr for n in ast.walk(function)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"item", "cumsum", "unique_consecutive", "repeat_interleave", "nonzero"}
    # The only host structural read is the existing per-row-wave CSR boundary.
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "tolist" for n in ast.walk(function)) == 1


def test_panel_static_loop_does_not_redeclare_constexpr():
    tree = ast.parse((SOURCE / "_spmm_csr_kernels.py").read_text(encoding="utf-8"))
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "row_panel_kernel")
    loops = [n for n in ast.walk(kernel) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute)
             and n.iter.func.attr == "static_range"]
    assert loops
    for loop in loops:
        assert not any(isinstance(n, ast.AnnAssign)
                       and isinstance(n.annotation, ast.Attribute)
                       and n.annotation.attr == "constexpr" for n in ast.walk(loop))


spec = importlib.util.spec_from_file_location(
    "_csr_policy_test_config", SOURCE / "_spmv_csr_config.py"
)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class ConfigPolicy(unittest.TestCase):
    def test_targeted_configuration_and_capabilities(self):
        refill, shard = policy.TARGETED_ALGORITHMS
        cfg, _ = policy.resolve_config(refill, self.cuda)
        self.assertEqual((cfg["rows_per_program"], cfg["window_rows"]), (8, 128))
        hip, _ = policy.resolve_config(refill, self.hip)
        self.assertEqual(hip["rows_per_program"], 16)
        for alg, bad in ((refill, {"window_rows": 7}), (refill, {"lanes_per_row": 3}),
                         (refill, {"rows_per_program": 0}), (refill, {"output_shards": 2}),
                         (shard, {"output_shards": 3}), (shard, {"workspace_limit_bytes": 0}),
                         (shard, {"num_stages": 2}), (shard, {"output_shards": True})):
            with self.subTest(alg=alg, config=bad), self.assertRaises(ValueError):
                policy.resolve_config(alg, self.cuda, bad)
        self.assertNotIn(shard, policy.list_algorithms(op="non"))
        with self.assertRaises(NotImplementedError):
            policy.validate_support(shard, "non", "float16", "int32", "int32", self.cuda)
        unknown = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True, scan=True)
        with self.assertRaisesRegex(policy.BackendCapabilityUnavailable, "atomic"):
            policy.validate_support(shard, "trans", "float16", "int32", "int32", unknown)
        fp32_only = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True,
                                      scan=True, fp32_atomic=True, fp64_atomic=False)
        policy.validate_support(shard, "trans", "float16", "int32", "int32", fp32_only)
        with self.assertRaisesRegex(policy.BackendCapabilityUnavailable, "fp64_atomic"):
            policy.validate_support(shard, "trans", "float32", "int32", "int32", fp32_only)
        with self.assertRaises(ValueError):
            policy.assert_route_match(refill, cfg, alg=shard)

    def test_mixed_nnz_type_and_workspace_contract(self):
        caps = policy.BackendCaps("rocm", "gfx936", "hip", 64, 1024, True, True, True)
        spec = policy.algorithm_spec("csr_nnz_tile_reduce")
        self.assertEqual(spec["value_dtypes"], ("float16", "bfloat16"))
        cfg, _ = policy.resolve_config("csr_nnz_tile_reduce", caps)
        self.assertEqual(cfg["block_nnz"], 128)
        self.assertEqual(cfg["plan_mode"], "compact")
        fixed, _ = policy.resolve_config("csr_nnz_tile_reduce", caps, {"plan_mode": "fixed"})
        self.assertEqual(fixed["plan_mode"], "fixed")
        for override in ({"plan_mode": "other"}, {"plan_mode": "fixed", "reduce_block_size": 2}):
            with self.assertRaises(ValueError):
                policy.resolve_config("csr_nnz_tile_reduce", caps, override)
        for override in ({"block_nnz": 3}, {"workspace_bytes": 35}, {"num_stages": 2}):
            with self.assertRaises(ValueError):
                policy.resolve_config("csr_nnz_tile_reduce", caps, override)
        with self.assertRaises(NotImplementedError):
            policy.validate_support("csr_nnz_tile_reduce", "non", "float32", "int32", "int32", caps)

    def setUp(self):
        self.cuda = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True, scan=True, fp32_atomic=True, fp64_atomic=True)
        self.hip = policy.BackendCaps(
            "rocm", "gfx90a", "hip", 64, 1024, True, True, True, scan=True, fp32_atomic=True, fp64_atomic=True
        )

    def test_registry_support(self):
        self.assertEqual(set(policy.list_algorithms()), set(policy.ALGORITHMS))
        for alg in policy.NEW_ALGORITHMS:
            if alg in ("csr_nnz_tile_reduce", "csr_subgroup_pair_mixed"):
                self.assertNotIn(alg, policy.list_algorithms(dtype="float32"))
                self.assertIn(alg, policy.list_algorithms(dtype="bfloat16", backend="rocm"))
                continue
            self.assertEqual(alg in policy.list_algorithms("non", "torch.float32", "rocm"), "non" in policy.algorithm_spec(alg)["ops"])
            self.assertIn(alg, policy.list_algorithms("trans", "float32", "cuda"))
            self.assertEqual(alg in policy.list_algorithms("non", "complex64", "cuda"), "non" in policy.algorithm_spec(alg)["ops"])
            for backend in ("metax", "mthreads", "ascend"):
                self.assertNotIn(alg, policy.list_algorithms(backend=backend))
        with self.assertRaises(ValueError):
            policy.algorithm_spec("auto")
        with self.assertRaises(ValueError):
            policy.normalize_alg("compare")

    def test_all_dtype_ops_and_precision_requirements(self):
        no_fp64 = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, False, True, True, scan=True, fp32_atomic=True, fp64_atomic=True)
        for alg in policy.ALGORITHMS:
            for dtype in policy.algorithm_spec(alg)["value_dtypes"]:
                for op in policy.algorithm_spec(alg)["ops"]:
                    policy.validate_support(alg, op, dtype, "int32", "int64", self.cuda)
            spec = policy.algorithm_spec(alg)
            self.assertEqual(spec["compute_dtype_by_input"]["complex64"], "complex64")
            self.assertEqual(spec["compute_dtype_by_input"]["float16"], "float32")
        for alg in policy.NEW_ALGORITHMS:
            if alg in ("csr_nnz_tile_reduce", "csr_subgroup_pair_mixed"):
                continue
            policy.resolve_config(alg, no_fp64)
            for dtype in ("float16", "bfloat16", "complex64"):
                policy.validate_support(alg, "conj", dtype, "int64", "int32", no_fp64)
            for dtype in ("float32", "float64", "complex128"):
                with self.assertRaisesRegex(NotImplementedError, "FP64"):
                    policy.validate_support(
                        alg, "trans", dtype, "int64", "int32", no_fp64
                    )

    def test_backend_profiles_and_mixed_indices(self):
        for caps, rows in ((self.cuda, 8), (self.hip, 16)):
            config, source = policy.resolve_config("row_adaptive_split", caps)
            self.assertEqual(config["row_tile"]["rows_per_program"], rows)
            self.assertEqual(config["row_split_reduce"]["reduce_block_size"], 256)
            self.assertIn(caps.backend, source)
            for col in ("int32", "int64"):
                for ptr in ("int32", "int64"):
                    policy.validate_support(
                        "row_tile", "non", "float64", col, ptr, caps
                    )

    def test_unknown_capabilities_fail_closed(self):
        with self.assertRaises(NotImplementedError):
            policy.validate_support(
                "row_vector",
                "non",
                "float32",
                "int32",
                "int64",
                policy.BackendCaps("cuda"),
            )
        for backend in ("metax", "mthreads", "ascend"):
            with self.assertRaises(NotImplementedError):
                policy.resolve_config("row_tile", policy.BackendCaps(backend))
            policy.validate_support(
                "legacy_rowpar",
                "conj",
                "complex128",
                "int64",
                "int32",
                policy.BackendCaps(backend),
            )

    def test_invalid_explicit_config(self):
        for bad in (
            {"bad": 1},
            {"row_tile": {"lanes_per_row": 3}},
            {"row_vector": {"num_warps": 64}},
            {"short_row_threshold": 1024},
            {"row_split_reduce": {"reduce_block_size": 1}},
            {"row_vector": {"block_nnz": True}},
            {"row_tile": {"lanes_per_row": 64}},
        ):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                policy.resolve_config("row_tile", self.cuda, bad)
        with self.assertRaises(ValueError):
            policy.resolve_config(
                "row_tile", self.hip, {"row_vector": {"num_warps": 16}}
            )

    def test_configuration_is_copied_and_fixed(self):
        override = {"row_vector": {"block_nnz": 64}}
        config, _ = policy.resolve_config("row_vector", self.cuda, override)
        override["row_vector"]["block_nnz"] = 256
        self.assertEqual(config["row_vector"]["block_nnz"], 64)
        policy.assert_route_match(
            "row_vector",
            config,
            config={"row_vector": {"block_nnz": 64}},
            caps=self.cuda,
        )
        with self.assertRaises(ValueError):
            policy.assert_route_match("row_vector", config, "row_tile")
        with self.assertRaises(ValueError):
            policy.assert_route_match(
                "row_vector", config, config=override, caps=self.cuda
            )

    def test_fallback_is_narrow(self):
        self.assertTrue(
            policy.is_index_compatibility_error(
                RuntimeError("unsupported int64 indices")
            )
        )
        for message in (
            "CUDA out of memory",
            "illegal memory access int64",
            "int64 launch failed",
            "unsupported fp64",
        ):
            self.assertFalse(policy.is_index_compatibility_error(RuntimeError(message)))

    def test_bad_builtin_profile_is_rejected_and_recorded(self):
        with patch.dict(
            policy.ARCH_PROFILES, {("cuda", "80"): {"row_vector": {"num_warps": 128}}}
        ):
            config, source, rejections = policy.resolve_config(
                "row_vector", self.cuda, return_rejections=True
            )
        self.assertEqual(config["row_vector"]["num_warps"], 2)
        self.assertEqual(source, "cuda:conservative-v1")
        self.assertEqual(rejections[0]["profile"], "cuda:80")
        limited = policy.BackendCaps("rocm", "test", "hip", 64, 64, True, True, True)
        config, source = policy.resolve_config("row_tile", limited)
        self.assertEqual(config["row_split_reduce"]["reduce_num_warps"], 1)
        self.assertEqual(source, "rocm:resource-conservative-v1")
        with self.assertRaises(ValueError):
            policy.resolve_config("row_tile", limited, {"row_vector": {"num_warps": 2}})


def runtime_namespace():
    """Execute the real Python orchestration functions with an instrumented backend.

    Only imports and JIT kernels are excluded; decisions/timing are production code.
    """
    names = {
        "flagsparse_spmv_csr_run",
        "flagsparse_spmv_csr",
        "_spmv_phase",
        "_execute_spmv_route",
        "_execute_spmv_route_with_fallback",
        "_spmv_execution_matrix",
        "_spmv_transpose_scatter_enabled",
    }
    tree = ast.parse((SOURCE / "spmv_csr.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    class Prepared:
        def __init__(self):
            self.alg = self.alg_requested = "row_adaptive_split"
            self.op, self.transpose, self.n_rows = 0, False, 3
            self.non_gather_plan = None
            self.shape = (3, 4)
            self.backend_caps = policy.BackendCaps(
                "cuda", "80", "cuda", 32, 1024, True, True, True
            )
            self.config, self.config_source = policy.resolve_config(
                self.alg, self.backend_caps
            )
            self.config_rejections = []
            self.data = types.SimpleNamespace(
                dtype="float32", device="cuda", is_conj=lambda: False
            )
            self.kernel_indices = types.SimpleNamespace(dtype="int64")
            self.kernel_indptr = types.SimpleNamespace(dtype="int32")
            self._baseline_compute_dtype = "float64"
            self.index_fallback_policy = "auto"
            self.index_fallback_applied, self.index_fallback_reason = False, None

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 7.0

    package = types.ModuleType("_csr_policy_runtime")
    kernels = types.SimpleNamespace(
        build_plan=Mock(side_effect=lambda *args: object()),
        compute=Mock(side_effect=lambda p, x, y, *args: y),
    )
    package._spmv_csr_kernels = kernels
    sys.modules[package.__name__] = package
    env = {
        "__package__": package.__name__,
        "_csr_config": policy,
        "copy": copy,
        "deepcopy": deepcopy,
        "asdict": asdict,
        "PreparedCsrSpmv": Prepared,
        "SPMV_CSR_NEW_ALGORITHMS": policy.NEW_ALGORITHMS,
        "torch": types.SimpleNamespace(empty=Mock(return_value=object())),
        "_ACCEL": types.SimpleNamespace(Event=Event, synchronize=lambda: None),
        "_spmv_device_context": lambda device: nullcontext(),
        "_validate_spmv_x": lambda x, p: types.SimpleNamespace(
            resolve_conj=lambda: types.SimpleNamespace(contiguous=lambda: x)
        ),
        "_spmv_check_output": lambda *a: None,
        "get_spmv_csr_algorithm_spec": policy.algorithm_spec,
        "_normalize_spmv_op": lambda op, transpose=False: (
            {"non": 0, "trans": 1, "conj": 2}.get(op, op)
            if op is not None
            else int(transpose)
        ),
        "_spmv_op_transposes": lambda op: op in (1, 2),
        "_spmv_csr_default_backend": lambda: "rowpar",
        "_spmv_uses_int64_indices": lambda p: True,
        "_is_ascend_runtime": lambda: False,
        "_is_rocm_runtime": lambda: False,
    }
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(SOURCE / "spmv_csr.py"), "exec"
        ),
        env,
    )
    return env, Prepared(), kernels


class RuntimePolicy(unittest.TestCase):
    def test_each_run_builds_plan_and_diagnostics_are_separate(self):
        env, p, kernels = runtime_namespace()
        run = env["flagsparse_spmv_csr_run"]
        _, plain = run(p, object(), return_meta=True)
        self.assertEqual(kernels.build_plan.call_count, 1)
        _, detailed = run(p, object(), return_meta=True, timing=True)
        self.assertEqual(kernels.build_plan.call_count, 3)
        self.assertEqual(kernels.compute.call_count, 3)
        self.assertEqual(plain["ms"], detailed["ms"])
        self.assertEqual(
            detailed["ms"], detailed["process_cpu_ms"] + detailed["gpu_ms"]
        )
        self.assertNotEqual(
            detailed["ms"], detailed["process_gpu_ms"] + detailed["compute_ms"]
        )
        self.assertEqual(detailed["compute_dtype"], "float64")
        self.assertEqual(detailed["alg_resolved"], p.alg)

    def test_row_routes_have_no_plan_and_out_is_forwarded(self):
        env, p, kernels = runtime_namespace()
        p.alg = p.alg_requested = "row_tile"
        out = object()
        result, meta = env["flagsparse_spmv_csr_run"](
            p, object(), out=out, return_meta=True, timing=True
        )
        self.assertIs(result, out)
        kernels.build_plan.assert_not_called()
        self.assertEqual(meta["process_gpu_ms"], 0)

    def test_prepared_and_legacy_argument_conflicts(self):
        env, p, _ = runtime_namespace()
        for kw in (
            {"op": "trans"},
            {"alg": "row_tile"},
            {"config": {"short_row_threshold": 16}},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                env["flagsparse_spmv_csr_run"](p, object(), **kw)
        for kw in (
            {"alg": "row_tile", "use_opt": True},
            {"use_opt": False},
            {"transpose": True},
            {"shape": (9, 9)},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                env["flagsparse_spmv_csr"](prepared=p, x=object(), **kw)

    def test_unrelated_errors_are_never_retried(self):
        env, p, kernels = runtime_namespace()
        kernels.compute.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            env["flagsparse_spmv_csr_run"](p, object())
        self.assertEqual(kernels.compute.call_count, 1)


def native_fp32_predicate():
    """The production ROCm FP32-accumulation predicate, without GPU imports."""
    tree = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "uses_native_rocm_fp32_accumulation"
    ]
    torch = types.SimpleNamespace(float32="float32", float64="float64")
    env = {"torch": torch}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(SOURCE / "_spmv_csr_kernels.py"),
            "exec",
        ),
        env,
    )
    return env["uses_native_rocm_fp32_accumulation"]


class RocmFp32AccumulationPolicy(unittest.TestCase):
    """auto.mtx's 37-nnz rows broke FP32 on DCU while the limit was 8192.

    Native FP32 accumulation is only accurate enough for rows of at most 32
    non-zeros; nothing pinned the old 8192 limit, so it went unnoticed.
    """

    def setUp(self):
        self.allowed = native_fp32_predicate()

    def prepared(self, max_row_nnz, dtype="float32", backend="rocm"):
        caps = types.SimpleNamespace(backend=backend)
        return types.SimpleNamespace(
            data=types.SimpleNamespace(dtype=dtype),
            backend_caps=caps,
            max_row_nnz=max_row_nnz,
        )

    def test_row_length_boundary(self):
        self.assertTrue(self.allowed(self.prepared(0)))
        self.assertTrue(self.allowed(self.prepared(32)))
        self.assertFalse(self.allowed(self.prepared(33)))
        self.assertFalse(self.allowed(self.prepared(37)))  # auto.mtx
        self.assertFalse(self.allowed(self.prepared(8192)))

    def test_only_rocm_fp32_accumulates_natively(self):
        for backend in ("cuda", "metax", "mthreads", "ascend", None):
            self.assertFalse(self.allowed(self.prepared(1, backend=backend)))
        self.assertFalse(self.allowed(self.prepared(1, dtype="float64")))
        caps_less = self.prepared(1)
        caps_less.backend_caps = None
        self.assertFalse(self.allowed(caps_less))

    def test_production_and_benchmark_share_the_predicate(self):
        kernels = (SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8")
        bench = (SOURCE / "_spmv_csr_benchmark.py").read_text(encoding="utf-8")
        self.assertIn(
            "native_rocm_fp32 = uses_native_rocm_fp32_accumulation(prepared)",
            kernels,
        )
        self.assertIn("kernels.uses_native_rocm_fp32_accumulation(prepared)", bench)
        self.assertNotIn("max_row_nnz <= 8192", kernels)
        self.assertNotIn("tl.float32 if prepared.data.dtype == torch.float32", bench)


class IntegrationPolicy(unittest.TestCase):
    def test_cli_aliases_and_csv_fields(self):
        from tests.test_spmv_csr import FIELDS, parser

        args = parser().parse_args(
            [
                "--synthetic",
                "--dtype",
                "float64",
                "--index-dtype",
                "int64",
                "--csv",
                "result.csv",
                "--no-cusparse",
                "--timing",
            ]
        )
        self.assertEqual(args.dtypes, "float64")
        self.assertEqual(args.index_dtypes, "int64")
        self.assertEqual(args.csv_csr, "result.csv")
        self.assertTrue(args.no_vendor and args.timing)
        self.assertEqual(len(FIELDS), len(set(FIELDS)))
        for key in (
            "vendor_ms",
            "speedup_vs_vendor",
            "config",
            "alg_resolved",
            "ms",
            "gpu_ms",
        ):
            self.assertIn(key, FIELDS)
        self.assertNotIn("cu_ms", FIELDS)

    def test_single_accuracy_module_and_static_support(self):
        import ops_support

        paths = list((ROOT / "tests").rglob("test_spmv_csr_accuracy.py"))
        self.assertEqual(paths, [ROOT / "tests/pytest/test_spmv_csr_accuracy.py"])
        rows = ops_support.build_rows(SOURCE)
        for alg in policy.NEW_ALGORITHMS:
            cases = [row for row in rows if row["route"] == alg]
            spec = policy.algorithm_spec(alg)
            self.assertEqual(len(cases), len(spec["value_dtypes"]) * len(spec["index_dtypes"]) * len(spec["ops"]))
            self.assertEqual({row["op"] for row in cases}, set(spec["ops"]))
            self.assertEqual({row["status"] for row in cases}, {"UNVERIFIED"})
        tree = ast.parse((SOURCE / "spmv_csr.py").read_text(encoding="utf-8"))
        declaration = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "SPMV_CSR_NEW_ALGORITHMS" for t in n.targets))
        self.assertEqual(set(ast.literal_eval(declaration)), set(policy.NEW_ALGORITHMS))

    def test_runner_preserves_algorithm_identity_and_latency(self):
        import run_flagsparse_pytest as runner

        row = dict(
            matrix="same.mtx",
            dtype="float32",
            index_dtype="int32",
            indptr_dtype="int64",
            alg_resolved="row_tile",
            op="non",
            vendor_ms="3",
            ms="2",
            speedup_vs_vendor="1.5",
        )
        detail = runner._benchmark_json_detail(row, 0)
        self.assertEqual(detail["latency_base"], 3)
        self.assertEqual(detail["latency"], 2)
        self.assertEqual(detail["speedup"], 1.5)
        other = runner._row_shape(dict(row, alg_resolved="row_vector"), 1)
        self.assertNotEqual(detail["shape_detail"], other)


if __name__ == "__main__":
    unittest.main()


def test_stream_configuration_uses_shape_only_and_preserves_explicit_values():
    caps = policy.BackendCaps("rocm", "gfx936", "hip", 64, 1024, True, True, True)
    cfg, _ = policy.resolve_config("row_subgroup_stream", caps, mean_row_nnz=3)
    assert cfg["row_subgroup_stream"]["lanes_per_row"] == 2
    assert cfg["row_subgroup_stream"]["rows_per_program"] == 64
    cfg, _ = policy.resolve_config("row_subgroup_stream", caps,
        {"row_subgroup_stream": {"lanes_per_row": 8, "row_tiles_per_program": 2}}, mean_row_nnz=3)
    assert cfg["row_subgroup_stream"]["lanes_per_row"] == 8
    assert cfg["row_subgroup_stream"]["row_tiles_per_program"] == 2


def _extension_policies():
    # Import only pure Python policy sources; never initialize torch or a driver.
    package = types.ModuleType("_migrated_policy_tests")
    package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    result = []
    for filename in ("_spmm_csr_config", "_spmx_column_config"):
        name = f"{package.__name__}.{filename}"
        spec = importlib.util.spec_from_file_location(name, SOURCE / f"{filename}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if filename == "_spmm_csr_config":
            module.set_known_backends(("cuda", "rocm", "iluvatar"))
        result.append(module)
    return result


def test_migrated_policies_validate_capabilities_and_workspace():
    csr, column = _extension_policies()
    caps = csr.BackendCaps("rocm", "hip", "gfx936", 64, 1024, (1, 2, 4, 8),
                           True, True, True, True, True, True, True)
    for load in (1, 4, 8):
        cfg, _ = csr.resolve_config("csr_row_panel", "float16", 33, "row", caps,
                                    {"panel_load_nnz": load})
        assert cfg["panel_load_nnz"] == load
    for mode in ("compact", "fixed"):
        cfg, _ = csr.resolve_config("csr_nnz_panel_reduce", "bfloat16", 33, "col", caps,
                                    {"plan_mode": mode})
        assert cfg["plan_mode"] == mode
    for algorithm, override in (
        ("csr_row_panel", {"panel_load_nnz": 2}),
        ("csr_row_tile", {"panel_load_nnz": 4}),
        ("csr_row_panel", {"plan_mode": "fixed"}),
        ("csr_nnz_panel_reduce", {"plan_mode": "bad"}),
        ("csr_nnz_panel_reduce", {"plan_mode": "fixed", "reduce_block_size": 2}),
    ):
        with unittest.TestCase().assertRaises(ValueError):
            csr.resolve_config(algorithm, "float16", 33, "row", caps, override)
    for alg in csr.NEW_ALGORITHMS:
        cfg, info = csr.resolve_config(alg, "float16" if alg in ("csr_nnz_panel_reduce", "csr_panel_staged_mixed") else "complex128", 33, "col", caps)
        assert info["backend_caps"]["backend"] == "rocm"
        assert cfg["workspace_bytes"] == 256 * 1024 * 1024
        assert csr.algorithm_spec(alg)["validation"] == "unverified"
        wave, capacity = csr.workspace_geometry(33, 16, cfg)
        assert 3 * wave * capacity * 16 <= cfg["workspace_bytes"]
    for alg in column.COO_ALGORITHMS:
        cfg, _ = column.resolve_coo_config(alg, "float16" if alg in column.COO_MIXED_ALGORITHMS else "float64", caps)
        assert cfg["num_warps"] in caps.legal_num_warps
    for alg, ops in column.CSC_ALGORITHMS.items():
        cfg, _ = column.resolve_csc_config(alg, "bfloat16" if alg in column.CSC_MIXED_ALGORITHMS else "complex128", 7, caps, op=ops[0])
        assert cfg["num_warps"] in caps.legal_num_warps
    with unittest.TestCase().assertRaises(NotImplementedError):
        csr.resolve_config("csr_row_panel", "float32", 7, "row", csr.BackendCaps("unknown"))
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_coo_config("coo_segmented_stream_atomic", "float32", caps, {"blocks_per_program": 3})
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_csc_config("csc_col_vector", "float32", 1, caps, {"num_warps": 3}, op="trans")


def test_split_config_does_not_require_adaptive_threshold_order():
    csr, _ = _extension_policies()
    csr.set_known_backends(("rocm",))
    caps = csr.BackendCaps("rocm", "hip", "gfx936", 64, 1024, (1, 2, 4, 8),
                           True, True, True, True, True, True, True)
    cfg, _ = csr.resolve_config("csr_split_nnz_reduce", "float16", 7, "row", caps,
                                {"segment_nnz": 8, "reduce_block_size": 2, "workspace_bytes": 256})
    assert cfg["segment_nnz"] == 8
    with unittest.TestCase().assertRaises(ValueError):
        csr.resolve_config("csr_adaptive_tile_split", "float16", 7, "row", caps,
                           {"segment_nnz": 8})


def test_independent_mixed_configuration_contracts():
    csr, column = _extension_policies()
    caps = csr.BackendCaps("rocm", "hip", "gfx936", 64, 1024, (1, 2, 4, 8),
                           True, True, True, True, True, True, True)
    check = unittest.TestCase()
    for load in (1, 4, 8):
        for accumulators in (1, 2, 4):
            for stage in ("storage", "compute"):
                cfg, _ = csr.resolve_config("csr_panel_staged_mixed", "float16", 33, "row", caps,
                    {"panel_load_nnz": load, "panel_accumulators": accumulators, "value_stage": stage})
                assert cfg["panel_load_nnz"] == load and cfg["value_stage"] == stage
                column.resolve_csc_config("csc_panel_staged_mixed", "bfloat16", 33, caps,
                    {"panel_load_nnz": load, "panel_accumulators": accumulators, "value_stage": stage}, op="trans")
    for alg in column.COO_MIXED_ALGORITHMS:
        for prefetch in (1, 2):
            for blocks in (1, 2, 4, 8):
                if blocks % prefetch:
                    with check.assertRaises(ValueError):
                        column.resolve_coo_config(alg, "float16", caps,
                            {"prefetch_blocks": prefetch, "blocks_per_program": blocks})
                else:
                    column.resolve_coo_config(alg, "float16", caps,
                        {"prefetch_blocks": prefetch, "blocks_per_program": blocks})
        with check.assertRaises(TypeError):
            column.resolve_coo_config(alg, "float32", caps)
    for alg in column.CSC_MIXED_ALGORITHMS:
        with check.assertRaises(ValueError):
            column.resolve_csc_config(alg, "float16", 7, caps, op="non")
        with check.assertRaises(TypeError):
            column.resolve_csc_config(alg, "float32", 7, caps, op="trans")
    for alg in ("csr_row_panel", "csr_panel_staged_mixed"):
        with check.assertRaises(ValueError):
            csr.resolve_config(alg, "float16", 7, "row", caps, {"value_stage": "invalid"})
    with check.assertRaises(ValueError):
        column.resolve_coo_config("coo_segmented_stream_atomic", "float16", caps, {"prefetch_blocks": 2})
    with check.assertRaises(ValueError):
        column.resolve_csc_config("csc_col_panel", "float16", 7, caps, {"value_stage": "storage"}, op="trans")
    pcaps = policy.BackendCaps("rocm", "gfx936", "hip", 64, 1024, True, True, True)
    for chunks in (1, 2):
        cfg, _ = policy.resolve_config("csr_subgroup_pair_mixed", pcaps,
            {"csr_subgroup_pair_mixed": {"prefetch_chunks": chunks}})
        assert cfg["csr_subgroup_pair_mixed"]["prefetch_chunks"] == chunks
    with check.assertRaises(ValueError):
        policy.resolve_config("row_subgroup_stream", pcaps, {"csr_subgroup_pair_mixed": {"prefetch_chunks": 2}})
    assert policy.algorithm_spec("csr_subgroup_pair_mixed")["value_dtypes"] == ("float16", "bfloat16")


def test_independent_mixed_kernels_are_distinct_and_native():
    expected = {
        "_spmv_csr_kernels.py": ("subgroup_pair_mixed_kernel",),
        "_spmm_csr_kernels.py": ("panel_staged_mixed_kernel",),
        "spmv_coo.py": ("_coo_prefetch_mixed_kernel",),
        "spmv_csc.py": ("_csc_subgroup_pair_mixed_kernel", "_csc_panel_staged_mixed_kernel"),
    }
    for filename, names in expected.items():
        tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
        for name in names:
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = {n.func.id for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            assert not calls & {"row_panel_kernel", "row_subgroup_stream_kernel", "_coo_stream_panel_kernel", "flagsparse_spmv_csr", "flagsparse_spmm_csr"}
            for loop in ast.walk(function):
                if isinstance(loop, ast.For):
                    assert not any(isinstance(n, ast.AnnAssign) for n in ast.walk(loop))


def test_generic_p1_config_and_capabilities():
    import pytest
    from dataclasses import replace

    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True, scan=True, fp32_atomic=True, fp64_atomic=True)
    alg = "csr_rowgroup_stream_reduce"
    cfg, _ = policy.resolve_config(alg, caps)
    assert cfg == dict(rows_per_program=16, block_nnz=128, num_warps=4, num_stages=1)
    for dtype in policy.VALUE_DTYPES:
        policy.validate_support(alg, "conj", dtype, "int64", "int32", caps)
    for bad in ({"rows_per_program": 3}, {"block_nnz": 32}, {"atomic_mode": "all"}, {"num_stages": 2}):
        with pytest.raises(ValueError):
            policy.resolve_config(alg, caps, bad)
    with pytest.raises(policy.BackendCapabilityUnavailable):
        policy.resolve_config(alg, replace(caps, scan=False))
    csr, _ = _extension_policies()
    pcaps = csr.BackendCaps("cuda", "cuda", "80", 32, 1024, (1, 2, 4, 8),
                            True, True, True, True, True, True, True)
    name = "csr_nnz_segment_panel_atomic"
    for dtype in csr.DTYPES:
        cfg, _ = csr.resolve_config(name, dtype, 7, "row", pcaps)
        assert cfg["block_n"] == 8
        assert cfg["atomic_mode"] == "boundary"
    for bad in ({"atomic_mode": "invalid"}, {"block_nnz": 16}, {"panel_load_nnz": 4}):
        with pytest.raises(ValueError):
            csr.resolve_config(name, "float32", 32, "row", pcaps, bad)
    for field, dtype in (("scan", "float32"), ("fp32_atomic", "complex64"), ("fp64_atomic", "complex128")):
        with pytest.raises(csr.BackendCapabilityUnavailable):
            csr.resolve_config(name, dtype, 32, "row", replace(pcaps, **{field: None}))


def test_nnz_owner_search_has_explicit_block_shape():
    tree = ast.parse((SOURCE / "_spmm_csr_kernels.py").read_text(encoding="utf-8"))
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_nnz_owner")
    assert helper.args.args[-1].arg == "WIDTH"
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("shape", "reshape")
                   for n in ast.walk(helper))
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "nnz_segment_panel_atomic_kernel")
    calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_nnz_owner"]
    assert len(calls) == 3
    assert sum(isinstance(n.args[-1], ast.Constant) and n.args[-1].value == 1 for n in calls) == 2
    assert sum(isinstance(n.args[-1], ast.Name) and n.args[-1].id == "T" for n in calls) == 1


def test_csr_spmm_benchmark_uses_four_field_reference_contract():
    tree = ast.parse((ROOT / "tests/test_spmm_csr.py").read_text(encoding="utf-8"))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == "_build_pytorch_reference"]
    assert len(assignments) == 1
    assert isinstance(assignments[0].targets[0], ast.Tuple)
    assert len(assignments[0].targets[0].elts) == 4
