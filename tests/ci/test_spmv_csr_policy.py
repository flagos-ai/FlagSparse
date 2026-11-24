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


def test_irregular_spmv_profiles_and_capabilities():
    from dataclasses import replace
    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True,
                              scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True)
    local, grid = "csr_nnz_local_reduce_atomic", "csr_longrow_grid_atomic"
    assert policy.algorithm_spec(local)["implementation_version"] == 2
    # Fixing the runtime branch typing does not change launch defaults or turn
    # the single-owner shortcut into a different algorithm selection.
    defaults, _ = policy.resolve_config(local, caps)
    assert defaults == dict(block_nnz=256, owner_search="bounded", single_row_fastpath=True,
                            atomic_mode="boundary", num_warps=4, num_stages=1)
    for search in ("bounded", "full"):
        for fast in (False, True):
            supplied = dict(owner_search=search, single_row_fastpath=fast)
            cfg, source = policy.resolve_config(local, caps, supplied)
            assert cfg == dict(defaults, **supplied) and source == "explicit"
            assert supplied == dict(owner_search=search, single_row_fastpath=fast)
    for alg in (local, grid):
        for dtype in policy.VALUE_DTYPES:
            for op in policy.algorithm_spec(alg)["ops"]:
                policy.validate_support(alg, op, dtype, "int64", "int32", caps)
        assert policy.algorithm_spec(alg)["half_output_dtypes"]["bfloat16"] == ("bfloat16", "float32")
        for field in ("scan", "fp32_atomic"):
            with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
                policy.validate_support(alg, "non", "float16", "int32", "int64", replace(caps, **{field: None}))
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.validate_support(alg, "non", "float32", "int32", "int32", replace(caps, fp64_atomic=None))
    with unittest.TestCase().assertRaises(NotImplementedError):
        policy.validate_support(grid, "trans", "float16", "int32", "int32", caps)
    for invalid in ({"single_row_fastpath": 1}, {"block_nnz": True}, {"block_nnz": 64},
                    {"owner_search": "guess"}, {"atomic_mode": "invalid"}, {"num_stages": 2},
                    {"worker_programs": 128}):
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config(local, caps, invalid)
    with unittest.TestCase().assertRaises(ValueError):
        policy.resolve_config(grid, caps, {"worker_programs": 128})
    cfg, _ = policy.resolve_config(grid, caps)
    original, _ = policy.resolve_config("csr_longrow_queue_atomic", caps)
    assert cfg == {k: v for k, v in original.items() if k != "worker_programs"}
    with unittest.TestCase().assertRaises(ValueError):
        policy.assert_route_match(local, policy.resolve_config(local, caps)[0], grid, None, caps)


def test_short_rowqueue_policy_capacity_and_no_fp_atomic_requirement():
    from dataclasses import replace
    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True,
                              scan=True, fp32_atomic=False, fp64_atomic=False, int32_atomic=True)
    alg = "csr_short_rowqueue_vector"
    assert policy.algorithm_spec(alg)["ops"] == ("non",)
    for dtype in policy.VALUE_DTYPES:
        policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
    for op in ("trans", "conj"):
        with unittest.TestCase().assertRaises(NotImplementedError):
            policy.validate_support(alg, op, "float16", "int32", "int32", caps)
    cfg, _ = policy.resolve_config(alg, caps)
    assert cfg["short_row_threshold"] == 32 and cfg["block_nnz"] == 128
    for invalid in ([], "", 0):
        with unittest.TestCase().assertRaises(TypeError):
            policy.resolve_config(alg, caps, invalid)
    for threshold in (0, 8, 16, 32, 64, 128, 256):
        resolved, _ = policy.resolve_config(alg, caps, {"short_row_threshold": threshold})
        for lengths in ([], [0] * 19, [0, 4, 128, 602, 713, 1024], [threshold + 1] * 129):
            result = policy.short_rowqueue_capacity(len(lengths), sum(lengths), resolved)
            assert sum(n > threshold for n in lengths) <= result["row_queue_capacity"] <= len(lengths)
            assert result["numeric_workspace_bytes"] == 0
            assert result["workspace_peak_bytes"] == result["row_queue_capacity"] * 8 + 4
    for invalid in ({"segment_nnz": 128}, {"worker_programs": 64}, {"short_row_threshold": True},
                    {"short_row_threshold": 33}, {"rows_per_program": 16}, {"block_nnz": 32},
                    {"workspace_limit_bytes": 0}, {"num_stages": 2}):
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config(alg, caps, invalid)
    for feature in ("scan", "int32_atomic"):
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.resolve_config(alg, replace(caps, **{feature: None}))
    with unittest.TestCase().assertRaises(ValueError):
        policy.short_rowqueue_capacity(2**31, (2**31) * 33, cfg)
    with unittest.TestCase().assertRaises(ValueError):
        policy.short_rowqueue_capacity(3, 99, dict(cfg, workspace_limit_bytes=27))
    assert policy.short_rowqueue_capacity(3, 99, dict(cfg, workspace_limit_bytes=28))["row_queue_capacity"] == 3
    with unittest.TestCase().assertRaises(ValueError):
        policy.assert_route_match(alg, cfg, "csr_longrow_queue_atomic", None, caps)


def test_bounded_rowqueue_profile_isolation_and_capabilities():
    from dataclasses import replace
    alg = "csr_short_rowqueue_strided_vector"
    old = "csr_short_rowqueue_vector"
    for caps in (policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True,
                                  scan=True, int32_atomic=True, fp32_atomic=False, fp64_atomic=False),
                 policy.BackendCaps("rocm", "gfx90a", "hip", 64, 1024, True, True, True,
                                  scan=True, int32_atomic=True, fp32_atomic=False, fp64_atomic=False)):
        cfg, source = policy.resolve_config(alg, caps)
        original, _ = policy.resolve_config(old, caps)
        assert cfg == dict(original, short_row_threshold=128, grid_blocks=256)
        assert source == "bounded-rowqueue-conservative-v1"
        assert original["short_row_threshold"] == 32 and "grid_blocks" not in original
        assert policy.algorithm_spec(alg)["ops"] == ("non",)
        assert policy.algorithm_spec(alg)["config_candidates"]["grid_blocks"] == (128, 256, 512, 1024)
        for dtype in policy.VALUE_DTYPES:
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
        for op in ("trans", "conj"):
            assert alg not in policy.list_algorithms(op, "float32", caps.backend)
            with unittest.TestCase().assertRaises(NotImplementedError):
                policy.validate_support(alg, op, "float32", "int32", "int32", caps)
        for grid in (1, 3, 128, 256, 512, 1024):
            override = dict(grid_blocks=grid, short_row_threshold=32)
            snapshot = dict(override)
            resolved, origin = policy.resolve_config(alg, caps, override)
            assert resolved == dict(original, **override) and origin == "explicit"
            assert override == snapshot
            for rows, nnz in ((0, 0), (100, 0), (1, 129), (10000, 330000)):
                bounded = policy.short_rowqueue_capacity(rows, nnz, resolved)
                assert bounded == policy.short_rowqueue_capacity(rows, nnz, original)
        for override in ({"grid_blocks": False}, {"grid_blocks": 0}, {"grid_blocks": -1},
                         {"grid_blocks": 1.5}, {"grid_blocks": 1025}, {"grid_blocks": "256"},
                         {"worker_programs": 128}, {"segment_nnz": 128}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, override)
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config(old, caps, {"grid_blocks": 256})
        for target in (old, "row_vector"):
            with unittest.TestCase().assertRaises(ValueError):
                policy.assert_route_match(alg, cfg, target, None, caps)
        for feature in ("scan", "int32_atomic"):
            with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
                policy.resolve_config(alg, replace(caps, **{feature: None}))
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.validate_support(alg, "non", "float32", "int32", "int32", replace(caps, fp64=False))


def test_cooperative_queue_contract_and_workspace():
    from dataclasses import replace
    alg = "csr_longrow_cooperative_reduce"
    for backend, subgroup in (("cuda", 32), ("rocm", 64)):
        caps = policy.BackendCaps(backend, "unknown", backend, subgroup, 1024, True, True, True,
                                 scan=True, int32_atomic=True, fp32_atomic=False, fp64_atomic=False)
        cfg, source = policy.resolve_config(alg, caps)
        assert cfg["cooperative_parts"] == 8 and cfg["long_row_threshold"] == 1024
        assert source == "longrow-cooperative-conservative-v1"
        spec = policy.algorithm_spec(alg)
        assert spec["ops"] == ("non",) and spec["value_dtypes"] == policy.VALUE_DTYPES
        assert "atomic_by_compute_dtype" not in spec["capability_requirements"]
        for dtype in policy.VALUE_DTYPES:
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
            stats = policy.cooperative_queue_capacity(100, 2050, cfg, dtype)
            width = {"float16": 4, "bfloat16": 4, "float32": 8, "float64": 8,
                     "complex64": 8, "complex128": 16}[dtype]
            assert stats["row_queue_capacity"] == 2 and stats["partial_slot_capacity"] == 16
            assert stats["numeric_workspace_bytes"] == 16 * width
            assert stats["descriptor_peak_bytes"] == 20
            assert stats["workspace_peak_bytes"] == 20 + 16 * width
            exact = dict(cfg, workspace_limit_bytes=stats["workspace_peak_bytes"])
            assert policy.cooperative_queue_capacity(100, 2050, exact, dtype) == stats
            with unittest.TestCase().assertRaisesRegex(ValueError, "workspace"):
                policy.cooperative_queue_capacity(100, 2050, dict(exact, workspace_limit_bytes=exact["workspace_limit_bytes"] - 1), dtype)
            for op in ("trans", "conj"):
                assert alg not in policy.list_algorithms(op=op, dtype=dtype, backend=backend)
                with unittest.TestCase().assertRaises(NotImplementedError):
                    policy.validate_support(alg, op, dtype, "int32", "int32", caps)
        assert policy.cooperative_queue_capacity(19, 0, cfg, "float16")["workspace_peak_bytes"] == 4
        with unittest.TestCase().assertRaisesRegex(ValueError, "int32"):
            policy.cooperative_queue_capacity(2**31, 2**31 * 1025, cfg, "float16")
        for bad in ({"cooperative_parts": 3}, {"cooperative_parts": True}, {"grid_blocks": 0},
                    {"grid_blocks": 1025}, {"long_row_threshold": -1}, {"long_row_threshold": 2.5},
                    {"block_nnz": 16}, {"num_stages": 2}, {"rows_per_program": 3},
                    {"workspace_limit_bytes": 0}, {"num_warps": 1024}, {"segment_nnz": 128},
                    {"short_row_threshold": 32}, {"worker_programs": 128}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, bad)
        for feature in ("scan", "int32_atomic"):
            with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
                policy.resolve_config(alg, replace(caps, **{feature: None}))
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.validate_support(alg, "non", "float32", "int32", "int32", replace(caps, fp64=False))
        for other in policy.ROWQUEUE_ALGORITHMS + policy.LONGROW_ALGORITHMS:
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(other, caps, {"cooperative_parts": 8})
            with unittest.TestCase().assertRaises(ValueError):
                policy.assert_route_match(alg, cfg, other, None, caps)
        supplied = {"cooperative_parts": 1, "long_row_threshold": 0, "grid_blocks": 3}
        resolved, _ = policy.resolve_config(alg, caps, supplied)
        assert resolved == dict(cfg, **supplied) and supplied == {"cooperative_parts": 1, "long_row_threshold": 0, "grid_blocks": 3}


def test_irregular_spmv_execution_has_no_host_plan_reads():
    tree = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("run_nnz_local_reduce", "run_longrow_grid", "run_short_rowqueue", "run_short_rowqueue_strided", "run_longrow_cooperative", "run_three_tier_queue", "run_three_tier_bounded", "run_three_tier_ticket"):
        calls = {n.func.attr for n in ast.walk(functions[name]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not calls & {"item", "tolist", "nonzero", "cumsum", "repeat_interleave", "unique_consecutive"}
    owner = functions["_nnz_local_owner"]
    assert owner.args.args[-1].arg == "WIDTH"
    assert not any(isinstance(n, ast.Attribute) and n.attr == "shape" for n in ast.walk(owner))
    consumer = ast.unparse(functions["longrow_grid_consumer_kernel"])
    assert "program_id" in consumer and "COUNTERS + 2" not in consumer
    assert "while" not in consumer and "PARTIAL" in consumer
    row_consumer = ast.unparse(functions["short_rowqueue_vector_kernel"])
    assert "atomic_add" not in row_consumer and "store" in row_consumer
    assert "PARTIAL" not in row_consumer and "tl.load(COUNT)" in row_consumer
    launch = ast.unparse(functions["run_short_rowqueue"])
    assert "torch.zeros(1, dtype=torch.int32" in launch
    assert "y.zero_" not in launch and "y.copy_" not in launch
    bounded = functions["short_rowqueue_strided_vector_kernel"]
    body = ast.unparse(bounded)
    assert "atomic" not in body and "tl.num_programs(0).to(tl.int64)" in body
    loop = next(node for node in bounded.body if isinstance(node, ast.While))
    assert ast.unparse(loop.test) == "slot < count"
    assert any(isinstance(node, ast.AugAssign) and ast.unparse(node) == "slot += grid_stride" for node in loop.body)
    for accumulator in ("re", "im"):
        assert any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == accumulator for t in node.targets)
                   and "tl.zeros" in ast.unparse(node.value) for node in loop.body)
    launch = ast.unparse(functions["run_short_rowqueue_strided"])
    assert "min(capacity, cfg['grid_blocks'])" in launch
    consumer_launch = next(node for node in ast.walk(functions["run_short_rowqueue_strided"])
                           if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                           and node.value.id == "short_rowqueue_strided_vector_kernel")
    assert isinstance(consumer_launch.slice, ast.Tuple)
    assert [ast.unparse(node) for node in consumer_launch.slice.elts] == ["grid"]
    assert "short_rowqueue_producer_kernel" in launch and "PARTIAL" not in launch
    for name in ("cooperative_row_consumer_kernel", "cooperative_row_finish_kernel"):
        body = ast.unparse(functions[name])
        assert "atomic" not in body and "tl.num_programs(0).to(tl.int64)" in body
        assert "slot < count" in body and "slot += stride" in body
    launch = ast.unparse(functions["run_longrow_cooperative"])
    assert "torch.empty(capacity * parts" in launch
    assert "y.zero_" not in launch and "y.copy_" not in launch
    assert "cooperative_row_producer_kernel" in launch
    assert "cooperative_row_consumer_kernel" in launch and "cooperative_row_finish_kernel" in launch


def test_csr_tile_structure_counts_without_device_imports():
    # Extract only the CPU helper: this policy test must not initialize a device.
    tree = ast.parse((ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csr_tile_structure")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "csr_structure_policy", "exec"), namespace)
    analyze = namespace["csr_tile_structure"]
    result = analyze([0, 0, 8, 8, 9, 10], tiles=(4,), config={"long_row_threshold": 4, "segment_nnz": 4})
    tile = result["tiles"]["4"]
    assert tile["tile_count"] == 3 and tile["single_owner_tile_count"] == 2
    assert tile["crossing_row_count"] == 1 and tile["full_row_count"] == 2
    assert tile["predicted_boundary_atomic_updates"] == 2
    assert tile["mixed_tile_row_span_p50"] == 2
    assert result["queue"]["expected_jobs"] == 2 and result["queue"]["job_capacity"] == 4
    assert result["row_lengths"]["empty_rows"] == 2 and result["row_lengths"]["maximum"] == 8
    assert result["atomic_count_basis"] == "per_real_component_logical_updates"
    assert result["scalar_groups"]["32"]["scheduled_positions"] == 40
    assert result["scalar_groups"]["32"]["utilization"] == 0.25
    empty = analyze([0, 0], tiles=(4,))["tiles"]["4"]
    assert empty["single_owner_tile_ratio"] is None and empty["ratio_reason"] == "no_nonzeros"
    with unittest.TestCase().assertRaises(ValueError):
        analyze([0, 2, 1])
    row_queue = analyze([0, 0, 8, 8, 9, 10], config={"short_row_threshold": 4,
                        "producer_num_warps": 2, "rows_per_program": 64})["row_queue"]
    assert row_queue["expected_queued_rows"] == 1 and row_queue["row_queue_capacity"] == 2
    assert row_queue["queued_nnz_fraction"] == 0.8
    assert row_queue["expected_active_program_ratio"] == 0.5
    bounded = analyze([0, 0, 8, 8, 9, 10], config={"short_row_threshold": 4,
                       "producer_num_warps": 2, "rows_per_program": 64, "grid_blocks": 1})["row_queue"]
    assert bounded["row_queue_capacity"] == 2 and bounded["consumer_grid_blocks"] == 1
    assert bounded["expected_active_program_ratio"] == 1 and bounded["expected_max_rounds"] == 1
    assert bounded["consumer_rounds_capacity"] == 2
    rounds = analyze([0, 8, 16, 24], config={"short_row_threshold": 4,
                      "producer_num_warps": 2, "rows_per_program": 64, "grid_blocks": 2})["row_queue"]
    assert rounds["expected_max_rounds"] == 2 and rounds["expected_tasks_per_program"] == 1.5
    with unittest.TestCase().assertRaises(ValueError):
        analyze([0, 1], config={"short_row_threshold": 4, "producer_num_warps": 2,
                              "rows_per_program": 64, "grid_blocks": True})
    cooperative = analyze([0, 0, 8, 8, 9, 10], config={"long_row_threshold": 4,
                           "cooperative_parts": 4, "block_nnz": 4, "grid_blocks": 1})["cooperative_queue"]
    assert cooperative["expected_queued_rows"] == 1 and cooperative["row_queue_capacity"] == 2
    assert cooperative["partial_slot_capacity"] == 8 and cooperative["consumer_grid_blocks"] == 4
    assert cooperative["expected_nonempty_part_tasks"] == 2 and cooperative["expected_empty_part_tasks"] == 2
    assert cooperative["queued_nnz_fraction"] == 0.8 and cooperative["consumer_rounds_capacity"] == 2


def test_coo_csc_structure_models_without_device_imports():
    import numpy as np
    tree = ast.parse((ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("coo_run_structure", "csc_column_structure")]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "column_structure_policy", "exec"), namespace)
    analyze = namespace["coo_run_structure"]
    # Includes the first==last trap, mixed runs and one-valid-lane padding.
    owners = [1, 1, 1, 3, 7, 3, 2, 2, 3, 4]
    tile = analyze(owners, tiles=(3,))["tiles"]["3"]
    assert tile["counts"] == dict(single_owner=2, no_adjacent_equal=1, segmented=1)
    assert tile["nnz_by_class"] == dict(single_owner=4, no_adjacent_equal=3, segmented=3)
    assert sum(tile["nnz_by_class"].values()) == len(owners)
    assert analyze([], tiles=(3,))["tiles"]["3"]["tile_fraction"]["single_owner"] is None
    # Force multiple bounded chunks and a partial final tile.
    large = analyze(np.ones(2**20 + 1, dtype=np.int64), tiles=(128,))["tiles"]["128"]
    assert large["counts"]["single_owner"] == 8193
    assert large["nnz_by_class"]["single_owner"] == 2**20 + 1
    csc = namespace["csc_column_structure"]([0, 0, 3, 4], [1, 1, 2, 0], 4)
    assert csc["columns"]["empty"] == 1 and csc["columns"]["maximum"] == 3
    assert csc["output_row_contributions"]["maximum"] == 2
    assert "output_row_contributions" not in namespace["csc_column_structure"]([0], [], 0, op="trans")


def test_runclass_configuration_and_capability_contracts():
    from dataclasses import replace
    csr, column = _extension_policies()
    caps = csr.BackendCaps("cuda", "cuda", "80", 32, 1024, (1, 2, 4, 8),
                          True, True, True, True, True, True, True)
    for alg in column.COO_RUNCLASS_ALGORITHMS:
        for dtype in column.DTYPES:
            cfg, _ = column.resolve_coo_config(alg, dtype, caps, n=7)
            assert cfg["local_reduce"] == "classify"
            assert cfg["num_warps"] == 4
            for mode in ("classify", "segment", "none"):
                cfg, _ = column.resolve_coo_config(alg, dtype, caps, {"local_reduce": mode})
                assert cfg["local_reduce"] == mode
        for bad in ({"local_reduce": "auto"}, {"blocks_per_program": 2},
                    {"block_nnz": 16}, {"block_nnz": True}, {"prefetch_blocks": 2},
                    {"num_stages": 2}, {"num_warps": 3}):
            with unittest.TestCase().assertRaises(ValueError):
                column.resolve_coo_config(alg, "float32", caps, bad)
        for capability in ("scan", "reduction", "int64", "fp32_atomic"):
            with unittest.TestCase().assertRaises(csr.BackendCapabilityUnavailable):
                column.resolve_coo_config(alg, "float16", replace(caps, **{capability: None}))
        for capability in ("fp64", "fp64_atomic"):
            with unittest.TestCase().assertRaises(csr.BackendCapabilityUnavailable):
                column.resolve_coo_config(alg, "float64", replace(caps, **{capability: None}))
        column.resolve_coo_config(alg, "float32", replace(caps, scan=None, reduction=None), {"local_reduce": "none"})
        cfg, _ = column.resolve_coo_config(alg, "float32", replace(caps, legal_num_warps=(1, 2)))
        assert cfg["num_warps"] == 2
        cfg, _ = column.resolve_coo_config(alg, "float32", replace(caps, max_threads_per_block=64))
        assert cfg["num_warps"] == 2
        with unittest.TestCase().assertRaises(csr.BackendCapabilityUnavailable):
            column.resolve_coo_config(alg, "float32", replace(caps, max_threads_per_block=0))
    cfg, _ = column.resolve_coo_config("coo_runclass_panel_atomic", "float64", caps, n=7)
    assert cfg["block_n"] == 8 and cfg["conversion_mode"] == "load"
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_coo_config("coo_runclass_atomic", "float32", caps, {"conversion_mode": "load"})
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_coo_config("coo_runclass_panel_atomic", "float64", caps, {"block_n": 64})
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_coo_config("coo_segmented_atomic", "float32", caps, {"local_reduce": "classify"})
    with patch.dict(column.COO_ARCH_PROFILES, {("cuda", "80"): {
            "coo_runclass_panel_atomic": {"block_nnz": 16, "local_reduce": "invalid"}}}):
        cfg, info = column.resolve_coo_config("coo_runclass_panel_atomic", "float64", caps)
        assert cfg["block_nnz"] == 32 and cfg["local_reduce"] == "classify"
        assert len(info["rejected_profiles"]) == 2


def test_runclass_kernel_is_independent_and_has_no_device_plan():
    tree = ast.parse((SOURCE / "spmv_coo.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    kernel = functions["_coo_runclass_panel_kernel"]
    calls = {n.func.attr for n in ast.walk(kernel) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"sum", "associative_scan", "atomic_add"} <= calls
    assert not calls & {"argsort", "sort", "unique", "nonzero", "cumsum"}
    launch = ast.unparse(functions["_launch_coo_extension"])
    assert "_coo_runclass_panel_kernel[" in launch
    assert launch.index("_COO_RUNCLASS_ALGORITHMS") < launch.index("_COO_ATOMIC_ALGORITHMS")


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
        self.cuda = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True, scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True)
        self.hip = policy.BackendCaps(
            "rocm", "gfx90a", "hip", 64, 1024, True, True, True, scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True
        )

    def test_registry_support(self):
        self.assertEqual(set(policy.list_algorithms()), set(policy.ALGORITHMS))
        for alg in policy.NEW_ALGORITHMS:
            if alg in ("csr_nnz_tile_reduce", "csr_subgroup_pair_mixed"):
                self.assertNotIn(alg, policy.list_algorithms(dtype="float32"))
                self.assertIn(alg, policy.list_algorithms(dtype="bfloat16", backend="rocm"))
                continue
            self.assertEqual(alg in policy.list_algorithms("non", "torch.float32", "rocm"), "non" in policy.algorithm_spec(alg)["ops"])
            self.assertEqual(alg in policy.list_algorithms("trans", "float32", "cuda"),
                             "trans" in policy.algorithm_spec(alg)["ops"])
            self.assertEqual(alg in policy.list_algorithms("non", "complex64", "cuda"), "non" in policy.algorithm_spec(alg)["ops"])
            for backend in ("metax", "mthreads", "ascend"):
                self.assertNotIn(alg, policy.list_algorithms(backend=backend))
        with self.assertRaises(ValueError):
            policy.algorithm_spec("auto")
        with self.assertRaises(ValueError):
            policy.normalize_alg("compare")

    def test_all_dtype_ops_and_precision_requirements(self):
        no_fp64 = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, False, True, True, scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True)
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
            op = policy.algorithm_spec(alg)["ops"][0]
            for dtype in ("float16", "bfloat16", "complex64"):
                policy.validate_support(alg, op, dtype, "int64", "int32", no_fp64)
            for dtype in ("float32", "float64", "complex128"):
                with self.assertRaisesRegex(NotImplementedError, "FP64"):
                    policy.validate_support(
                        alg, op, dtype, "int64", "int32", no_fp64
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
        cfg, _ = column.resolve_csc_config(alg, "bfloat16" if alg in column.CSC_MIXED_ALGORITHMS else "complex128", 7, caps, op=ops[0],
                                             kind="spmv" if alg in ("csc_col_scalar_bound", "csc_col_scalar_tail_reduce") else "spmm")
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

    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True, scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True)
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


def test_csr_scalar_arguments_use_specialization_safe_casts():
    # Unity dimensions/strides can specialize to Python ints, not tl.tensor.
    cases = {
        "_spmv_csr_kernels.py": {"row_refill_kernel": {"XS", "YS"}},
        "_spmv_csr_transpose.py": {
            "_sharded_scatter": {"XS", "ZS", "N"},
            "_sharded_finish": {"YS", "N"},
        },
        "_spmm_csr_kernels.py": {"nnz_segment_panel_atomic_kernel": {"M"}},
    }
    for filename, functions in cases.items():
        tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
        for name, scalars in functions.items():
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for node in ast.walk(function):
                assert not (isinstance(node, ast.Attribute) and node.attr == "to"
                            and isinstance(node.value, ast.Name) and node.value.id in scalars), (filename, name)
            casted = {node.args[0].id for node in ast.walk(function)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl"
                      and node.func.attr == "cast" and node.args
                      and isinstance(node.args[0], ast.Name)}
            assert scalars <= casted, (filename, name)


def test_short_row_specs_and_configuration_isolation():
    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True)
    for alg, rows in (("csr_row_scalar_strip", 64), ("csr_row_static_tile", 8), ("csr_row_scalar_bound", 64), ("csr_row_scalar_tail_reduce", 64)):
        cfg, _ = policy.resolve_config(alg, caps)
        assert cfg["rows_per_program"] == rows
        assert cfg["num_warps"] == 2 and cfg["num_stages"] == 1
        spec = policy.algorithm_spec(alg)
        assert spec["ops"] == ("non",) and not spec["runtime_process_ops"]
        assert spec["compute_dtype_by_input"]["float32"] == "float64"
        for dtype in policy.VALUE_DTYPES:
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
        assert alg in policy.list_algorithms("non", "float16", "cuda")
        assert alg not in policy.list_algorithms("trans", "float16", "cuda")
        with unittest.TestCase().assertRaises(NotImplementedError):
            policy.validate_support(alg, "trans", "float16", "int32", "int32", caps)
        for invalid in ({"num_stages": 2}, {"num_warps": 3}, {"rows_per_program": True},
                        {"rows_per_program": 3}, {"output_shards": 2},
                        {"row_vector": {"block_nnz": 128}}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, invalid)
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.resolve_config(alg, policy.BackendCaps("unknown"))
    with unittest.TestCase().assertRaises(ValueError):
        policy.resolve_config("csr_row_scalar_strip", caps, {"block_nnz": 16})
    for k in (8, 16, 32, 64):
        cfg, _ = policy.resolve_config("csr_row_static_tile", caps, {"block_nnz": k})
        assert cfg["block_nnz"] == k


def test_longrow_queue_capacity_and_capabilities():
    from dataclasses import replace
    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True,
                              scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True)
    alg = "csr_longrow_queue_atomic"
    cfg, _ = policy.resolve_config(alg, caps)
    assert policy.algorithm_spec(alg)["ops"] == ("non",)
    for dtype in policy.VALUE_DTYPES:
        policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
    for field in ("scan", "int32_atomic", "fp64_atomic"):
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.validate_support(alg, "non", "float32", "int32", "int64", replace(caps, **{field: None}))
    for bad in ({"long_row_threshold": 3}, {"segment_nnz": 64}, {"num_stages": 2},
                {"worker_programs": 0}, {"worker_programs": 31}, {"rows_per_program": 32},
                {"workspace_limit_bytes": True}, {"output_shards": 2}):
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config(alg, caps, bad)
    for lengths in ([], [0, 1, 1024], [1025] * 17, [8193], [0, 1025, 8193, 1]):
        capacity = policy.longrow_queue_capacity(len(lengths), sum(lengths), cfg, 8, 2)
        long = [n for n in lengths if n > cfg["long_row_threshold"]]
        jobs = sum((n + cfg["segment_nnz"] - 1) // cfg["segment_nnz"] for n in long)
        assert len(long) <= capacity["long_row_capacity"] and jobs <= capacity["job_capacity"]
        assert capacity["workspace_peak_bytes"] == capacity["descriptor_bytes"] + capacity["numeric_workspace_bytes"]
    with unittest.TestCase().assertRaisesRegex(ValueError, "workspace"):
        policy.longrow_queue_capacity(1, 8193, dict(cfg, workspace_limit_bytes=1), 8, 2)
    with unittest.TestCase().assertRaisesRegex(ValueError, "int32"):
        policy.longrow_queue_capacity(1 << 31, 1 << 50, dict(cfg, workspace_limit_bytes=1 << 62), 4, 1)


def test_native_column_scalar_configuration():
    csr, column = _extension_policies()
    caps = csr.BackendCaps("cuda", "cuda", "80", 32, 1024, (1, 2, 4, 8),
                          True, True, True, True, True, True, True)
    for dtype in csr.DTYPES:
        for op in ("trans", "conj"):
            cfg, _ = column.resolve_csc_config("csc_col_scalar_bound", dtype, 1, caps, op=op, kind="spmv")
            assert cfg == dict(columns_per_program=64, num_warps=2, num_stages=1)
    for invalid in ({"block_nnz": 16}, {"columns_per_program": 16}, {"num_stages": 2}):
        with unittest.TestCase().assertRaises(ValueError):
            column.resolve_csc_config("csc_col_scalar_bound", "float32", 1, caps, invalid, op="trans", kind="spmv")
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_csc_config("csc_col_scalar_bound", "float32", 1, caps, op="non", kind="spmv")


def test_queue_has_ordered_launches_and_no_host_count_readback():
    tree = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    host = functions["run_longrow_queue"]
    calls = {n.func.attr for n in ast.walk(host) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"item", "tolist", "nonzero", "argsort", "cumsum", "repeat_interleave"}
    for name in ("row_scalar_bound_kernel",):
        assert not any(isinstance(n, ast.While) for n in ast.walk(functions[name]))
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "atomic_add" for n in ast.walk(functions[name]))
    consumer = functions["longrow_consumer_kernel"]
    assert "OWNERS" not in {arg.arg for arg in consumer.args.args}
    assert "row_scalar_bound_kernel" in functions


def test_scalar_tail_configuration_contract():
    caps = policy.BackendCaps("rocm", "gfx936", "hip", 64, 1024, True, True, True)
    for steps in (0, 4, 8, 16, 32):
        for width in (8, 16, 32, 64):
            cfg, _ = policy.resolve_config("csr_row_scalar_tail_reduce", caps,
                                           {"scalar_steps": steps, "block_nnz": width})
            assert cfg["scalar_steps"] == steps and cfg["block_nnz"] == width
    for invalid in ({"scalar_steps": -1}, {"scalar_steps": True}, {"scalar_steps": 3},
                    {"block_nnz": 128}, {"segment_nnz": 1024}):
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config("csr_row_scalar_tail_reduce", caps, invalid)
    with unittest.TestCase().assertRaises(ValueError):
        policy.resolve_config("csr_row_scalar_bound", caps, {"scalar_steps": 8})
    csr, column = _extension_policies()
    csc_caps = csr.BackendCaps("cuda", "cuda", "80", 32, 1024, (1, 2, 4, 8),
                              True, True, True, True, True, True, True)
    for dtype in csr.DTYPES:
        for op in ("trans", "conj"):
            cfg, _ = column.resolve_csc_config("csc_col_scalar_tail_reduce", dtype, 1, csc_caps,
                                               {"scalar_steps": 0}, op=op, kind="spmv")
            assert cfg["scalar_steps"] == 0 and cfg["columns_per_program"] == 64
    for invalid in ({"scalar_steps": True}, {"scalar_steps": 3}, {"block_nnz": 128}):
        with unittest.TestCase().assertRaises(ValueError):
            column.resolve_csc_config("csc_col_scalar_tail_reduce", "float32", 1, csc_caps,
                                      invalid, op="trans", kind="spmv")
    for op, kind in (("non", "spmv"), ("trans", "spmm")):
        with unittest.TestCase().assertRaises(ValueError):
            column.resolve_csc_config("csc_col_scalar_tail_reduce", "float32", 1, csc_caps, op=op, kind=kind)


def test_scalar_tail_kernels_keep_native_single_program_ownership():
    for filename, name in (("_spmv_csr_kernels.py", "row_scalar_tail_reduce_kernel"),
                           ("spmv_csc.py", "_csc_col_scalar_tail_reduce_kernel")):
        tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        calls = [n.func.attr for n in ast.walk(kernel) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert "atomic_add" not in calls and "cumsum" not in calls
        assert calls.count("max") == 1
        assert "sum" in calls


def test_column_partition_policy_and_workspace():
    csr, column = _extension_policies()
    caps = csr.BackendCaps("cuda", "cuda", "80", 32, 1024, (1, 2, 4, 8),
                          True, True, True, True, True, True, True)
    alg = "csc_col_partition_panel_atomic"
    for dtype in csr.DTYPES:
        cfg, _ = column.resolve_csc_config(alg, dtype, 33, caps)
        assert cfg["segments_per_column"] == 2 and cfg["output_shards"] == 1
        assert cfg["block_n"] == (8 if dtype == "complex128" else 16)
    for s in (1, 2, 4):
        for h in (1, 2, 4):
            cfg, _ = column.resolve_csc_config(alg, "float16", 7, caps,
                {"segments_per_column": s, "output_shards": h, "workspace_bytes": 101})
            assert cfg["workspace_bytes"] == 101 and cfg["block_n"] == 8
    for invalid in ({"segments_per_column": 3}, {"output_shards": 8},
                    {"workspace_bytes": 0}, {"workspace_bytes": True},
                    {"block_nnz": 8}, {"columns_per_program": 4}, {"num_stages": 2}):
        with unittest.TestCase().assertRaises(ValueError):
            column.resolve_csc_config(alg, "float16", 7, caps, invalid)
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_csc_config(alg, "float16", 7, caps, op="trans")
    with unittest.TestCase().assertRaises(ValueError):
        column.resolve_csc_config(alg, "float16", 7, caps, kind="spmv")
    from dataclasses import replace
    for capability in ("fp32_atomic", "reduction", "int64"):
        with unittest.TestCase().assertRaises(csr.BackendCapabilityUnavailable):
            column.resolve_csc_config(alg, "float16", 7, replace(caps, **{capability: None}))


def test_targeted_native_kernels_do_not_construct_structural_plans():
    tree = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    kernels = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("row_scalar_strip_kernel", "row_static_tile_kernel"):
        calls = {node.func.attr for node in ast.walk(kernels[name])
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert "store" in calls and "atomic_add" not in calls
    tree = ast.parse((SOURCE / "spmv_csc.py").read_text(encoding="utf-8"))
    launch = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_launch_csc_partition_panel")
    calls = {node.func.attr for node in ast.walk(launch)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls & {"item", "tolist", "cumsum", "repeat_interleave", "nonzero", "argsort"}
    # Test delivery remains the existing operator marker and registry-driven benchmark.
    for filename, names in (
        ("test_spmv_csr_accuracy.py", ("csr_row_scalar_strip", "csr_row_static_tile")),
        ("test_spmm_csc_accuracy.py", ("csc_col_partition_panel_atomic",)),
    ):
        source = (ROOT / "tests/pytest" / filename).read_text(encoding="utf-8")
        for name in names:
            assert name in source


def test_three_tier_profiles_capacity_and_capabilities():
    from dataclasses import replace
    alg = "csr_three_tier_queue_reduce"
    for backend, subgroup in (("cuda", 32), ("rocm", 64)):
        caps = policy.BackendCaps(backend, "unverified", backend, subgroup, 1024, True, True, True,
                                  scan=True, int32_atomic=True, fp32_atomic=False, fp64_atomic=False)
        cfg, source = policy.resolve_config(alg, caps)
        assert source == "three-tier-conservative-v1"
        assert cfg["short_row_threshold"] == 32 and cfg["long_row_threshold"] == 1024
        assert cfg["cooperative_parts"] == 4 and cfg["rows_per_program"] == 128
        assert policy.algorithm_spec(alg)["ops"] == ("non",)
        assert policy.algorithm_spec(alg)["cross_call_plan_cache"] is False
        for dtype in policy.VALUE_DTYPES:
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
            stats = policy.three_tier_queue_capacity(100, 2050, cfg, dtype)
            assert stats["medium_queue_capacity"] == 62 and stats["long_queue_capacity"] == 2
            assert stats["partial_slot_capacity"] == 8
            component = 8 if dtype in ("float32", "float64", "complex128") else 4
            components = 2 if dtype.startswith("complex") else 1
            assert stats["numeric_workspace_bytes"] == 8 * component * components
            assert stats["descriptor_peak_bytes"] == 8 + 8 * 64
            limit = stats["workspace_peak_bytes"]
            assert policy.three_tier_queue_capacity(100, 2050, dict(cfg, workspace_limit_bytes=limit), dtype) == stats
            with unittest.TestCase().assertRaisesRegex(ValueError, "workspace"):
                policy.three_tier_queue_capacity(100, 2050, dict(cfg, workspace_limit_bytes=limit - 1), dtype)
        assert policy.three_tier_queue_capacity(0, 0, cfg, "float16")["workspace_peak_bytes"] == 8
        with unittest.TestCase().assertRaisesRegex(ValueError, "int32"):
            policy.three_tier_queue_capacity(2**31, 2**31 * 1025, cfg, "float16")
        for op in ("trans", "conj"):
            assert alg not in policy.list_algorithms(op=op)
            with unittest.TestCase().assertRaises(NotImplementedError):
                policy.validate_support(alg, op, "float32", "int32", "int32", caps)
        for feature in ("scan", "int32_atomic", "reduction", "int64"):
            with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
                policy.validate_support(alg, "non", "float16", "int32", "int32", replace(caps, **{feature: None}))
        for bad in ({"long_row_threshold": 32}, {"short_row_threshold": -1}, {"cooperative_parts": 3},
                    {"long_grid_blocks": 0}, {"long_grid_blocks": True}, {"rows_per_program": 8},
                    {"num_warps": 3}, {"num_stages": 2}, {"grid_blocks": 128}, {"lanes_per_row": 8}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, bad)
        supplied = {"long_grid_blocks": 3, "long_row_threshold": 128, "cooperative_parts": 1}
        snapshot = dict(supplied)
        resolved, source = policy.resolve_config(alg, caps, supplied)
        assert resolved == dict(cfg, **supplied) and supplied == snapshot and source == "explicit"
        with unittest.TestCase().assertRaises(ValueError):
            policy.assert_route_match(alg, cfg, "csr_longrow_cooperative_reduce", None, caps)
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config("csr_longrow_cooperative_reduce", caps, {"long_grid_blocks": 128})


def test_three_tier_structure_model_and_launch_contract():
    tree = ast.parse((ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csr_tile_structure")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "three_tier_structure_policy", "exec"), namespace)
    stats = namespace["csr_tile_structure"]([0, 0, 2, 6, 11], config={
        "short_row_threshold": 2, "long_row_threshold": 4, "long_grid_blocks": 1,
        "cooperative_parts": 4, "block_nnz": 4, "producer_num_warps": 2, "rows_per_program": 32})
    assert "row_queue" not in stats and "cooperative_queue" not in stats
    q = stats["three_tier_queue"]
    assert (q["expected_short_rows"], q["expected_medium_rows"], q["expected_long_rows"]) == (2, 1, 1)
    assert q["expected_medium_nnz"] == 4 and q["expected_long_nnz"] == 5
    assert q["medium_queue_capacity"] == 3 and q["long_queue_capacity"] == 2
    assert q["expected_nonempty_part_tasks"] == 2 and q["expected_empty_part_tasks"] == 2
    kernels = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    functions = {n.name:n for n in kernels.body if isinstance(n, ast.FunctionDef)}
    runtime = ast.unparse(functions["run_three_tier_queue"])
    for name in ("three_tier_producer_kernel", "short_rowqueue_vector_kernel", "cooperative_row_consumer_kernel", "cooperative_row_finish_kernel"):
        assert name in runtime
    assert "run_longrow_cooperative(" not in runtime and "run_short_rowqueue(" not in runtime
    for path in (ROOT / "tests/test_spmv_csr.py", ROOT / "tests/benchmark_utils.py"):
        source = path.read_text(encoding="utf-8")
        for field in ("medium_queue_capacity", "long_queue_capacity", "medium_consumer_ms", "long_consumer_ms"):
            assert field in source


def test_three_tier_bounded_policy_and_schedule():
    from dataclasses import replace
    alg = "csr_three_tier_bounded_reduce"
    for backend, subgroup in (("cuda", 32), ("rocm", 64)):
        caps = policy.BackendCaps(backend, "unverified", backend, subgroup, 1024, True, True, True,
                                  scan=True, int32_atomic=True, fp32_atomic=False, fp64_atomic=False)
        cfg, source = policy.resolve_config(alg, caps)
        old, _ = policy.resolve_config("csr_three_tier_queue_reduce", caps)
        assert cfg == dict(old, medium_grid_blocks=512)
        assert source == "three-tier-bounded-conservative-v1"
        for dtype in policy.VALUE_DTYPES:
            assert alg in policy.list_algorithms(op="non", dtype=dtype, backend=backend)
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
            assert policy.three_tier_queue_capacity(100, 2050, cfg, dtype) == policy.three_tier_queue_capacity(100, 2050, old, dtype)
        for op in ("trans", "conj"):
            assert alg not in policy.list_algorithms(op=op)
            with unittest.TestCase().assertRaises(NotImplementedError):
                policy.validate_support(alg, op, "float16", "int32", "int32", caps)
        for feature in ("scan", "int32_atomic", "int64", "reduction"):
            with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
                policy.validate_support(alg, "non", "float16", "int32", "int32", replace(caps, **{feature: None}))
        for value in (0, -1, 1025, True, 1.5, "256"):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, {"medium_grid_blocks": value})
        for value in (1, 4, 256, 512, 1024):
            supplied = {"medium_grid_blocks": value}
            resolved, _ = policy.resolve_config(alg, caps, supplied)
            assert resolved == dict(cfg, **supplied) and supplied == {"medium_grid_blocks": value}
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config("csr_three_tier_queue_reduce", caps, {"medium_grid_blocks": 256})
        with unittest.TestCase().assertRaises(ValueError):
            policy.assert_route_match(alg, cfg, "csr_three_tier_queue_reduce", None, caps)
    tree = ast.parse((ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csr_tile_structure")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "bounded_structure_policy", "exec"), namespace)
    stats = namespace["csr_tile_structure"]([0, 0, 2, 6, 10, 14, 19], config={
        "short_row_threshold": 2, "long_row_threshold": 4, "long_grid_blocks": 1,
        "medium_grid_blocks": 2, "cooperative_parts": 4, "block_nnz": 4})["three_tier_queue"]
    assert stats["medium_queue_capacity"] == 6 and stats["medium_grid_blocks"] == 2
    assert stats["expected_medium_rows"] == 3 and stats["expected_medium_max_rounds"] == 2
    assert stats["medium_rounds_capacity"] == 3 and stats["expected_medium_empty_programs"] == 0
    functions = {n.name:n for n in ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    runtime = ast.unparse(functions["run_three_tier_bounded"])
    for name in ("three_tier_producer_kernel", "short_rowqueue_strided_vector_kernel", "cooperative_row_consumer_kernel", "cooperative_row_finish_kernel"):
        assert name in runtime
    assert "short_rowqueue_vector_kernel" not in runtime and "run_three_tier_queue(" not in runtime
    for path in (ROOT / "tests/test_spmv_csr.py", ROOT / "tests/benchmark_utils.py"):
        source = path.read_text(encoding="utf-8")
        assert all(field in source for field in ("medium_grid_limit", "medium_rounds_capacity"))


def test_three_tier_strided_out_validation_regression():
    """Validate both names through the real guard without initializing a GPU."""
    tree = ast.parse((SOURCE / "spmv_csr.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_spmv_check_output")
    class Tensor:
        device, dtype, shape = "cuda", "float32", (35,)
        def __init__(self, stride=3, pointer=100):
            self.step, self.pointer = stride, pointer
        def is_contiguous(self):
            return self.step == 1
        def stride(self, axis):
            return self.step
        def numel(self):
            return self.shape[0]
        def is_conj(self):
            return False
        def is_neg(self):
            return False
        def untyped_storage(self):
            return types.SimpleNamespace(data_ptr=lambda: self.pointer)
    namespace = dict(torch=types.SimpleNamespace(is_tensor=lambda v: isinstance(v, Tensor)),
                     _is_accel_tensor=lambda v: True, _csr_config=policy)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "strided_out_policy", "exec"), namespace)
    guard = namespace["_spmv_check_output"]
    for alg in policy.THREE_TIER_ALGORITHMS:
        prepared = types.SimpleNamespace(alg=alg, data=Tensor(pointer=1), kernel_indices=Tensor(pointer=2),
                                         kernel_indptr=Tensor(pointer=3), transpose=False, n_rows=35, n_cols=0)
        vector = Tensor(pointer=4)
        guard(Tensor(), prepared, vector)
        for stride in (0, -1):
            with unittest.TestCase().assertRaisesRegex(ValueError, "positive strides"):
                guard(Tensor(stride=stride), prepared, vector)
        with unittest.TestCase().assertRaisesRegex(ValueError, "overlap"):
            guard(Tensor(pointer=1), prepared, vector)
        with unittest.TestCase().assertRaisesRegex(ValueError, "shape/dtype"):
            guard(Tensor(), prepared, vector, output_dtype="float16")



def test_three_tier_ticket_configuration_capacity_and_overflow():
    from dataclasses import replace
    alg = "csr_three_tier_ticket_reduce"
    for backend, subgroup in (("cuda", 32), ("rocm", 64)):
        caps = policy.BackendCaps(backend, "unverified", backend, subgroup, 1024, True, True, True,
                                  scan=True, int32_atomic=True, fp32_atomic=False, fp64_atomic=False)
        old, _ = policy.resolve_config("csr_three_tier_bounded_reduce", caps)
        cfg, source = policy.resolve_config(alg, caps)
        assert cfg == dict(old, long_schedule="ticket", ticket_batch=1, long_worker_programs=0)
        assert source == "three-tier-ticket-conservative-v1"
        spec = policy.algorithm_spec(alg)
        assert spec["ops"] == ("non",) and not spec["cross_call_plan_cache"]
        assert alg in policy.list_algorithms("non", "float16", backend)
        assert alg not in policy.list_algorithms("trans", "float16", backend)
        for dtype in policy.VALUE_DTYPES:
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
            base = policy.three_tier_queue_capacity(100, 2050, old, dtype)
            stats = policy.ticket_queue_capacity(100, 2050, cfg, dtype)
            assert stats["workspace_peak_bytes"] == base["workspace_peak_bytes"] + 4
            assert stats["numeric_workspace_bytes"] == base["numeric_workspace_bytes"]
            assert stats["long_worker_programs"] == 8 and stats["task_capacity"] == 8
            assert stats["ticket_counter_max_bound"] == 0
            with unittest.TestCase().assertRaises(ValueError):
                policy.ticket_queue_capacity(100, 2050, dict(cfg, workspace_limit_bytes=stats["workspace_peak_bytes"] - 1), dtype)
            with unittest.TestCase().assertRaises(NotImplementedError):
                policy.validate_support(alg, "trans", dtype, "int32", "int32", caps)
        assert policy.ticket_queue_capacity(0, 0, cfg, "float16")["workspace_peak_bytes"] == 12
        for invalid in ({"ticket_batch": 2}, {"ticket_batch": True}, {"long_schedule": "dynamic"},
                        {"long_schedule": "static", "ticket_batch": 4}, {"long_worker_programs": 1},
                        {"long_worker_programs": True}, {"long_worker_programs": -32}, {"unknown": 1}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, invalid)
        for schedule, batch in (("static", 1), ("ticket", 1), ("ticket", 4)):
            explicit = dict(long_schedule=schedule, ticket_batch=batch, long_worker_programs=32)
            original = dict(explicit)
            resolved, source = policy.resolve_config(alg, caps, explicit)
            assert explicit == original and source == "explicit"
            assert resolved == dict(cfg, **explicit)
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config("csr_three_tier_bounded_reduce", caps, {"long_schedule": "ticket"})
        for feature in ("scan", "int32_atomic", "reduction", "int64"):
            with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
                policy.validate_support(alg, "non", "float16", "int32", "int32", replace(caps, **{feature: None}))
    # Terminal claims must fit too; T <= int32 max alone is insufficient for B4.
    assert policy.ticket_counter_bound(7936, 1024, 1) == 7936
    assert policy.ticket_counter_bound(7936, 1024, 4) == 11008
    assert policy.ticket_counter_bound(32, 32, 4) == 0
    assert policy.ticket_counter_bound(35, 32, 4) == 132
    assert policy.ticket_counter_bound(2**31 - 1, 32, 1) == 2**31 - 1
    assert policy.ticket_counter_bound(2**31 - 128, 32, 4) == 2**31 - 32
    for args in ((2**31 - 1, 32, 4), (1, 0, 1), (0, 1, 1), (32, 32, True)):
        with unittest.TestCase().assertRaises(ValueError):
            policy.ticket_counter_bound(*args)


def test_three_tier_ticket_structure_and_wiring():
    import numpy as np
    source = (ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "csr_tile_structure")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "ticket_structure_policy", "exec"), namespace)
    config = dict(short_row_threshold=8, long_row_threshold=64, block_nnz=64, cooperative_parts=1,
                  long_grid_blocks=2, medium_grid_blocks=4, long_schedule="ticket", ticket_batch=4, long_worker_programs=32)
    ptr = np.concatenate(([0], np.cumsum([257] * 35)))
    q = namespace["csr_tile_structure"](ptr, config=config)["three_tier_queue"]
    assert q["expected_long_tasks"] == 35 and q["long_consumer_grid_blocks"] == 32
    assert q["predicted_ticket_acquisitions"] == 33 and q["predicted_ticket_final_value"] == 132
    q = namespace["csr_tile_structure"](ptr, config=dict(config, long_schedule="static", ticket_batch=1))["three_tier_queue"]
    assert q["predicted_ticket_acquisitions"] == 0
    functions = {n.name: n for n in ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    runtime = ast.unparse(functions["run_three_tier_ticket"])
    for name in ("ticket_queue_capacity", "three_tier_producer_kernel", "short_rowqueue_strided_vector_kernel",
                 "three_tier_ticket_consumer_kernel", "cooperative_row_finish_kernel"):
        assert name in runtime
    assert "run_three_tier_bounded(" not in runtime and ".item(" not in runtime
    helper = ast.unparse(functions["_ticket_exclusive_part"])
    assert "atomic_add" not in helper
    consumer = ast.unparse(functions["three_tier_ticket_consumer_kernel"])
    assert "total > WORKERS" in consumer and "scope='gpu'" in consumer
    assert "tl.static_range(0, BATCH)" in consumer
    for path in (ROOT / "tests/test_spmv_csr.py", ROOT / "tests/benchmark_utils.py"):
        content = path.read_text(encoding="utf-8")
        assert all(field in content for field in ("ticket_counter_max_bound", "long_schedule", "ticket_batch", "long_worker_limit"))


def test_three_tier_fused_static_policy_and_wiring():
    from dataclasses import replace
    alg = "csr_three_tier_fused_static_reduce"
    for backend, subgroup in (("cuda", 32), ("rocm", 64)):
        caps = policy.BackendCaps(backend, "test", backend, subgroup, 1024, True, True, True,
                                  scan=True, fp32_atomic=True, fp64_atomic=True, int32_atomic=True)
        cfg, source = policy.resolve_config(alg, caps)
        old, _ = policy.resolve_config("csr_three_tier_bounded_reduce", caps)
        assert cfg == old and source == "three-tier-fused-static-conservative-v1"
        assert policy.algorithm_spec(alg)["ops"] == ("non",)
        for dtype in policy.VALUE_DTYPES:
            policy.validate_support(alg, "non", dtype, "int64", "int32", caps)
            base = policy.three_tier_queue_capacity(100, 2050, cfg, dtype)
            stats = policy.fused_static_queue_capacity(100, 2050, cfg, dtype)
            assert stats["workspace_peak_bytes"] == base["workspace_peak_bytes"] + 4
            assert stats["consumer_grid_blocks"] == min(base["medium_queue_capacity"], cfg["medium_grid_blocks"]) + stats["long_worker_programs"]
            with unittest.TestCase().assertRaises(ValueError):
                policy.fused_static_queue_capacity(100, 2050, dict(cfg, workspace_limit_bytes=stats["workspace_peak_bytes"]-1), dtype)
        assert policy.fused_static_queue_capacity(0, 0, cfg, "float16")["workspace_peak_bytes"] == 12
        for invalid in ({"long_schedule": "ticket"}, {"ticket_batch": 1}, {"long_worker_programs": 32},
                        {"medium_grid_blocks": 0}, {"cooperative_parts": 3}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, invalid)
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.resolve_config(alg, replace(caps, int32_atomic=None))
        with unittest.TestCase().assertRaises(NotImplementedError):
            policy.validate_support(alg, "trans", "float32", "int32", "int32", caps)
        with unittest.TestCase().assertRaises(ValueError):
            policy.assert_route_match(alg, cfg, "csr_three_tier_ticket_reduce", None, caps)
    tree = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    body = functions["three_tier_fused_static_consumer_kernel"]
    assert "slot += GM" in body and "task += WORKERS" in body
    assert "tl.num_programs" not in body and "atomic_add" not in body
    runtime = functions["run_three_tier_fused_static"]
    assert "three_tier_fused_static_consumer_kernel" in runtime and "cooperative_row_finish_kernel" in runtime
    for forbidden in ("short_rowqueue_strided_vector_kernel", "three_tier_ticket_consumer_kernel", ".item(", "run_three_tier_ticket("):
        assert forbidden not in runtime
    assert "medium_consumer_ms=None" in runtime and "long_consumer_ms=None" in runtime
    for path in (ROOT / "tests/test_spmv_csr.py", ROOT / "tests/benchmark_utils.py"):
        text = path.read_text(encoding="utf-8")
        assert "consumer_fused_ms" in text and "write_csr_task_diagnostics" in text


def test_three_tier_diagnostic_work_models_and_boundaries():
    import numpy as np
    tree = ast.parse((ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("csr_tile_structure", "csr_queue_work_model")]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "task_work_models", "exec"), namespace)
    ptr = np.concatenate(([0], np.cumsum([0, 8, 9, 64, 65, 129, 4097])))
    cfg = dict(short_row_threshold=8, long_row_threshold=64, block_nnz=64, cooperative_parts=8,
               medium_grid_blocks=4, long_grid_blocks=2)
    structure = namespace["csr_tile_structure"](ptr, config=cfg)["three_tier_queue"]["long_part_chunks"]
    assert structure["tasks"] == 24 and structure["total_chunks"] == 70
    assert structure["zero_tasks"] == 11 and structure["maximum"] == 9
    assert sum(int(k)*v for k,v in structure["histogram"].items()) == 70
    model = namespace["csr_queue_work_model"]
    for mapping in ("flat", "two_dimensional"):
        result = model(ptr, [3,2], [6,4,5], block=64, parts=8, medium_grid=4, workers=16, long_grid=2, mapping=mapping)
        assert result["task_count"] == 24 and result["total_long_chunks"] == 70
        assert sum(result["medium_worker_tasks"]) == 2 and sum(result["medium_worker_chunks"]) == 2
        assert sum(result["long_worker_initial_chunks"]) + sum(result["long_worker_remaining_chunks"]) == 70
        assert result["long_row_ids"] == [6,4,5]
    empty = model([0], [], [], block=64, parts=8, medium_grid=0, workers=0, long_grid=0)
    assert empty["task_count"] == 0 and empty["long_summary"]["max_over_mean"] is None
    for med, long, gm, workers in (([2],[4],0,16), ([2],[4],4,0), ([2],[2],4,16), ([9],[],4,0)):
        with unittest.TestCase().assertRaises(ValueError):
            model(ptr, med, long, block=64, parts=8, medium_grid=gm, workers=workers, long_grid=2)


def test_fused_diagnostics_are_separate_from_runtime_and_sampling():
    runtime = (SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8")
    functions = {n.name: ast.unparse(n) for n in ast.parse(runtime).body if isinstance(n, ast.FunctionDef)}
    for name, body in functions.items():
        if name.startswith("run_"):
            assert "three_tier_ticket_trace_kernel" not in body
    helper = (ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(helper)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    diagnostic = functions["collect_csr_task_diagnostics"]
    restores = [node for node in ast.walk(diagnostic) if isinstance(node, ast.Try) and node.finalbody]
    assert len(restores) == 1
    cleanup = ast.unparse(ast.Module(body=restores[0].finalbody, type_ignores=[]))
    assert "kernels.three_tier_producer_kernel = producer" in cleanup
    assert "kernels.three_tier_ticket_consumer_kernel = ticket" in cleanup
    assert "equal_nan=True" in ast.unparse(diagnostic)
    # There must be no diagnostic launch in an event/wall sampling loop.
    for path, function_name in ((ROOT / "tests/benchmark_utils.py", "run_type_combo_benchmark"),
                                (ROOT / "tests/test_spmv_csr.py", "main")):
        function = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(n, ast.FunctionDef) and n.name == function_name)
        for node in ast.walk(function):
            if isinstance(node, ast.For) and "args.iters" in ast.unparse(node.iter):
                assert "write_csr_task_diagnostics" not in ast.unparse(node)
    assert "--task-diagnostics" in helper and "--task-diagnostic-repeats" in helper
    public = (SOURCE / "spmv_csr.py").read_text(encoding="utf-8")
    assert '"csr_three_tier_fused_static_reduce"' in public
    for name in ("ops_support.py", "conf/operators.yaml", "tests/ci/test_gpu_ci_policy.py"):
        assert "csr_three_tier_fused_static_reduce" in (ROOT / name).read_text(encoding="utf-8")



def test_window_merge_profiles_and_actual_compute_capabilities():
    from dataclasses import replace
    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True,
                              scan=True, fp32_atomic=True, fp64_atomic=True)
    window, merge = policy.EXPERIMENT_ALGORITHMS
    for alg in (window, merge):
        assert policy.normalize_alg(alg) == alg
        spec = policy.algorithm_spec(alg)
        assert spec["ops"] == ("non", "trans", "conj")
        assert spec["value_dtypes"] == policy.VALUE_DTYPES
        assert spec["half_output_dtypes"]["float16"] == ("float16", "float32")
        assert spec["half_output_dtypes"]["bfloat16"] == ("bfloat16", "float32")
        for dtype in policy.VALUE_DTYPES:
            for op in spec["ops"]:
                policy.validate_support(alg, op, dtype, "int64", "int32", caps)
        for invalid in ({"num_stages": 2}, {"num_warps": 3}, {"num_warps": True}, {"unused": 1}):
            with unittest.TestCase().assertRaises(ValueError):
                policy.resolve_config(alg, caps, invalid)
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.validate_support(alg, "non", "float32", "int32", "int32", replace(caps, fp64=False))
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.resolve_config(alg, replace(caps, subgroup_width=0))
    wc, _ = policy.resolve_config(window, caps)
    mc, _ = policy.resolve_config(merge, caps)
    assert wc == dict(window_rows=64, rows_per_group=8, lanes_per_row=8, row_order="length_desc", num_warps=2, num_stages=1)
    assert mc == dict(strips_per_program=64, events_per_strip=4, workspace_limit_bytes=268435456, num_warps=4, num_stages=1)
    for invalid in ({"window_rows": 48}, {"lanes_per_row": 64}, {"row_order": "random"}, {"events_per_strip": 2}):
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config(window, caps, invalid)
    for invalid in ({"events_per_strip": 1}, {"strips_per_program": 16}, {"workspace_limit_bytes": 0}, {"row_order": "original"}):
        with unittest.TestCase().assertRaises(ValueError):
            policy.resolve_config(merge, caps, invalid)
    cfg, _ = policy.resolve_config(window, caps, dict(window_rows=16, rows_per_group=16))
    assert cfg["window_rows"] == cfg["rows_per_group"]
    for feature in ("scan", "fp32_atomic"):
        with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
            policy.validate_support(merge, "non", "float16", "int32", "int64", replace(caps, **{feature: None}))
    with unittest.TestCase().assertRaises(policy.BackendCapabilityUnavailable):
        policy.validate_support(merge, "non", "float32", "int32", "int32", replace(caps, fp64_atomic=None))
    policy.validate_support(window, "non", "float32", "int32", "int32", replace(caps, scan=False, fp64_atomic=False))
    with unittest.TestCase().assertRaises(ValueError):
        policy.assert_route_match(window, wc, merge, None, caps)
    for alg in (window, merge):
        tiny, _ = policy.resolve_config(alg, replace(caps, max_threads_per_block=32))
        assert tiny["num_warps"] == 1


def test_window_merge_independent_routes_and_locality_sampling_boundary():
    kernel_tree = ast.parse((SOURCE / "_spmv_csr_kernels.py").read_text(encoding="utf-8"))
    functions = {n.name: ast.unparse(n) for n in kernel_tree.body if isinstance(n, ast.FunctionDef)}
    runtime = functions["run_window_or_merge"]
    for expected in ("window_grouped_reduce_kernel", "merge_event_atomic_kernel", "workspace_limit_bytes"):
        assert expected in runtime
    for forbidden in ("build_plan(", ".item(", "run_nnz_local_reduce(", "run_three_tier", "torch.sort", "nonzero"):
        assert forbidden not in runtime
    assert "atomic" not in functions["window_grouped_reduce_kernel"]
    assert "_merge_event_corank" in functions["merge_event_atomic_kernel"]
    assert "_nnz_local_write" in functions["merge_event_atomic_kernel"]
    public = ast.parse((SOURCE / "spmv_csr.py").read_text(encoding="utf-8"))
    route = ast.unparse(next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == "_execute_spmv_route"))
    assert route.index("if needs_process:") < route.index("if alg in _csr_config.EXPERIMENT_ALGORITHMS:")
    assert "run_window_or_merge(execution, x, out, timing, _spmv_phase)" in route
    for path, fname in ((ROOT / "tests/test_spmv_csr.py", "main"),
                         (ROOT / "tests/benchmark_utils.py", "run_type_combo_benchmark")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname)
        assert "write_csr_locality_diagnostics" in ast.unparse(body)
        for node in ast.walk(body):
            if isinstance(node, ast.For) and "args.iters" in ast.unparse(node.iter):
                assert "write_csr_locality_diagnostics" not in ast.unparse(node)
    for name in ("ops_support.py", "conf/operators.yaml", "tests/ci/test_gpu_ci_policy.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert all(alg in text for alg in policy.EXPERIMENT_ALGORITHMS)


def test_csr_locality_models_preserve_duplicates_and_actual_adjacent_windows():
    import numpy as np
    tree = ast.parse((ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csr_locality_structure")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "csr_locality_model", "exec"), namespace)
    model = namespace["csr_locality_structure"]
    ptr, col = [0, 2, 2, 4, 5], [0, 1, 1, 2, 0]
    result = model(ptr, col, (4, 3), window_rows=2, rows_per_group=1, lanes_per_row=1)
    assert result["summary"]["total_nnz"] == 5 and result["summary"]["sampled_nnz"] == 5
    assert result["windows"][0]["adjacent_reuse_32"] is None
    assert result["windows"][1]["adjacent_reuse_32"] == 1.0
    assert result["summary"]["scheduled_original"] == result["summary"]["scheduled_grouped"] == 5
    trans = model(ptr, col, (4, 3), op="trans", window_rows=2, rows_per_group=1, lanes_per_row=1)
    assert trans["summary"]["total_nnz"] == 5 and [w["rows"] for w in trans["windows"]] == [2, 1]
    conj = model(ptr, col, (4, 3), op="conj", window_rows=2, rows_per_group=1, lanes_per_row=1)
    assert trans["windows"] == conj["windows"] and trans["supplemental_windows"] == conj["supplemental_windows"]
    # Alternating long/short rows reduce scheduled padded positions after local grouping.
    grouped = model([0, 32, 33, 65, 66], np.zeros(66, dtype=np.int64), (4, 1),
                    window_rows=4, rows_per_group=2, lanes_per_row=8)
    assert grouped["summary"]["scheduled_original"] == 128
    assert grouped["summary"]["scheduled_grouped"] == 80
    sample = model(ptr, col, (4, 3), window_rows=1, rows_per_group=1, lanes_per_row=1, max_windows=2)
    assert sample["summary"]["coverage"] == "deterministic_even_windows_not_unbiased"
    assert [w["window"] for w in sample["windows"]] == [0, 3]
    assert sample["windows"][1]["adjacent_reuse_32"] == 1.0
    empty = model([0], [], (0, 0))
    assert empty["summary"]["sampled_windows"] == 0 and empty["summary"]["grouped_position_ratio"] is None
    with unittest.TestCase().assertRaises(ValueError):
        model([0, 1], [2], (1, 2))
