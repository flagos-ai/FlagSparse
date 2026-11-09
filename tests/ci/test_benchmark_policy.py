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

"""Checks for the benchmark framework policy."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
BENCHMARK_CONFTEST = BENCHMARK_DIR / "conftest.py"


def _read(path):
    return (BENCHMARK_DIR / path).read_text(encoding="utf-8")


def test_performance_utils_defines_required_metrics_and_flow():
    text = _read("performance_utils.py")
    for snippet in [
        "DEFAULT_METRICS",
        '"latency_base"',
        '"latency"',
        '"speedup"',
        "statistics.median",
        "synchronize",
        "two_level_average_speedup",
        "warmup",
        "iterations",
    ]:
        assert snippet in text


def test_benchmark_shapes_are_centralized():
    assert (BENCHMARK_DIR / "attri_util.py").is_file()
    assert (BENCHMARK_DIR / "core_shapes.yaml").is_file()


def test_summary_for_plot_entrypoint_exists():
    text = _read("summary_for_plot.py")
    assert "two_level_average_speedup" in text
    assert "result_file" in text


def test_benchmark_pytest_json_record_plugin_exists():
    text = BENCHMARK_CONFTEST.read_text(encoding="utf-8")
    for snippet in [
        '"--record"',
        '"--output"',
        "def update_result",
        "RECORD_JSON",
        "pytest_terminal_summary",
        "benchmark_result.json",
    ]:
        assert snippet in text


def test_spgemm_csv_is_written_incrementally():
    # A rocSPARSE csrgemm page fault on DCU aborts the process with SIGABRT,
    # which no Python handler can intercept.  SpGEMM used to buffer every row
    # until the sweep ended, so the crash left an empty CSV and discarded the
    # matrices that had already succeeded.  Rows must reach disk per matrix.
    source = (PROJECT_ROOT / "tests" / "test_spgemm.py").read_text(encoding="utf-8")
    export = source.split("def run_all_dtypes_export_csv(", 1)[1].split("\ndef ", 1)[0]
    assert "on_entry=_emit" in export, "per-matrix CSV callback is not wired up"
    assert "handle.flush()" in source and "os.fsync(handle.fileno())" in source
    # the batch must not be buffered into a list that is only written at the end
    assert "rows.append(" not in export


def test_spgemm_records_the_item_in_flight_for_crash_attribution():
    source = (PROJECT_ROOT / "tests" / "test_spgemm.py").read_text(encoding="utf-8")
    assert "def _record_spgemm_inflight(" in source
    assert "def _clear_spgemm_inflight(" in source
    export = source.split("def run_all_dtypes_export_csv(", 1)[1].split("\ndef ", 1)[0]
    # written before the work starts, cleared only after a clean finish
    assert "_record_spgemm_inflight(" in export
    assert "_clear_spgemm_inflight(csv_path)" in export


def _timing_function(path, name, namespace=None):
    """Load a pure timing helper without importing GPU operator modules."""
    import ast

    source = path.read_text(encoding="utf-8")
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_filtered_timing_statistics_match_all_entry_points():
    helpers = [
        (PROJECT_ROOT / "tests/utils.py", "filtered_avg_ms"),
        (PROJECT_ROOT / "src/flagsparse/sparse_operations/_common.py", "_filtered_benchmark_avg_ms"),
        (PROJECT_ROOT / "src/flagsparse/sparse_operations/_spmv_csr_benchmark.py", "_filtered_avg_ms"),
    ]
    cases = [([], None), ([3.0], 3.0), ([0.0, 0.0, 10.0], 0.0),
             ([0.01, 1.0, 1.0, 1.0, 100.0], 1.0), ([1.0, 100.0], 50.5)]
    for path, name in helpers:
        reduce_samples = _timing_function(path, name)
        for samples, expected in cases:
            original = list(samples)
            assert reduce_samples(samples) == expected
            assert samples == original


def test_prepared_filtered_timing_keeps_one_descriptor_lifetime():
    from types import SimpleNamespace

    durations = iter([1.0, 1.0, 100.0])
    calls = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return next(durations)

    common = PROJECT_ROOT / "src/flagsparse/sparse_operations/_common.py"
    measure = _timing_function(common, "_benchmark_prepared_cuda_op", {
        "_ACCEL": SimpleNamespace(Event=Event, synchronize=lambda: None),
        "_is_rocm_runtime": lambda: False,
        "_destroy_hip_event": lambda event: None,
        "_filtered_benchmark_avg_ms": _timing_function(common, "_filtered_benchmark_avg_ms"),
    })
    state = object()

    def prepare():
        calls.append("prepare")
        return state

    def run(value):
        assert value is state
        calls.append("run")
        return value

    def destroy(value):
        assert value is state
        calls.append("destroy")

    result, elapsed = measure(prepare, run, destroy, warmup=1, iters=3, filter_samples=True)
    assert result is state
    assert elapsed == 1.0
    assert calls == ["prepare", "run", "run", "run", "run", "destroy"]


def test_spmx_filtered_measurement_preserves_optional_shared_default():
    import ast

    common = ast.parse((PROJECT_ROOT / "src/flagsparse/sparse_operations/_common.py").read_text(encoding="utf-8"))
    for name in ("_benchmark_cuda_op", "_benchmark_prepared_cuda_op"):
        function = next(node for node in common.body if isinstance(node, ast.FunctionDef) and node.name == name)
        position = [arg.arg for arg in function.args.kwonlyargs].index("filter_samples")
        assert ast.literal_eval(function.args.kw_defaults[position]) is False
    source = (PROJECT_ROOT / "tests/test_spmm.py").read_text(encoding="utf-8")
    assert "_spmm_csr_auto_prefers_alg1(data, indptr, shape)" in source
    assert "filter_samples=True" in source
    assert "first_call_ms = (time.perf_counter() - t0) * 1000.0" in source


def test_sparse_work_metrics_use_complete_time_and_explicit_byte_model():
    import math
    import unittest
    work = _timing_function(PROJECT_ROOT / "tests/utils.py", "sparse_work_metrics")
    arguments = dict(operator="spmm_csr", shape=(3, 4), nnz=5, dense_cols=7,
                     a_dtype="float16", rhs_dtype="float16", output_dtype="float32",
                     index_dtype="int32", indptr_dtype="int64", op="non", ms=2.0, gpu_ms=1.0)
    result = work(**arguments)
    assert result["useful_flops"] == 70
    assert result["sparse_input_bytes"] == 62
    assert result["rhs_gather_bytes"] == 70
    assert result["output_write_bytes"] == 84
    assert result["logical_bytes"] == 216
    assert math.isclose(result["useful_gflops"], 70 / 2e6)
    assert math.isclose(result["logical_gbs"], 216 / 2e6)
    assert math.isclose(result["gnnz_s"], 5 / 2e6)
    assert result["gpu_useful_gflops"] == 2 * result["useful_gflops"]
    assert work(**dict(arguments, operator="spmm_coo"))["logical_bytes"] == 224
    assert work(**dict(arguments, operator="spmm_csc", op="trans"))["logical_bytes"] == 252
    assert work(**dict(arguments, operator="spmv_csr"))["useful_flops"] == 10
    assert work(**dict(arguments, output_dtype="float16"))["logical_bytes"] == 174
    for dtype in ("float16", "bfloat16"):
        assert work(**dict(arguments, a_dtype=dtype, rhs_dtype=dtype))["useful_flops"] == 70
    for elapsed in (None, 0.0, -1.0, float("inf"), float("nan")):
        row = work(**dict(arguments, ms=elapsed))
        assert row["logical_gbs"] is None and row["metrics_reason"]
    assert work(**dict(arguments, dense_cols=0))["logical_bytes"] == 0
    assert work(**dict(arguments, nnz=0))["useful_gflops"] == 0
    with unittest.TestCase().assertRaises(ValueError):
        work(**dict(arguments, a_dtype="complex64"))


def test_metric_sample_summary_does_not_modify_filter_rule():
    summary = _timing_function(PROJECT_ROOT / "tests/utils.py", "benchmark_sample_summary")
    result = summary([0.01, 1.0, 1.0, 1.0, 100.0])
    assert result["sample_count"] == 5 and result["sample_kept"] == 3
    assert result["sample_median_ms"] == 1.0
    assert summary([1.0, 100.0])["sample_kept"] == 0  # filtered_avg_ms falls back to median
    assert summary([])["sample_median_ms"] is None
    assert summary([0.0, 0.0])["sample_cv"] == 0.0


def test_mixed_metric_cli_is_shared_by_all_six_benchmarks():
    import argparse
    path = PROJECT_ROOT / "tests/benchmark_utils.py"
    add = _timing_function(path, "add_type_combo_arguments")
    parser = argparse.ArgumentParser()
    add(parser, spmm=True)
    assert parser.parse_args([]).perf_metrics is False
    assert parser.parse_args(["--perf-metrics"]).perf_metrics is True
    for kind in ("spmv", "spmm"):
        for fmt in ("csr", "coo", "csc"):
            source = (PROJECT_ROOT / f"tests/test_{kind}_{fmt}.py").read_text(encoding="utf-8")
            assert "add_type_combo_arguments(" in source and "run_type_combo_benchmark(" in source
    runner = (PROJECT_ROOT / "run_flagsparse_pytest.py").read_text(encoding="utf-8")
    assert '"--type-combos"' in runner and '"tests/test_spmm_csr.py"' in runner
    source = path.read_text(encoding="utf-8")
    assert "--perf-metrics currently requires --type-combos" in source
    assert 'row["ms"] = row["gpu_ms"] + row["process_cpu_ms"]' in source
    assert "benchmark_sample_summary(gpu)" in source
    assert "NOT DRAM/HBM" in source


def test_mixed_benchmark_run_signatures_preserve_type_and_operation_contract():
    import ast

    for kind in ("spmv", "spmm"):
        for fmt in ("csr", "coo", "csc"):
            op = f"{kind}_{fmt}"
            path = PROJECT_ROOT / "src/flagsparse/sparse_operations" / f"{op}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            for name in (f"flagsparse_{op}", f"flagsparse_{op}_run", f"list_{op}_algorithms"):
                function = functions[name]
                arguments = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
                assert "out_dtype" in arguments, name
                if name.startswith("list_"):
                    assert "rhs_dtype" in arguments, name
                else:
                    assert {"out", "return_time", "return_meta"} <= arguments, name
                    if kind == "spmm":
                        assert "op_b" in arguments, name
            run = functions[f"flagsparse_{op}_run"]
            if kind == "spmm":
                applications = [node for node in ast.walk(run) if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and node.func.attr == "apply_registered_op_b"]
                assert len(applications) == 1, op


def test_sparse_reference_normalization_is_not_clamped_utilization():
    import unittest

    normalize = _timing_function(PROJECT_ROOT / "tests/utils.py", "sparse_reference_metrics")
    metrics = dict(logical_gbs=200.0, gpu_logical_gbs=400.0,
                   arithmetic_intensity=2.0, gpu_useful_gflops=800.0)
    result = normalize(metrics, reference_gbs=100.0, bandwidth_source="BabelStream Triad FP32 test",
                       peak_gflops=1000.0, peak_source="FP32 scalar FMA, test reference")
    assert result["logical_bw_ratio"] == 2.0
    assert result["gpu_logical_bw_ratio"] == 4.0
    assert result["logical_roofline_gflops"] == 200.0
    assert result["gpu_logical_roofline_ratio"] == 4.0  # logical model can exceed one
    assert normalize(metrics)["logical_bw_ratio"] is None
    assert normalize({}, reference_gbs=100.0, bandwidth_source="test")["logical_bw_ratio"] is None
    for options in (dict(reference_gbs=0.0, bandwidth_source="test"),
                    dict(reference_gbs=float("nan"), bandwidth_source="test"),
                    dict(reference_gbs=100.0), dict(peak_gflops=1000.0, peak_source="test"),
                    dict(bandwidth_source="test")):
        with unittest.TestCase().assertRaises(ValueError):
            normalize(metrics, **options)


def test_mixed_wall_measurement_is_separate_from_event_sum():
    source = (PROJECT_ROOT / "tests/benchmark_utils.py").read_text(encoding="utf-8")
    assert 'row["ms"] = row["gpu_ms"] + row["process_cpu_ms"]' in source
    assert 'row["prepared_run_ms"] = filtered_avg_ms(wall_samples)' in source
    assert 'wall_kwargs = dict(out_dtype=output_dtype)' in source
    assert 'wall_result = run(prepared, rhs, **wall_kwargs)' in source
    assert 'row["cpu_gpu_overlap_risk"] = any(value > 0 for value in cpu)' in source
    assert 'reference bandwidth/peak options require --perf-metrics' in source
