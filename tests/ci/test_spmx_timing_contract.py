"""CPU regression tests for actual timing orchestration, without GPU imports."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "src/flagsparse/sparse_operations"


def load(path, name, **scope):
    tree = ast.parse(path.read_text(encoding="utf8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_cpu_gpu_samples_are_filtered_as_pairs():
    summary = load(ROOT / "tests/utils.py", "paired_timing_summary")
    result = summary([1, 2, 9, 100], [9, 8, 1, 1])
    assert result == dict(ms=10, process_cpu_ms=4, gpu_ms=6, sample_count=4, sample_kept=3)
    # Empty median window uses the two central *pairs*, not independent medians.
    result = summary([1, 1, 2, 100], [10, 20, 29, 1])
    assert result["ms"] == 26
    assert result["process_cpu_ms"] + result["gpu_ms"] == result["ms"]
    with pytest.raises(ValueError):
        summary([1], [])


def test_operator_samples_keep_metadata_from_each_invocation():
    paired = load(ROOT / "tests/utils.py", "paired_timing_summary")
    measure = load(ROOT / "tests/utils.py", "operator_benchmark_filtered", paired_timing_summary=paired)
    calls = []
    def run():
        calls.append(len(calls))
        cpu = float(len(calls))
        return "output", 10.0, dict(gpu_ms=10-cpu, process_cpu_ms=cpu)
    out, summary, last = measure(run, warmup=0, iters=3)
    assert len(calls) == 4  # mandatory unmeasured compile call
    assert out == "output" and summary["ms"] == 10
    assert summary["process_cpu_ms"] == 3 and summary["gpu_ms"] == 7
    assert last["process_cpu_ms"] == 4


def test_csr_transpose_does_not_reuse_a_numerical_snapshot():
    calls = []
    def materialize(data, indices, ptr, shape, op):
        calls.append(data[0])
        return data[0], indices, ptr, shape[::-1]
    make = load(OPS / "spmm_csr.py", "_materialize_spmm_csr_route_op",
                _normalize_spmm_op=lambda op: op,
                _materialize_spmm_csr_op=materialize,
                _spmm_csr_route_from_materialized=lambda p, *args: args[0])
    source = [2]
    prepared = SimpleNamespace(data=source, kernel_indices=object(), kernel_indptr=object(),
                               shape=(3, 5), materialized_route="stale", materialized_op="conj")
    assert make(prepared, "conj")[0] == 2
    source[0] = 9
    assert make(prepared, "conj")[0] == 9
    assert calls == [2, 9]
    assert make(prepared, "non")[0] is prepared


def test_coo_deferred_operation_uses_live_values_each_run():
    calls = []
    def materialize(data, row, col, shape, op):
        calls.append((data[0], op))
        return data[0], col, row, shape[::-1]
    def prepare(data, row, col, shape, op, alg, original_inputs):
        return SimpleNamespace(data=data, op=op, shape=shape)
    make = load(OPS / "spmm_coo.py", "_materialize_spmm_coo_route_op",
                _normalize_spmm_coo_op=lambda op: op,
                _materialize_spmm_coo_op=materialize, _prepare_spmm_coo_materialized_route=prepare)
    values = [2+3j]
    original = (values, [0], [1])
    prepared = SimpleNamespace(deferred_op=True, original_inputs=original, source_shape=(3, 5),
                               op="conj", alg="coo_rowrun")
    assert make(prepared).data == 2+3j
    values[0] = 7-1j
    assert make(prepared).data == 7-1j
    assert calls == [(2+3j, "conj"), (7-1j, "conj")]
    unchanged = SimpleNamespace(deferred_op=False)
    assert make(unchanged) is unchanged


@pytest.mark.parametrize("name,matrix,coo", [
    ("spmv_csr_mixed", False, False), ("spmv_coo_mixed", False, True),
    ("spmm_csr_mixed", True, False), ("spmm_coo_mixed", True, True)])
def test_mixed_timer_starts_before_input_conversion(name, matrix, coo):
    events = []
    class ConversionReached(Exception):
        pass
    class Tensor:
        dtype, device, ndim, shape = "float16", "cuda", 2 if matrix else 1, (3, 2)
        def __init__(self, size): self.size = size
        def numel(self): return self.size
        def resolve_conj(self): return self
        def resolve_neg(self): return self
        def contiguous(self):
            events.append("conversion")
            raise ConversionReached
    run = load(OPS / "mixed_spmx.py", name,
               _normalize_non_op=lambda *a: None, _check_1d=lambda *a: None,
               _resolve_types=lambda *a: ("float32", "float32"),
               _is_accel_tensor=lambda t: True, torch=SimpleNamespace(is_tensor=lambda t: True),
               _start=lambda enabled: events.append("start"))
    data, first, second, rhs = Tensor(4), Tensor(4), Tensor(4), Tensor(3)
    with pytest.raises(ConversionReached):
        run(data, first, second, rhs, (3, 3), return_time=True)
    assert events == ["start", "conversion"]


def test_phase_diagnostics_do_not_change_coo_primary_latency():
    durations = iter([7.0, 20.0, 30.0])
    def measure(op, warmup, iters):
        return op(), next(durations)
    timed = load(ROOT / "tests/test_spmv_coo.py", "_time_flagsparse_coo_row_run",
                 _cuda_event_benchmark=measure,
                 _build_flagsparse_coo_launch=lambda *args: "plan",
                 _run_flagsparse_coo_launch=lambda *args: "result")
    result = timed(None, None, None, None, (3, 4), "conj", 0, 1, timing=True)
    assert result["ms"] == result["gpu_ms"] == 7
    assert result["process_gpu_ms"] == 20 and result["compute_ms"] == 30


def test_native_coo_baseline_runs_operation_inside_events(monkeypatch):
    log = []
    class Event:
        def record(self): log.append("event")
        def synchronize(self): pass
    class Matrix:
        def sum_duplicates(self): pass
        def conj(self):
            log.append("conj")
            return self
    def spmv(matrix, vector, transa):
        assert transa
        log.append("multiply")
        return "answer"
    monkeypatch.setitem(sys.modules, "cupyx", SimpleNamespace(
        cusparse=SimpleNamespace(check_availability=lambda name: True, spmv=spmv)))
    cuda = SimpleNamespace(ExternalStream=lambda stream: nullcontext(), Event=Event,
                           get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
                           get_elapsed_time=lambda *args: 2.0)
    run = load(OPS / "_common.py", "_benchmark_spmv_coo_sparse_ref",
               _normalize_sparse_reference_op=lambda op: op,
               _spmv_coo_sparse_ref_backend=lambda *a, **k: ("cupy_cusparse", None),
               _ACCEL=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=0)),
               cp=SimpleNamespace(cuda=cuda),
               cpx_sparse=SimpleNamespace(coo_matrix=lambda *a, **k: Matrix()),
               _cupy_from_torch=lambda value: value, _torch_from_cupy=lambda value: value,
               _filtered_benchmark_avg_ms=lambda values: sum(values)/len(values))
    tensor = SimpleNamespace(dtype="complex64", is_complex=lambda: True)
    result = run(tensor, tensor, tensor, tensor, (3, 4), 0, 2, op="conj")
    assert result["ms"] == 2 and result["values"] == "answer"
    assert log == ["conj", "multiply"] + ["event", "conj", "multiply", "event"]*2


def test_dense_conjugate_operation_remains_a_lazy_view():
    calls = []
    class Tensor:
        ndim = 2
        def transpose(self, *axes): calls.append("transpose"); return self
        def is_complex(self): return True
        def conj(self): calls.append("conj_view"); return self
        def resolve_conj(self): raise AssertionError("materialized before operator timer")
    apply = load(OPS / "spmm_csr.py", "_apply_spmm_op_b",
                 torch=SimpleNamespace(is_tensor=lambda value: True))
    source = Tensor()
    assert apply(source, "conj") is source
    assert calls == ["transpose", "conj_view"]


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_coo_prepare_defers_requested_matrix_operation(monkeypatch, op):
    def remember(plan, sources):
        plan.original_inputs = tuple(sources)
        return plan
    monkeypatch.setitem(sys.modules, "_timing_policy.mixed_spmx", SimpleNamespace(
        remember_registered_inputs=remember))
    def constructor(data, row, col, shape, starts, lengths, output, compute, code, alg, **kwargs):
        return SimpleNamespace(shape=shape, op=code, **kwargs)
    def forbidden(*args):
        raise AssertionError("trans/conj materialized before run")
    prepare = load(OPS / "spmm_coo.py", "prepare_spmm_coo_route",
                   __package__="_timing_policy", _normalize_spmm_coo_alg=lambda alg: alg,
                   SPMM_COO_NEW_ALGORITHMS=(),
                   _normalize_spmm_coo_op=lambda code: code, _spmm_coo_op_to_name=lambda code: code,
                   _prepare_spmm_coo_matrix=lambda a, r, c, shape: (a, r, c, shape),
                   _materialize_spmm_coo_op=forbidden,
                   resolve_spmm_coo_algorithm=lambda *args: None,
                   _spmm_coo_compute_dtype=lambda dtype: "float32",
                   PreparedCooSpmmRoute=constructor,
                   torch=SimpleNamespace(int32="int32", float16="float16", bfloat16="bfloat16", empty=lambda *args, **kwargs: object()))
    values, row, col = SimpleNamespace(dtype="float16", device="gpu"), object(), object()
    plan = prepare(values, row, col, (3, 5), op=op, alg="auto")
    assert plan.deferred_op and plan.source_shape == (3, 5)
    assert plan.shape == ((3, 5) if op == "non" else (5, 3))
    assert plan.original_inputs == (values, row, col)


@pytest.mark.parametrize("dtype", ["float16", "complex64"])
def test_csc_native_and_mixed_run_measure_materialization_and_output(monkeypatch, dtype):
    log = []
    class Tensor:
        device = "gpu"
        def __init__(self): self.dtype = dtype
        def resolve_conj(self): log.append("resolve_conj"); return self
        def resolve_neg(self): log.append("resolve_neg"); return self
        def contiguous(self): log.append("contiguous"); return self
        def float(self): log.append("widen"); return self
        def to(self, dtype): log.append("output_cast"); return self
    class Event:
        def __init__(self, **kwargs): pass
        def record(self): log.append("event")
        def synchronize(self): pass
        def elapsed_time(self, other): return 3.0
    class Phases:
        def __init__(self, enabled): pass
        def measure(self, name): return nullcontext()
    accel = SimpleNamespace(Event=Event)
    measure = load(OPS / "spmv_coo.py", "_coo_measure_run", __package__="_timing_policy", _ACCEL=accel)
    monkeypatch.setitem(sys.modules, "_timing_policy._spmm_csr_runtime", SimpleNamespace(
        Phases=Phases, output_has_internal_overlap=lambda out: False))
    monkeypatch.setitem(sys.modules, "_timing_policy.spmv_coo", SimpleNamespace(_coo_measure_run=measure))
    monkeypatch.setitem(sys.modules, "_timing_policy.mixed_spmx", SimpleNamespace(
        validate_registered_output_alias=lambda *args: None))
    def kernel(*args): log.append("kernel"); return Tensor()
    def validate(x, prepared, code, *, materialize):
        assert not materialize
        return x
    run = load(OPS / "spmv_csc.py", "flagsparse_spmv_csc_run", __package__="_timing_policy",
               PreparedCscRoute=type("Registered", (), {}),
               _mixed_spmx=SimpleNamespace(resolve_registered_types=lambda *args, **kwargs: {"output_dtype": dtype}),
               torch=SimpleNamespace(float16="float16", bfloat16="bfloat16"),
               _spmv_csc_op_transposes=lambda code: code != 0,
               _validate_spmv_csc_x=validate, _run_spmv_csc_prepared_with_fallback=kernel,
               _spmv_csc_op_to_name=lambda code: "conj", _is_ascend_runtime=lambda: False)
    prepared = SimpleNamespace(data=Tensor(), op=2, n_rows=3, n_cols=5)
    out, elapsed, meta = run(prepared, Tensor(), return_time=True, return_meta=True)
    assert elapsed == meta["gpu_ms"] == meta["op_total_ms"] == 3
    assert log[0] == log[-1] == "event"
    assert log.index("resolve_conj") < log.index("kernel") < log.index("output_cast")


@pytest.mark.parametrize("return_time", [False, True])
def test_phase_instrumentation_cannot_change_primary_time(return_time):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_isolated_timing_contract", OPS / "_timing_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    @module.separate_timing_diagnostics
    def run(*, timing=False, return_time=False, return_meta=False):
        calls.append(timing)
        elapsed = 100.0 if timing else 7.0
        meta = dict(gpu_ms=elapsed, process_cpu_ms=0.0, operator_ms=elapsed,
                    process_gpu_ms=40.0 if timing else None, compute_ms=60.0 if timing else None)
        return ("answer", elapsed, meta) if return_time else ("answer", meta)
    result = run(timing=True, return_meta=True, return_time=return_time)
    meta = result[-1]
    assert calls == [False, True]
    assert meta["gpu_ms"] == meta["operator_ms"] == meta["op_total_ms"] == 7
    assert meta["process_gpu_ms"] == 40 and meta["compute_ms"] == 60
    assert meta["phase_diagnostics"] == "separately_measured"
    if return_time: assert result[1] == 7
