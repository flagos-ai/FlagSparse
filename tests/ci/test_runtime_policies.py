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

"""CPU-only regression checks for pure runtime policy helpers."""

import os

import pytest

if os.environ.get("FLAGSPARSE_TRITON_SMOKE") != "1":
    pytest.skip(
        "triton smoke is opt-in and excluded from CPU-only CI", allow_module_level=True
    )

torch = pytest.importorskip("torch")

# fmt: off
# isort: off
from flagsparse.sparse_operations import _common  # noqa: E402
from flagsparse.sparse_operations import (  # noqa: E402
    gather_scatter as gather_scatter_ops,
)
from flagsparse.sparse_operations import spmv_coo as spmv_coo_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_bsr as spmv_bsr_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csc as spmv_csc_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csr as spmv_csr_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_bell as spmm_bell_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_bsr as spmm_bsr_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_csc as spmm_csc_ops  # noqa: E402
# isort: on
# fmt: on


def test_scatter_dtype_policy_complex64_resolves_directly():
    value_dtype, fell_back, reason = _common._resolve_scatter_value_dtype(
        "complex64", dtype_policy="auto"
    )
    assert value_dtype == torch.complex64
    assert fell_back is False
    assert reason is None


def test_scatter_dtype_policy_complex128_resolves_directly():
    value_dtype, fell_back, reason = _common._resolve_scatter_value_dtype(
        "complex128", dtype_policy="strict"
    )
    assert value_dtype == torch.complex128
    assert fell_back is False
    assert reason is None


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_index_fallback_policy_normalization(policy):
    assert spmv_csr_ops._normalize_spmv_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_coo_index_fallback_policy_normalization(policy):
    assert spmv_coo_ops._normalize_spmv_coo_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_csc_index_fallback_policy_normalization(policy):
    assert spmv_csc_ops._normalize_spmv_csc_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_bsr_index_fallback_policy_normalization(policy):
    assert spmv_bsr_ops._normalize_spmv_bsr_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmm_bsr_index_fallback_policy_normalization(policy):
    assert spmm_bsr_ops._normalize_spmm_bsr_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmm_csc_index_fallback_policy_normalization(policy):
    assert spmm_csc_ops._normalize_spmm_csc_index_fallback_policy(policy) == policy


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmv_coo_op_normalization(op, expected):
    assert spmv_coo_ops._normalize_spmv_coo_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmv_csc_op_normalization(op, expected):
    assert spmv_csc_ops._normalize_spmv_csc_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmv_bsr_op_normalization(op, expected):
    assert spmv_bsr_ops._normalize_spmv_bsr_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmm_bsr_op_normalization(op, expected):
    assert spmm_bsr_ops._normalize_spmm_bsr_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmm_csc_op_normalization(op, expected):
    assert spmm_csc_ops._normalize_spmm_csc_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmm_bell_op_normalization(op, expected):
    assert spmm_bell_ops._normalize_spmm_bell_op(op) == expected


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_csr_op_transpose_contract(op):
    if op == "non":
        assert (
            spmv_csr_ops._spmv_op_transposes(spmv_csr_ops._normalize_spmv_op(op))
            is False
        )
    else:
        assert (
            spmv_csr_ops._spmv_op_transposes(spmv_csr_ops._normalize_spmv_op(op))
            is True
        )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_csc_op_transpose_contract(op):
    if op == "non":
        assert (
            spmv_csc_ops._spmv_csc_op_transposes(
                spmv_csc_ops._normalize_spmv_csc_op(op)
            )
            is False
        )
    else:
        assert (
            spmv_csc_ops._spmv_csc_op_transposes(
                spmv_csc_ops._normalize_spmv_csc_op(op)
            )
            is True
        )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_bsr_op_transpose_contract(op):
    if op == "non":
        assert (
            spmv_bsr_ops._spmv_bsr_op_transposes(
                spmv_bsr_ops._normalize_spmv_bsr_op(op)
            )
            is False
        )
    else:
        assert (
            spmv_bsr_ops._spmv_bsr_op_transposes(
                spmv_bsr_ops._normalize_spmv_bsr_op(op)
            )
            is True
        )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_bsr_supported_ops_accepted_by_policy(op):
    assert (
        spmv_bsr_ops._ensure_spmv_bsr_supported_op(
            spmv_bsr_ops._normalize_spmv_bsr_op(op)
        )
        is None
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_bsr_supported_ops_accepted_by_policy(op):
    assert (
        spmm_bsr_ops._ensure_spmm_bsr_supported_op(
            spmm_bsr_ops._normalize_spmm_bsr_op(op)
        )
        is None
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_csc_op_transpose_contract(op):
    if op == "non":
        assert (
            spmm_csc_ops._spmm_csc_op_transposes(
                spmm_csc_ops._normalize_spmm_csc_op(op)
            )
            is False
        )
    else:
        assert (
            spmm_csc_ops._spmm_csc_op_transposes(
                spmm_csc_ops._normalize_spmm_csc_op(op)
            )
            is True
        )


@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmm_bell_op_transpose_contract(op):
    assert (
        spmm_bell_ops._spmm_bell_op_transposes(
            spmm_bell_ops._normalize_spmm_bell_op(op)
        )
        is True
    )


def test_spmm_bell_non_op_transpose_contract():
    assert (
        spmm_bell_ops._spmm_bell_op_transposes(
            spmm_bell_ops._normalize_spmm_bell_op("non")
        )
        is False
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_csc_supported_ops_accepted_by_policy(op):
    assert (
        spmm_csc_ops._ensure_spmm_csc_supported_op(
            spmm_csc_ops._normalize_spmm_csc_op(op)
        )
        is None
    )


def test_spmm_bell_supported_non_op_accepted_by_policy():
    assert (
        spmm_bell_ops._ensure_spmm_bell_supported_op(
            spmm_bell_ops._normalize_spmm_bell_op("non")
        )
        is None
    )


@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmm_bell_reserved_ops_rejected_by_policy(op):
    with pytest.raises(ValueError, match="reserved"):
        spmm_bell_ops._ensure_spmm_bell_supported_op(
            spmm_bell_ops._normalize_spmm_bell_op(op)
        )


@pytest.mark.parametrize("alg", ["auto", "base", "spmm_bsr_base"])
def test_spmm_bsr_algorithm_normalization(alg):
    normalized = spmm_bsr_ops._normalize_spmm_bsr_alg(alg)
    assert normalized in ("auto", "spmm_bsr_base")


@pytest.mark.parametrize("alg", ["auto", "base", "spmm_csc_base"])
def test_spmm_csc_algorithm_normalization(alg):
    normalized = spmm_csc_ops._normalize_spmm_csc_alg(alg)
    assert normalized in ("auto", "spmm_csc_base")


@pytest.mark.parametrize("alg", ["auto", "base", "spmm_bell_base"])
def test_spmm_bell_algorithm_normalization(alg):
    normalized = spmm_bell_ops._normalize_spmm_bell_alg(alg)
    assert normalized in ("auto", "spmm_bell_base")


def test_scatter_policy_validator_rejects_unknown_policy():
    with pytest.raises(
        ValueError, match="index_fallback_policy must be 'auto' or 'strict'"
    ):
        gather_scatter_ops._triton_scatter_impl(
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.int64),
            1,
            index_fallback_policy="invalid",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wide", [False, True])
def test_registered_half_type_resolution(dtype, wide):
    from types import SimpleNamespace
    from flagsparse.sparse_operations.mixed_spmx import resolve_registered_types

    output_dtype = torch.float32 if wide else dtype
    out = SimpleNamespace(dtype=output_dtype)
    result = resolve_registered_types(dtype, dtype, out=out, out_dtype=output_dtype)
    assert result == dict(a_dtype=dtype, rhs_dtype=dtype, compute_dtype=torch.float32,
                          output_dtype=output_dtype)
    with pytest.raises(ValueError, match="conflicts"):
        resolve_registered_types(dtype, dtype, out=out,
                                 out_dtype=dtype if wide else torch.float32)
    with pytest.raises(TypeError, match="matching"):
        resolve_registered_types(dtype, torch.float32, out_dtype=output_dtype)
    with pytest.raises(TypeError, match="unsupported"):
        resolve_registered_types(dtype, dtype, out_dtype=torch.float64)


@pytest.mark.parametrize("operator,algorithm,op", [
    ("spmv_csr", "legacy_rowpar", "non"),
    ("spmv_csr", "legacy_segbin", "trans"),
    ("spmv_csr", "row_subgroup_stream", "conj"),
    ("spmv_coo", "coo_segmented_stream_atomic", "non"),
    ("spmv_csc", "csc_col_vector", "trans"),
    ("spmm_csr", "csr_row_panel", "conj"),
    ("spmm_csr", "csr_base_accuracy", "trans"),
    ("spmm_csr", "spmm_csr_alg2_accuracy_hp", "non"),
    ("spmm_coo", "spmm_coo_alg1", "trans"),
    ("spmm_coo", "coo_segmented_stream_panel_atomic", "non"),
    ("spmm_csc", "csc_col_kparallel", "trans"),
    ("spmm_csc", "spmm_csc_base", "conj"),
    ("spmv_csc", "spmv_csc_base", "non"),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_registered_half_algorithm_query(operator, algorithm, op, dtype):
    import importlib

    module = importlib.import_module(f"flagsparse.sparse_operations.{operator}")
    query = getattr(module, f"list_{operator}_algorithms")
    assert algorithm in query(op=op, dtype=dtype, rhs_dtype=dtype, out_dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype is required"):
        query(rhs_dtype=dtype, out_dtype=torch.float32)


def test_generic_baseline_binds_only_requested_symbols():
    from types import SimpleNamespace
    from tests.cusparse_generic_baseline import _GenericLibrary, _SIGNATURES

    class Function:
        def __call__(self, *args):
            return 0

    create = Function()
    spmv = Function()
    library = _GenericLibrary(SimpleNamespace(cusparseCreate=create, cusparseSpMV=spmv))
    assert library.cusparseCreate is create
    assert library.cusparseSpMV is spmv
    assert spmv.argtypes == _SIGNATURES["cusparseSpMV"]
    with pytest.raises(NotImplementedError, match="cusparseCreateSlicedEll"):
        library.cusparseCreateSlicedEll
    # Missing SELL does not invalidate already bound generic multiplication.
    assert library.cusparseSpMV is spmv


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_registered_type_combo_cli_contract(storage):
    import argparse
    from tests.benchmark_utils import add_type_combo_arguments, parse_type_combos

    value = f"{storage}:{storage}:{storage},{storage}:{storage}:float32"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--dense-cols", type=int, default=32)
    add_type_combo_arguments(parser, spmm=True)
    args = parser.parse_args(["--type-combos", value, "--op-b", "all", "--dense-cols", "1,7,33"])
    assert args.dtype == "float32"  # implicit defaults must not conflict
    assert args.dense_cols == "1,7,33"
    assert parse_type_combos(args.type_combos, ["--type-combos", value]) == (
        (storage, storage, storage), (storage, storage, "float32")
    )
    for token in ("--dtype=float16", "--dtypes=bfloat16"):
        with pytest.raises(ValueError, match="conflicts"):
            parse_type_combos(value, [token])
    with pytest.raises(ValueError, match="matching inputs"):
        parse_type_combos("float16:bfloat16:float32", [])


def test_registered_original_alias_survives_preparation():
    from types import SimpleNamespace
    from flagsparse.sparse_operations.mixed_spmx import (
        remember_registered_inputs, validate_registered_output_alias,
    )

    values = torch.ones(4, dtype=torch.float16)
    prepared = remember_registered_inputs(SimpleNamespace(data=values.float()), (values,))
    with pytest.raises(ValueError, match="original input"):
        validate_registered_output_alias(prepared, torch.ones(4), values[::2])
    validate_registered_output_alias(prepared, torch.ones(4), torch.empty(4))


def test_generic_plan_close_is_idempotent():
    from tests.cusparse_generic_baseline import _Plan

    released = []
    plan = _Plan(lambda: None, (), [(released.append, "descriptor")])
    plan.close()
    plan.close()
    assert released == ["descriptor"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("op_a", ["non", "trans", "conj"])
@pytest.mark.parametrize("op_b", ["non", "trans", "conj"])
def test_vendor_real_operation_mapping(dtype, op_a, op_b):
    from tests.cusparse_generic_baseline import _native_operations

    expected = lambda op: "trans" if op == "conj" and not dtype.is_complex else op
    assert _native_operations(op_a, op_b, dtype, dtype) == (expected(op_a), expected(op_b))
    # A and B are normalized independently, never discard complex conjugation.
    assert _native_operations("conj", "conj", torch.float16, torch.complex64) == ("trans", "conj")
    with pytest.raises(ValueError):
        _native_operations("invalid", op_b, dtype, dtype)


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_vendor_effective_operations_reach_native_call(monkeypatch, backend, matrix, fmt, dtype):
    import ctypes
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tests import cusparse_generic_baseline as baseline

    prefix = "hipsparse" if backend == "rocm" else "cusparse"
    operation = "SpMM" if matrix else "SpMV"
    captured, released = [], []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            if self.name == prefix + operation + "_bufferSize":
                captured.append(args[1:3] if matrix else args[1:2])
                return 3  # Unknown invalid arguments must remain ERROR, never SKIP.
            if self.name.startswith(prefix + "Create"):
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(17)
            elif self.name == prefix + "GetVersion":
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = 12000
            elif self.name.startswith(prefix + "Destroy"):
                released.append(self.name)
            return 0

    suffixes = ["Create", "Destroy", "SetStream", "GetVersion", "Create" + fmt.capitalize(), "DestroySpMat",
                "CreateDnMat" if matrix else "CreateDnVec", "DestroyDnMat" if matrix else "DestroyDnVec",
                operation, operation + "_bufferSize"]
    library = SimpleNamespace(**{prefix + suffix: Function(prefix + suffix) for suffix in suffixes})
    monkeypatch.setattr(baseline.ctypes, "CDLL", lambda name: library)
    monkeypatch.setattr(baseline.ctypes.util, "find_library", lambda name: name)
    monkeypatch.setattr(torch.version, "hip", "6.1" if backend == "rocm" else None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=23))

    def tensor(dt, dense=False):
        return SimpleNamespace(dtype=dt, device=torch.device("cuda:0"), ndim=2 if dense and matrix else 1,
                               shape=(1, 1), is_contiguous=lambda: True, numel=lambda: 1,
                               stride=lambda axis: 1, data_ptr=lambda: 19)

    arrays = (tensor(dtype), tensor(torch.int32), tensor(torch.int32))
    with pytest.raises(RuntimeError, match="bufferSize.*status 3"):
        baseline._prepare_registered_generic(fmt, arrays, (1, 1), tensor(dtype, True),
            tensor(dtype, True), "conj", "conj" if matrix else "non",
            torch.complex64 if dtype.is_complex else torch.float32, backend, matrix)
    op = 2 if dtype.is_complex else 1
    assert captured == [(op, op) if matrix else (op,)]
    assert len(released) == 4  # two dense descriptors, sparse descriptor, handle


def test_hipsparse_partial_descriptor_failure_releases_resources(monkeypatch):
    import ctypes
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tests import cusparse_generic_baseline as baseline

    released = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            if self.name == "hipsparseCreateDnVec":
                return 3  # invalid value is an interface fault, not unsupported
            if self.name.startswith("hipsparseCreate"):
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(17)
            elif self.name == "hipsparseGetVersion":
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = 301000
            elif self.name.startswith("hipsparseDestroy"):
                released.append(self.name)
            return 0

    library = SimpleNamespace(**{name: Function(name) for name in (
        "hipsparseCreate", "hipsparseDestroy", "hipsparseSetStream", "hipsparseGetVersion",
        "hipsparseCreateCsr", "hipsparseDestroySpMat", "hipsparseCreateDnVec", "hipsparseDestroyDnVec",
    )})
    monkeypatch.setattr(baseline.ctypes, "CDLL", lambda name: library)
    monkeypatch.setattr(baseline.ctypes.util, "find_library", lambda name: name)
    monkeypatch.setattr(torch.version, "hip", "6.1")
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=23))

    def tensor(dtype):
        return SimpleNamespace(dtype=dtype, device=torch.device("cuda:0"), ndim=1,
                               is_contiguous=lambda: True, numel=lambda: 1,
                               stride=lambda axis: 1, data_ptr=lambda: 19)

    a, index, ptr = tensor(torch.float16), tensor(torch.int32), tensor(torch.int32)
    with pytest.raises(RuntimeError, match="CreateDnVec.*status 3"):
        baseline.prepare_spmv("csr", (a, index, ptr), (1, 1), tensor(torch.float16),
                              tensor(torch.float32), "non", torch.float32, backend="rocm")
    assert released == ["hipsparseDestroySpMat", "hipsparseDestroy"]


@pytest.mark.parametrize("outcome", ["abort", "startup", "timeout", "unsupported", "pass", "malformed"])
def test_vendor_worker_result_isolation(monkeypatch, tmp_path, outcome):
    import json
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from tests import cusparse_generic_baseline as baseline

    def fake_run(command, **kwargs):
        assert command[-3] == "--isolated-benchmark"
        assert command[1:3] == ["-S", "-c"]
        assert "PYTHONHOME" not in kwargs["env"]
        assert kwargs["env"]["PYTHONPATH"]
        assert kwargs["check"] is False
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "abort":
            return SimpleNamespace(returncode=-6)
        if outcome == "startup":
            kwargs["stderr"].write("Fatal Python error: init_fs_encoding\nModuleNotFoundError: No module named 'encodings'\n")
            return SimpleNamespace(returncode=1)
        response = {"vendor_status": "SKIP", "vendor_reason": "not supported"}
        if outcome == "pass":
            response = {"vendor_status": "PASS", "vendor_ms": 0.25}
        Path(command[-1]).write_text("invalid json" if outcome == "malformed" else json.dumps(response))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = baseline.benchmark_isolated({}, log_prefix=tmp_path / "case")
    expected = {"abort": "ERROR", "startup": "ERROR", "timeout": "ERROR", "malformed": "ERROR", "unsupported": "SKIP", "pass": "PASS"}
    assert result["vendor_status"] == expected[outcome]
    assert Path(result["vendor_stderr"]).exists()
    if outcome == "abort":
        assert result["vendor_exit_code"] == -6
    if outcome == "startup":
        assert result["vendor_error_stage"] == "python_startup"
        assert "stdlib" in result["vendor_reason"]
    if outcome != "pass":
        assert "vendor_ms" not in result


def test_vendor_worker_inherits_working_python_paths(monkeypatch, tmp_path):
    import encodings
    import json
    import subprocess
    import sys
    from pathlib import Path
    from tests import cusparse_generic_baseline as baseline

    # Exercise startup without importing an accelerator or invoking a vendor API.
    script = tmp_path / "worker.py"
    script.write_text("import sys, encodings, json\n"
                      "print(json.dumps([sys.prefix, sys.exec_prefix, sys.argv[1:]]))\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "wrong-prefix"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "wrong-search-path"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-example")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/container/vendor/libs")
    command, env = baseline._vendor_python_launch(script, "request", "response")
    assert command[0] == sys.executable  # Do not resolve a venv symlink to host Python.
    assert str(Path(encodings.__file__).parent.parent) in env["PYTHONPATH"].split(os.pathsep)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-example"
    assert env["LD_LIBRARY_PATH"] == "/container/vendor/libs"
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    prefix, exec_prefix, argv = json.loads(result.stdout.splitlines()[-1])
    assert (prefix, exec_prefix) == (sys.prefix, sys.exec_prefix)
    assert argv == ["--isolated-benchmark", "request", "response"]


@pytest.mark.parametrize("status", ["PASS", "FAIL", "SKIP", "ERROR"])
def test_benchmark_checkpoint_replaces_pending(tmp_path, status):
    import json
    from tests.benchmark_utils import _write_benchmark_state

    path = tmp_path / "logs" / "case.flagsparse.json"
    _write_benchmark_state(path, {"vendor_status": "PENDING", "checkpoint_state": "before_vendor"})
    final = {"vendor_status": status, "checkpoint_state": "completed", "matrix": "test.mtx"}
    _write_benchmark_state(path, final)
    assert json.loads(path.read_text(encoding="utf-8")) == final
    assert not path.with_name(path.name + ".tmp").exists()


def test_mixed_progress_is_outside_measured_iteration_loops():
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "benchmark_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    bench = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "run_type_combo_benchmark")
    calls = lambda node, name: [n for n in ast.walk(node) if isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Name) and n.func.id == name]
    stages = {n.args[0].value for n in calls(bench, "progress") if n.args
              and isinstance(n.args[0], ast.Constant)}
    assert {"source_load", "prepare", "first_run_compile_and_execute", "warmup",
            "event_samples", "wall_samples", "accuracy", "phase_diagnostics",
            "vendor_payload", "vendor_worker", "case_completed", "benchmark_completed"} <= stages
    for node in ast.walk(bench):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if any(isinstance(n, ast.Attribute) and n.attr in ("iters", "warmup")
                   for n in ast.walk(node.iter)):
                assert not calls(node, "progress")
                assert not calls(node, "_write_benchmark_state")
    emit = next(n for n in bench.body if isinstance(n, ast.FunctionDef) and n.name == "emit")
    assert calls(emit, "_write_benchmark_state")
    assert any(isinstance(n, ast.keyword) and n.arg == "checkpoint_state"
               and isinstance(n.value, ast.Constant) and n.value.value == "completed"
               for n in ast.walk(emit))


def test_vendor_diagnostic_cli_and_accuracy_metrics():
    import argparse
    from tests.benchmark_utils import add_type_combo_arguments, parse_vendor_options
    from tests.utils import finite_speedup, tensor_accuracy_metrics

    parser = argparse.ArgumentParser()
    add_type_combo_arguments(parser, spmm=True)
    args = parser.parse_args(["--type-combos", "bfloat16:bfloat16:float32",
                              "--vendor-diagnostic", "ones_fanin,rectangular",
                              "--vendor-alg", "default,csr_alg1,csr_alg2,csr_alg3",
                              "--vendor-preprocess", "off,on"])
    algorithms, preprocessing, cases, lengths = parse_vendor_options("spmm_csc", args)
    assert algorithms == ("default", "csr_alg1", "csr_alg2", "csr_alg3")
    assert preprocessing == (False, True) and cases == ("ones_fanin", "rectangular")
    assert lengths == (128, 256, 512, 1024, 2048, 4096)
    with pytest.raises(ValueError):
        parse_vendor_options("spmv_csc", args)
    args.type_combos = None
    with pytest.raises(ValueError):
        parse_vendor_options("spmm_csc", args)
    assert finite_speedup(2.0, 4.0) == 0.5  # Independent of either accuracy status.
    for value in (None, 0, -1, float("nan"), float("inf")):
        assert finite_speedup(value, 1.0) is None
        assert finite_speedup(1.0, value) is None
    expected = torch.tensor([1., 0., 2.])
    metrics = tensor_accuracy_metrics(torch.tensor([1., 1., float("nan")]), expected, 1e-3, 1e-4)
    assert not metrics["passed"] and metrics["failed_elements"] == 2
    assert metrics["output_nonfinite"] == 1 and metrics["reference_nonfinite"] == 0
    assert tensor_accuracy_metrics(expected, expected, 0, 0)["passed"]


@pytest.mark.parametrize("algorithm,number,preprocess", [
    ("default", 0, False), ("csr_alg1", 4, False), ("csr_alg1", 4, True),
    ("csr_alg2", 6, False), ("csr_alg3", 12, False), ("csr_alg3", 12, True),
])
@pytest.mark.parametrize("dense_shape,dense_stride,layout,ld", [
    ((2, 2), (2, 1), "row", 2),
    ((1, 4), (1, 1), "row", 4),
    ((4, 1), (1, 1), "row", 1),
    ((2, 2), (1, 3), "col", 3),
], ids=["ordinary", "singleton-row", "singleton-col", "padded-col"])
def test_cuda_vendor_algorithm_and_preprocess_reach_c_api(monkeypatch, algorithm, number, preprocess,
                                                        dense_shape, dense_stride, layout, ld):
    import ctypes
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tests import cusparse_generic_baseline as baseline

    calls, released, descriptors = [], [], []
    class Function:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args):
            if self.name == "CreateDnMat":
                descriptors.append((args[1:4], args[4].value, args[6]))
            if self.name.startswith("Create"):
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(17)
            if self.name == "GetVersion":
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = 12900
            if self.name.startswith("SpMM"):
                calls.append((self.name, args[-2]))
            if self.name.startswith("Destroy"):
                released.append(self.name)
            return 0
    library = SimpleNamespace(_name="mock-cusparse", **{"cusparse" + name: Function(name) for name in (
        "Create", "Destroy", "SetStream", "GetVersion", "CreateCsc", "DestroySpMat",
        "CreateDnMat", "DestroyDnMat", "SpMM", "SpMM_bufferSize", "SpMM_preprocess")})
    monkeypatch.setattr(baseline.ctypes, "CDLL", lambda name: library)
    monkeypatch.setattr(baseline.ctypes.util, "find_library", lambda name: name)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=23))
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: SimpleNamespace(data_ptr=lambda: 999))
    def tensor(dtype, dense=False):
        return SimpleNamespace(dtype=dtype, device=torch.device("cuda:0"), ndim=2 if dense else 1,
            shape=dense_shape if dense else (4,), stride=lambda axis: dense_stride[axis],
            is_contiguous=lambda: True, numel=lambda: 4, data_ptr=lambda: 19)
    arrays = (tensor(torch.bfloat16), tensor(torch.int32), tensor(torch.int64))
    out = tensor(torch.float32, True)
    plan = baseline.prepare_spmm("csc", arrays, (2, 2), tensor(torch.bfloat16, True), out,
                                 "non", "non", torch.float32, backend="cuda",
                                 algorithm=algorithm, preprocess=preprocess)
    assert plan.run() is out
    assert calls == [("SpMM_bufferSize", number)] + ([("SpMM_preprocess", number)] if preprocess else []) + [("SpMM", number)]
    assert plan.metadata["vendor_preprocessed"] is preprocess
    assert plan.metadata["vendor_alg"] == algorithm
    order = baseline._ORDER_ROW if layout == "row" else baseline._ORDER_COL
    expected_descriptor = ((*dense_shape, ld), 19, order)
    assert descriptors == [expected_descriptor, expected_descriptor]
    plan.close()
    plan.close()
    assert len(released) == 4

    # The historical q4 entry uses the same descriptor policy and data pointer.
    monkeypatch.setattr(baseline, "_lib", lambda: library)
    baseline._dnmat(tensor(torch.float32, True))
    assert descriptors[-1] == expected_descriptor


def test_vendor_scan_reuses_native_measurements_outside_timing():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "benchmark_utils.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_type_combo_benchmark")
    variant_loop = next(n for n in ast.walk(function) if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Tuple) and isinstance(n.target.elts[0], ast.Name)
                        and n.target.elts[0].id == "variant_index")
    names = {n.func.id for n in ast.walk(variant_loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "emit" in names and "finite_speedup" in names and "run" not in names
    assert not any(isinstance(n, ast.Compare) and any(isinstance(v, ast.Constant) and v.value == "PASS"
                   for v in n.comparators) for n in ast.walk(variant_loop))
