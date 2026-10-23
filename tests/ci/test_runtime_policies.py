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
