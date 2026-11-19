# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Which oracle a backend uses does not decide whether it has a timing baseline.

The SpMM scripts gated the PyTorch baseline on _use_scipy_accuracy_reference(),
which is true for every backend outside CUDA and ROCm. Two different questions
were being answered by one predicate:

    is the torch.sparse VALUE trustworthy here   -- false on MACA (its fp32 CSR
                                                    path returns non-finite)
    does a torch.sparse route RUN here           -- still true on MACA, and its
                                                    latency is that backend's
                                                    delivery baseline

Only MUSA answers no to both: it registers no sparse matmul in any layout or
dtype. The conflation left all four SpMM delivery variants on MACA reporting
`Passed` with an empty speedup, and the MACA CSV even carried the MUSA wording.

Asserted on the source: importing these modules needs torch and an accelerator,
so a behavioural test would skip in CI -- the one environment that has to hold
this line. The behaviour itself was checked by running both scripts under
FLAGSPARSE_BACKEND=metax, where pytorch_ms went from empty to measured.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SPMM_SCRIPTS = ("tests/test_spmm.py", "tests/test_spmm_coo.py")


def _calls_in(node):
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


def _function(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.mark.parametrize("script", SPMM_SCRIPTS)
def test_the_baseline_decision_names_the_musa_predicate(script):
    """A backend without a sparse matmul is the only one with nothing to time."""
    calls = _calls_in(_function(script, "_build_pytorch_reference"))
    assert "_is_mthreads_runtime" in calls, (
        f"{script}: _build_pytorch_reference must decide the PyTorch baseline on "
        "_is_mthreads_runtime(), not on which accuracy oracle the backend uses"
    )


@pytest.mark.parametrize("script", SPMM_SCRIPTS)
def test_the_oracle_choice_is_still_made_where_it_belongs(script):
    """The fix must not have taken SciPy away from the backends that need it."""
    calls = _calls_in(_function(script, "_build_pytorch_reference"))
    assert "_use_scipy_accuracy_reference" in calls, script


def test_no_spmm_script_hardcodes_the_musa_wording_for_every_backend():
    """The MACA CSV reported "unavailable on MUSA" while running on a C550."""
    for script in SPMM_SCRIPTS:
        source = (ROOT / script).read_text(encoding="utf-8")
        assert "baseline unavailable on MUSA" not in source, script


def test_the_csr_script_reports_why_a_baseline_is_missing():
    """`-` in the report has to be traceable to a reason, not to a silent skip."""
    source = (ROOT / "tests/test_spmm.py").read_text(encoding="utf-8")
    assert "if pytorch_op is None:" in source
    assert 'result.setdefault("pytorch_reason"' in source


@pytest.mark.parametrize("shape,strides,expected", [
    ((1, 4), (1, 1), ("row", 4)),
    ((1, 4), (0, 1), ("row", 4)),
    ((4, 1), (1, 1), ("row", 1)),
    ((4, 1), (1, 0), ("col", 4)),
    ((4, 1), (1, 4), ("col", 4)),
    ((1, 4), (8, 1), ("row", 8)),
    ((4, 1), (1, 8), ("col", 8)),
    ((1, 4), (9, 2), ("col", 2)),
    ((4, 1), (2, 9), ("row", 2)),
    ((1, 1), (0, 0), ("row", 1)),
    ((3, 4), (4, 1), ("row", 4)),
    ((3, 4), (7, 1), ("row", 7)),
    ((3, 4), (1, 3), ("col", 3)),
    ((3, 4), (1, 7), ("col", 7)),
    ((0, 4), (4, 1), ("row", 4)),
    ((4, 0), (1, 4), ("col", 4)),
    ((0, 0), (0, 0), ("row", 1)),
], ids=["single-row-transposed", "single-row-unused-zero", "single-col",
        "single-col-unused-zero", "single-col-preserves-order", "single-row-padded", "single-col-padded",
        "strided-single-row", "strided-single-col",
        "scalar", "row", "padded-row", "col", "padded-col",
        "zero-rows", "zero-cols", "empty"])
def test_native_dnmat_layout_preserves_element_addresses(shape, strides, expected):
    """Exercise the pure descriptor policy without importing torch or a GPU runtime."""
    from types import SimpleNamespace

    function = _function("tests/cusparse_generic_baseline.py", "_dense_matrix_layout")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<dnmat-policy>", "exec"), namespace)
    tensor = SimpleNamespace(ndim=2, shape=shape, stride=lambda axis: strides[axis])
    order, ld = namespace["_dense_matrix_layout"](tensor)
    assert (order, ld) == expected
    assert ld >= max(1, shape[1] if order == "row" else shape[0])
    for row in range(shape[0]):
        for col in range(shape[1]):
            actual_offset = row * strides[0] + col * strides[1]
            native_offset = row * ld + col if order == "row" else col * ld + row
            assert native_offset == actual_offset


@pytest.mark.parametrize("shape,strides", [
    ((3, 4), (1, 1)), ((3, 4), (3, 1)), ((3, 4), (1, 2)),
    ((3, 4), (0, 1)), ((3, 4), (1, 0)), ((3, 4), (8, 2)),
    ((1, 4), (1, 0)), ((4, 1), (0, 1)), ((4,), (1,)),
], ids=["overlap", "short-row-ld", "short-col-ld", "broadcast-row",
        "broadcast-col", "non-unit-minor", "broadcast-single-row",
        "broadcast-single-col", "not-a-matrix"])
def test_native_dnmat_rejects_unrepresentable_layout(shape, strides):
    from types import SimpleNamespace

    function = _function("tests/cusparse_generic_baseline.py", "_dense_matrix_layout")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<dnmat-policy>", "exec"), namespace)
    tensor = SimpleNamespace(ndim=len(shape), shape=shape, stride=lambda axis: strides[axis])
    with pytest.raises(ValueError, match="native DnMat"):
        namespace["_dense_matrix_layout"](tensor)


@pytest.mark.parametrize("entry", ["_dnmat", "_prepare_registered_generic"])
def test_native_dnmat_entries_share_layout_policy(entry):
    function = _function("tests/cusparse_generic_baseline.py", entry)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_dense_matrix_layout" for n in ast.walk(function))
