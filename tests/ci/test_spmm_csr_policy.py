"""CPU policy coverage, loaded without importing flagsparse, torch or Triton."""

import importlib.util
import sys
from pathlib import Path

import pytest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/flagsparse/sparse_operations/_spmm_csr_config.py"
)
_SPEC = importlib.util.spec_from_file_location("spmm_policy_under_test", _PATH)
policy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = policy
_SPEC.loader.exec_module(policy)


def caps(backend="cuda", **changes):
    fields = dict(
        backend=backend,
        target="test",
        arch="test",
        subgroup_width=32,
        max_threads_per_block=1024,
        legal_num_warps=(1, 2, 4, 8),
        fp64=True,
        int64=True,
        reduction=True,
        stable_sort=True,
        scan=True,
    )
    return policy.BackendCaps(**(fields | changes))


@pytest.mark.parametrize("backend", policy.BACKENDS)
@pytest.mark.parametrize("algorithm", policy.NEW_ALGORITHMS)
def test_profiles_depend_on_capabilities_not_cuda_alias(backend, algorithm):
    config, meta = policy.resolve_config(
        algorithm, "complex128", 17, "row", caps(backend)
    )
    assert config["block_n"] == 16
    assert meta["backend_caps"]["backend"] == backend
    assert policy.algorithm_spec(algorithm)["validation"] == "unverified"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(block_k=3),
        dict(num_warps=32),
        dict(num_stages=2),
        dict(reduce_block_size=1),
        dict(unknown=1),
        dict(workspace_bytes=4),
    ],
)
def test_explicit_invalid_config_is_not_clamped(overrides):
    with pytest.raises(ValueError):
        policy.resolve_config("csr_row_tile", "float32", 32, "row", caps(), overrides)


def test_unknown_capabilities_do_not_enable_routes():
    with pytest.raises(NotImplementedError, match="fp64"):
        policy.resolve_config("csr_row_tile", "complex128", 32, "row", caps(fp64=None))
    with pytest.raises(NotImplementedError, match="stable_sort"):
        policy.resolve_config(
            "csr_row_tile",
            "float32",
            32,
            "row",
            caps(stable_sort=None),
            op="conj",
        )
    policy.resolve_config("csr_row_tile", "complex64", 32, "row", caps(fp64=False))


def test_width_and_layout_change_deterministic_thresholds():
    short, _ = policy.resolve_config(
        "csr_adaptive_tile_split", "float32", 16, "row", caps()
    )
    wide, _ = policy.resolve_config(
        "csr_adaptive_tile_split", "float32", 128, "col", caps()
    )
    assert short["short_row_threshold"] == 32
    assert wide["short_row_threshold"] == 16
    assert wide["split_row_threshold"] == 2 * short["split_row_threshold"]


@pytest.mark.parametrize("dtype,rows,limit", [("float32", 8, 32), ("complex128", 4, 16)])
@pytest.mark.parametrize("n", [0, 1, 7, 31, 33, 128])
def test_panel_defaults_and_config_isolation(dtype, rows, limit, n):
    config, _ = policy.resolve_config("csr_row_panel", dtype, n, "row", caps())
    assert config["tile_rows"] == rows
    assert config["tile_n"] == min(limit, 1 << (max(1, n) - 1).bit_length())
    assert config["num_warps"] == 2
    assert config["panel_accumulators"] == 2
    for accumulators in (1, 2, 4):
        explicit, meta = policy.resolve_config("csr_row_panel", dtype, n, "col", caps(),
                                               {"panel_accumulators": accumulators})
        assert explicit["panel_accumulators"] == accumulators
        assert meta["config_source"] == "explicit"
    for bad in (0, 3, 8):
        with pytest.raises(ValueError):
            policy.resolve_config("csr_row_panel", dtype, n, "row", caps(), {"panel_accumulators": bad})
    with pytest.raises(ValueError, match="unknown"):
        policy.resolve_config("csr_row_tile", dtype, n, "row", caps(), {"panel_accumulators": 2})


def test_panel_profile_does_not_enable_panel_fields_on_other_algorithms(monkeypatch):
    monkeypatch.setitem(policy.BACKEND_PROFILES, "cuda", {"panel_accumulators": 4})
    config, meta = policy.resolve_config("csr_row_tile", "float32", 32, "row", caps())
    assert "panel_accumulators" not in config
    assert meta["config_rejections"]
    panel, _ = policy.resolve_config("csr_row_panel", "float32", 32, "row", caps())
    assert panel["panel_accumulators"] == 4


def test_illegal_builtin_profile_is_rejected(monkeypatch):
    monkeypatch.setitem(policy.ARCH_PROFILES, ("cuda", "test"), dict(num_warps=128))
    config, meta = policy.resolve_config("csr_row_tile", "float32", 16, "row", caps())
    assert config["num_warps"] == 4
    assert meta["config_rejections"]


def test_architecture_algorithm_profile_does_not_leak(monkeypatch):
    monkeypatch.setitem(
        policy.ARCH_PROFILES, ("cuda", "test", "csr_row_kparallel"), dict(block_k=64)
    )
    vector, _ = policy.resolve_config(
        "csr_row_kparallel", "float32", 32, "row", caps()
    )
    split, _ = policy.resolve_config(
        "csr_split_nnz_reduce", "float32", 32, "row", caps()
    )
    assert vector["block_k"] == 64
    assert split["block_k"] == 32


@pytest.mark.parametrize("n", (1, 7, 32, 129, 4096))
@pytest.mark.parametrize("element_bytes", (4, 8, 16))
@pytest.mark.parametrize("budget", (48, 192, 1 << 20, 256 << 20))
def test_numeric_workspace_geometry_is_bounded(n, element_bytes, budget):
    config = dict(block_n=32, workspace_bytes=budget)
    wave, capacity = policy.workspace_geometry(n, element_bytes, config)
    assert 1 <= wave <= min(n, 32)
    assert capacity >= 1
    assert 3 * wave * capacity * element_bytes <= budget


@pytest.mark.parametrize("algorithm", policy.COO_ALGORITHMS)
@pytest.mark.parametrize("backend", policy.BACKENDS)
def test_coo_profiles_require_actual_capabilities(algorithm, backend):
    device = caps(backend, fp32_atomic=True, fp64_atomic=True)
    config, meta = policy.resolve_coo_config(algorithm, "complex128", device, n=7)
    assert config["num_stages"] == 1
    assert meta["backend"] == backend
    assert meta["validation"] == "unverified"
    with pytest.raises(ValueError):
        policy.resolve_coo_config(algorithm, "float32", device, {"unknown": 1})
    with pytest.raises(ValueError):
        policy.resolve_coo_config(algorithm, "float32", device, {"num_warps": 3})


def test_coo_atomic_and_fp64_are_independent_capabilities():
    with pytest.raises(NotImplementedError, match="fp64_atomic"):
        policy.resolve_coo_config("coo_segmented_panel_atomic", "float64", caps())
    config, _ = policy.resolve_coo_config("coo_rowrun_panel", "float64", caps(), n=7)
    assert config["tile_n"] == 8
    with pytest.raises(NotImplementedError, match="sort"):
        policy.resolve_coo_config("coo_rowrun_panel", "float64", caps(stable_sort=None))


def test_coo_subgroup_configuration_validates_before_division():
    with pytest.raises(ValueError):
        policy.resolve_coo_config("coo_rowrun_subgroup", "float32", caps(), {"lanes_per_row": 0})
    cfg, _ = policy.resolve_coo_config("coo_rowrun_subgroup", "float32", caps(legal_num_warps=(1,)))
    assert cfg["num_warps"] == 1
    assert cfg["rows_per_program"] == 4



def test_coo_invalid_builtin_profile_is_reported(monkeypatch):
    monkeypatch.setitem(policy.COO_ARCH_PROFILES, ("cuda", "test"),
                        {"coo_rowrun_panel": {"num_warps": 3}})
    cfg, meta = policy.resolve_coo_config("coo_rowrun_panel", "float64", caps())
    assert cfg["num_warps"] == 2
    assert meta["rejected_profiles"]
    with pytest.raises(ValueError):
        policy.resolve_coo_config("coo_rowrun_panel", "float64", caps(), {"num_warps": 3})


@pytest.mark.parametrize("alg,ops", list(policy.CSC_ALGORITHMS.items()))
@pytest.mark.parametrize("backend", policy.BACKENDS)
def test_csc_capability_and_config_contract(alg, ops, backend):
    device = caps(backend, fp32_atomic=True, fp64_atomic=True)
    cfg, meta = policy.resolve_csc_config(alg, "complex128", 7, device, op=ops[0])
    assert cfg["num_stages"] == 1
    assert meta["validation"] == "unverified"
    with pytest.raises(ValueError):
        policy.resolve_csc_config(alg, "float32", 7, device, {"unknown": 1}, op=ops[0])
    with pytest.raises(ValueError):
        policy.resolve_csc_config(alg, "float32", 7, device, {"num_warps": 3}, op=ops[0])
    wrong_op = "trans" if ops == ("non",) else "non"
    with pytest.raises(ValueError, match="op"):
        policy.resolve_csc_config(alg, "float32", 7, device, op=wrong_op)


def test_csc_atomic_capability_and_zero_subgroup():
    with pytest.raises(NotImplementedError, match="atomic"):
        policy.resolve_csc_config("csc_col_tile_atomic", "float64", 1, caps(), op="non")
    with pytest.raises(ValueError):
        policy.resolve_csc_config("csc_col_subgroup", "float32", 1, caps(), {"lanes_per_column": 0}, op="trans")
    cfg, _ = policy.resolve_csc_config("csc_col_subgroup", "float32", 1, caps(legal_num_warps=(1,)), op="trans")
    assert cfg["num_warps"] == 1



def test_csc_unknown_capability_and_profile_rejection(monkeypatch):
    with pytest.raises(NotImplementedError):
        policy.resolve_csc_config("csc_col_panel", "float32", 7, caps(reduction=None), op="trans")
    monkeypatch.setitem(policy.CSC_ARCH_PROFILES, ("cuda", "test", "csc_col_panel"), {"num_warps": 3})
    cfg, info = policy.resolve_csc_config("csc_col_panel", "float32", 7, caps(), op="trans")
    assert cfg["num_warps"] == 2
    assert info["config_rejections"]


@pytest.mark.parametrize("algorithm", policy.NEW_ALGORITHMS)
def test_spmm_csr_half_uses_fp32_capabilities(algorithm):
    config, _ = policy.resolve_config(algorithm, "float16", 7, "row", caps(fp64=False))
    assert config["num_stages"] == 1


@pytest.mark.parametrize("algorithm,ops", list(policy.CSC_ALGORITHMS.items()))
def test_csc_half_does_not_require_fp16_atomic(algorithm, ops):
    config, _ = policy.resolve_csc_config(algorithm, "float16", 7,
        caps(fp64=False, fp32_atomic=True, fp64_atomic=False), op=ops[0])
    assert config["num_stages"] == 1
