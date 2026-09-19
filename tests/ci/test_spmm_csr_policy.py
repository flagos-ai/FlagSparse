"""CPU policy coverage, loaded without importing flagsparse, torch or Triton."""

import importlib.util
from pathlib import Path
import sys

import pytest


_PATH = Path(__file__).resolve().parents[2] / "src/flagsparse/sparse_operations/_spmm_csr_config.py"
_SPEC = importlib.util.spec_from_file_location("spmm_policy_under_test", _PATH)
policy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = policy
_SPEC.loader.exec_module(policy)


def caps(backend="cuda", **changes):
    fields = dict(backend=backend, target="test", arch="test", subgroup_width=32,
                  max_threads_per_block=1024, legal_num_warps=(1, 2, 4, 8),
                  fp64=True, int64=True, reduction=True, stable_sort=True, scan=True)
    return policy.BackendCaps(**(fields | changes))


@pytest.mark.parametrize("backend", policy.BACKENDS)
@pytest.mark.parametrize("algorithm", policy.NEW_ALGORITHMS)
def test_profiles_depend_on_capabilities_not_cuda_alias(backend, algorithm):
    config, meta = policy.resolve_config(algorithm, "complex128", 17, "row", caps(backend))
    assert config["block_n"] == 16
    assert meta["backend_caps"]["backend"] == backend
    assert policy.algorithm_spec(algorithm)["validation"] == "unverified"


@pytest.mark.parametrize("overrides", [dict(block_k=3), dict(num_warps=32),
    dict(num_stages=2), dict(reduce_block_size=1), dict(unknown=1), dict(workspace_bytes=4)])
def test_explicit_invalid_config_is_not_clamped(overrides):
    with pytest.raises(ValueError):
        policy.resolve_config("csr_row_tile", "float32", 32, "row", caps(), overrides)


def test_unknown_capabilities_do_not_enable_routes():
    with pytest.raises(NotImplementedError, match="fp64"):
        policy.resolve_config("csr_row_tile", "complex128", 32, "row", caps(fp64=None))
    with pytest.raises(NotImplementedError, match="stable_sort"):
        policy.resolve_config("csr_row_tile", "float32", 32, "row", caps(stable_sort=None), op="conj")
    policy.resolve_config("csr_row_tile", "complex64", 32, "row", caps(fp64=False))


def test_width_and_layout_change_deterministic_thresholds():
    short, _ = policy.resolve_config("csr_adaptive_tile_split", "float32", 16, "row", caps())
    wide, _ = policy.resolve_config("csr_adaptive_tile_split", "float32", 128, "col", caps())
    assert short["short_row_threshold"] == 32
    assert wide["short_row_threshold"] == 16
    assert wide["split_row_threshold"] == 2 * short["split_row_threshold"]


def test_illegal_builtin_profile_is_rejected(monkeypatch):
    monkeypatch.setitem(policy.ARCH_PROFILES, ("cuda", "test"), dict(num_warps=128))
    config, meta = policy.resolve_config("csr_row_tile", "float32", 16, "row", caps())
    assert config["num_warps"] == 4
    assert meta["config_rejections"]


def test_architecture_algorithm_profile_does_not_leak(monkeypatch):
    monkeypatch.setitem(policy.ARCH_PROFILES, ("cuda", "test", "csr_row_kparallel"),
                        dict(block_k=64))
    vector, _ = policy.resolve_config("csr_row_kparallel", "float32", 32, "row", caps())
    split, _ = policy.resolve_config("csr_split_nnz_reduce", "float32", 32, "row", caps())
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
