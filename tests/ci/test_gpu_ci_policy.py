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

"""Static policy checks for GPU CI wiring.

These tests make the GPU workflow reviewable from CPU-only CI. The workflow
itself still requires a self-hosted CUDA runner to execute.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
TOOLS_DIR = PROJECT_ROOT / "tools" / "ci"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_gpu_accuracy_workflow_is_self_hosted_and_manual():
    text = _read(WORKFLOWS_DIR / "gpu-ci.yml")
    assert "workflow_dispatch:" in text
    # Runner scale sets (ARC) must be referenced by name alone in `runs-on`;
    # combining it with labels like self-hosted/linux breaks job matching.
    assert "runs-on: test-flagsparse" in text
    assert "actions/setup-python" not in text
    assert "python3.12 --version" in text
    assert "$GITHUB_PATH" not in text
    assert (
        "pip install --upgrade -r tools/ci/requirements-triton-smoke.lock.txt"
        not in text
    )
    assert "python3.12 -m pip show build pytest torch" in text
    assert "python3.12 -m pip uninstall -y triton" in text
    assert "flagtree===0.6.0rc2" in text
    assert "mkdir -p pytest_results" in text
    assert "gpu_ci_metadata.json" in text


def test_gpu_accuracy_workflow_checks_cuda_before_tests():
    text = _read(WORKFLOWS_DIR / "gpu-ci.yml")
    assert "nvidia-smi" in text
    assert "tools/ci/check_gpu_environment.py" in text
    assert "--require-cuda" in text
    assert "run_flagsparse_accuracy.py" in text


def test_gpu_benchmark_workflow_uploads_artifacts():
    text = _read(WORKFLOWS_DIR / "gpu-benchmark.yml")
    assert "runs-on: test-flagsparse" in text
    assert (
        "pip install --upgrade -r tools/ci/requirements-triton-smoke.lock.txt"
        not in text
    )
    assert "python3.12 -m pip show build pytest torch" in text
    assert "python3.12 -m pip uninstall -y triton" in text
    assert "flagtree===0.6.0rc2" in text
    assert "mkdir -p benchmark_results" in text
    assert "run_flagsparse_performance.py" in text
    assert "matrix_dir:" in text
    assert "actions/upload-artifact@v7" in text


def test_gpu_dependency_bundle_leaves_triton_to_flagtree():
    text = _read(TOOLS_DIR / "requirements-triton-smoke.lock.txt")
    assert "torch==2.9.0" in text
    assert "triton==" not in text


def test_gpu_environment_check_can_run_without_gpu():
    text = _read(TOOLS_DIR / "check_gpu_environment.py")
    assert "--require-cuda" in text
    assert "torch.cuda.is_available" in text
    assert "nvidia-smi" in text
    assert "metadata" in text


def test_migrated_algorithms_use_existing_accuracy_files_and_markers():
    for op in ("spmv_csr", "spmm_csr", "spmv_coo", "spmm_coo", "spmv_csc", "spmm_csc"):
        source = _read(PROJECT_ROOT / "tests" / "pytest" / f"test_{op}_accuracy.py")
        assert f"@pytest.mark.{op}" in source
        assert f"def test_{op}_migrated_routes" in source
        assert f"def test_{op}_migrated_empty" in source


def test_independent_mixed_cases_use_existing_gpu_delivery_files():
    candidates = {
        "spmv_csr": "csr_subgroup_pair_mixed",
        "spmm_csr": "csr_panel_staged_mixed",
        "spmv_coo": "coo_stream_prefetch_mixed",
        "spmm_coo": "coo_stream_panel_prefetch_mixed",
        "spmv_csc": "csc_subgroup_pair_mixed",
        "spmm_csc": "csc_panel_staged_mixed",
    }
    for op, name in candidates.items():
        text = _read(PROJECT_ROOT / "tests" / "pytest" / f"test_{op}_accuracy.py")
        assert name in text
        assert f"@pytest.mark.{op}" in text
        assert f"test_{op}_independent_mixed_boundaries" in text
    benchmark = _read(PROJECT_ROOT / "tests" / "benchmark_utils.py")
    assert '"compare"' in benchmark and "supported = query(" in benchmark
    assert 'row["metadata"] = meta' in benchmark


def test_coo_runclass_candidates_use_existing_delivery():
    for op, algorithm in (("spmv_coo", "coo_runclass_atomic"),
                          ("spmm_coo", "coo_runclass_panel_atomic")):
        accuracy = PROJECT_ROOT / "tests" / "pytest" / f"test_{op}_accuracy.py"
        assert list((PROJECT_ROOT / "tests").rglob(accuracy.name)) == [accuracy]
        text = _read(accuracy)
        assert algorithm in text and f"@pytest.mark.{op}" in text
        assert f"def test_{op}_runclass_native" in text
        benchmark = _read(PROJECT_ROOT / "tests" / f"test_{op}.py")
        assert "run_type_combo_benchmark" in benchmark
        assert f"list_{op}_algorithms" in benchmark
    import ast
    tree = ast.parse(_read(PROJECT_ROOT / "tests" / "test_spmm_coo.py"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    case = functions["run_one_alg_case"]
    assert "config" in {arg.arg for arg in case.args.args}
    prepare = [node for node in ast.walk(case) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "prepare_spmm_coo_route"
               and any(kw.arg == "config" for kw in node.keywords)]
    assert len(prepare) == 1
    for caller in ("run_all_dtypes_export_csv", "main"):
        calls = [node for node in ast.walk(functions[caller]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "run_one_alg_case"]
        assert calls and all(any(kw.arg == "config" for kw in node.keywords) for node in calls)


def test_short_csr_and_partition_csc_use_original_delivery():
    for op, names in {
        "spmv_csr": ("csr_row_scalar_strip", "csr_row_static_tile", "csr_row_scalar_bound", "csr_longrow_queue_atomic", "csr_row_scalar_tail_reduce", "csr_nnz_local_reduce_atomic", "csr_longrow_grid_atomic", "csr_short_rowqueue_vector"),
        "spmv_csc": ("csc_col_scalar_bound", "csc_col_scalar_tail_reduce"),
        "spmm_csc": ("csc_col_partition_panel_atomic",),
    }.items():
        accuracy = PROJECT_ROOT / "tests" / "pytest" / f"test_{op}_accuracy.py"
        assert list((PROJECT_ROOT / "tests").rglob(accuracy.name)) == [accuracy]
        text = _read(accuracy)
        assert f"@pytest.mark.{op}" in text
        for name in names:
            assert name in text
        benchmark = _read(PROJECT_ROOT / "tests" / f"test_{op}.py")
        assert "run_type_combo_benchmark" in benchmark
    # The existing runner and workflows collect the same operator markers.
    assert "run_flagsparse_accuracy.py" in _read(WORKFLOWS_DIR / "gpu-ci.yml")


def test_csc_regular_benchmark_forwards_explicit_configuration():
    import ast
    tree = ast.parse(_read(PROJECT_ROOT / "tests/test_spmm_csc.py"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    prepare = next(node for node in ast.walk(functions["_time_flagsparse_csc"])
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "prepare_spmm_csc_route")
    assert any(key.arg == "config" and isinstance(key.value, ast.Name)
               and key.value.id == "config" for key in prepare.keywords)
    call = next(node for node in ast.walk(functions["_run_case"])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_time_flagsparse_csc")
    assert any(key.arg == "config" for key in call.keywords)
