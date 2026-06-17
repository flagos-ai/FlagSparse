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
    assert 'python-version: "3.12"' in text
    assert "tools/ci/requirements-triton-smoke.lock.txt" in text


def test_gpu_accuracy_workflow_checks_cuda_before_tests():
    text = _read(WORKFLOWS_DIR / "gpu-ci.yml")
    assert "nvidia-smi" in text
    assert "tools/ci/check_gpu_environment.py" in text
    assert "--require-cuda" in text
    assert "run_flagsparse_accuracy.py" in text


def test_gpu_benchmark_workflow_uploads_artifacts():
    text = _read(WORKFLOWS_DIR / "gpu-benchmark.yml")
    assert "runs-on: test-flagsparse" in text
    assert "run_flagsparse_performance.py" in text
    assert "matrix_dir:" in text
    assert "actions/upload-artifact@v4" in text


def test_gpu_environment_check_can_run_without_gpu():
    text = _read(TOOLS_DIR / "check_gpu_environment.py")
    assert "--require-cuda" in text
    assert "torch.cuda.is_available" in text
    assert "nvidia-smi" in text
    assert "metadata" in text
