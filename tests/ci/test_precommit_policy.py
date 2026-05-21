"""Static policy checks for repository pre-commit wiring."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_precommit_config_exists():
    path = PROJECT_ROOT / ".pre-commit-config.yaml"
    assert path.is_file()


def test_makefile_runs_precommit():
    text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "pre-commit run --all-files" in text
