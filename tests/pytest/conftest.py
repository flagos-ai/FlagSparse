"""Pytest hooks: ``--mode quick|normal`` toggles shape/dtype lists ``QUICK_MODE``."""

import pytest

QUICK_MODE = False


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="normal",
        choices=["normal", "quick"],
        help="quick: fewer shapes/dtypes (FlagGems-style QUICK_MODE).",
    )


def pytest_configure(config):
    global QUICK_MODE
    QUICK_MODE = config.getoption("--mode") == "quick"
