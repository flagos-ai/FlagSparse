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

"""Pytest hooks for FlagSparse accuracy tests.

``--mode quick|normal`` toggles shape/dtype lists via ``QUICK_MODE``.
``--record json --output <path>`` writes FlagGems-compatible case results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

QUICK_MODE = False
RECORD_JSON = False
REPORT_FILE = "accuracy_result.json"
TEST_RESULTS: dict[str, dict[str, object]] = {}
BUILTIN_MARKS = {
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "timeout",
    "tryfirst",
    "trylast",
    "usefixtures",
    "xfail",
}


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="normal",
        choices=["normal", "quick"],
        help="quick: fewer shapes/dtypes (FlagGems-style QUICK_MODE).",
    )
    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            choices=["none", "log", "json"],
            help="record test results in log/json files or not",
        )
        parser.addoption(
            "--output", default=REPORT_FILE, help="path to JSON result file"
        )
    except ValueError:
        pass
    try:
        parser.addoption(
            "--collect-marks",
            action="store_true",
            help="collect tests with marker information without executing them",
        )
    except ValueError:
        pass


def pytest_configure(config):
    global QUICK_MODE
    global RECORD_JSON
    global REPORT_FILE
    QUICK_MODE = config.getoption("--mode") == "quick"
    RECORD_JSON = config.getoption("--record") == "json"
    if RECORD_JSON:
        REPORT_FILE = config.getoption("--output") or REPORT_FILE


def _params_for_item(item) -> dict[str, object]:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return {}
    return dict(callspec.params)


def _operator_marks(item) -> list[str]:
    return [mark.name for mark in item.iter_markers() if mark.name not in BUILTIN_MARKS]


def _reason_from_report(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if hasattr(longrepr, "reprcrash"):
        return longrepr.reprcrash.message
    if isinstance(longrepr, tuple):
        return str(longrepr[2])
    return str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not RECORD_JSON:
        return

    result = TEST_RESULTS.setdefault(
        item.nodeid,
        {
            "params": _params_for_item(item),
            "result": None,
            "opname": _operator_marks(item),
        },
    )

    if report.when == "setup" and report.outcome == "skipped":
        result["result"] = "skipped"
        result["reason"] = _reason_from_report(report)
    elif report.when == "call":
        result["result"] = report.outcome
        result["reason"] = (
            _reason_from_report(report) if report.outcome != "passed" else None
        )
    elif report.when == "teardown" and report.outcome == "failed":
        result["result"] = "failed"
        result["reason"] = _reason_from_report(report)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not RECORD_JSON:
        return
    path = Path(REPORT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if path.exists() and path.stat().st_size:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            existing = {}
    existing.update(TEST_RESULTS)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def pytest_collection_modifyitems(session, config, items):
    if not config.getoption("--collect-marks"):
        return
    report = []
    for item in items:
        report.append(
            {
                "file": item.location[0],
                "test_case": item.name,
                "function": item.originalname,
                "marks": _operator_marks(item),
            }
        )
    print(json.dumps(report, indent=2, default=str))
    items.clear()
