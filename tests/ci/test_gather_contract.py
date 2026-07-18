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

"""Static contract checks for the gather benchmark entrypoint."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PATH = (
    PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "benchmarks.py"
)
GATHER_TEST_PATH = PROJECT_ROOT / "tests" / "test_gather.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _benchmark_gather_parameters():
    for node in _tree(BENCHMARKS_PATH).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "benchmark_gather_case":
                return {arg.arg for arg in node.args.args}
    raise AssertionError("benchmark_gather_case definition not found")


def _gather_test_keywords():
    keywords = set()
    for node in ast.walk(_tree(GATHER_TEST_PATH)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "ast"
            and func.attr == "benchmark_gather_case"
        ):
            keywords.update(keyword.arg for keyword in node.keywords if keyword.arg)
    assert keywords, "tests/test_gather.py does not call ast.benchmark_gather_case"
    return keywords


def test_gather_cli_keywords_match_benchmark_signature():
    assert _gather_test_keywords() <= _benchmark_gather_parameters()


def test_gather_csv_rows_cover_all_summary_fields():
    tree = _tree(GATHER_TEST_PATH)
    summary_fields = None
    row_key_sets = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "summary_fields" and isinstance(node.value, ast.List):
            summary_fields = {
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
        if target.id == "row" and isinstance(node.value, ast.Dict):
            keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if "case_id" in keys and "status" in keys:
                row_key_sets.append(keys)

    assert summary_fields, "summary_fields list not found"
    assert len(row_key_sets) == 2, "expected success and error summary row dictionaries"
    for row_keys in row_key_sets:
        assert summary_fields <= row_keys
