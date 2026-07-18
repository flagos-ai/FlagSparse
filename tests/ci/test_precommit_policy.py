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

"""Static policy checks for repository pre-commit wiring."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_precommit_config_exists():
    path = PROJECT_ROOT / ".pre-commit-config.yaml"
    assert path.is_file()


def test_makefile_runs_precommit():
    text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "pre-commit run --all-files" in text
