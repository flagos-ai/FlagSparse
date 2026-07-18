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

"""Pytest entry points for FlagSparse performance suites.

These tests are intentionally opt-in. Run them on a GPU machine with, for example:

    pytest benchmark/test_sparse_perf.py -m spmv_perf --record log
"""

import pytest


pytestmark = pytest.mark.performance


@pytest.mark.spmv_perf
def test_spmv_perf_placeholder():
    pytest.skip("SpMV performance suite is opt-in and requires CUDA benchmark inputs.")


@pytest.mark.spmm_perf
def test_spmm_perf_placeholder():
    pytest.skip("SpMM performance suite is opt-in and requires CUDA benchmark inputs.")
