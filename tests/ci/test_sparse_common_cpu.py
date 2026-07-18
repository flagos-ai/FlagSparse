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

"""CPU-only smoke coverage for the shared sparse runtime helpers."""

import os

import pytest

if os.environ.get("FLAGSPARSE_TRITON_SMOKE") != "1":
    pytest.skip(
        "triton smoke is opt-in and excluded from CPU-only CI", allow_module_level=True
    )

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import _common  # noqa: E402


def test_common_module_exports_are_available():
    assert _common.SUPPORTED_INDEX_DTYPES == (torch.int32, torch.int64)
    assert _common._INDEX_LIMIT_INT32 == 2**31 - 1
    if _common.cp is None:
        assert _common.cpx_sparse is None
    else:
        assert _common.cpx_sparse is not None


def test_common_random_dense_helper_runs_on_cpu():
    dense = _common._build_random_dense(8, torch.float32, torch.device("cpu"))
    assert dense.device.type == "cpu"
    assert dense.shape == (8,)
