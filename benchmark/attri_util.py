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

"""Default benchmark shape grids."""

CORE_SHAPES = {
    "small": [(32, 32), (64, 64)],
    "medium": [(256, 256), (512, 512)],
    "large": [(1024, 1024), (2048, 2048)],
}

SPMV_SHAPES = {
    "small": [(32, 32, 128)],
    "medium": [(512, 512, 4096)],
    "large": [(4096, 4096, 32768)],
}

SPMM_SHAPES = {
    "small": [(32, 32, 16, 128)],
    "medium": [(512, 512, 64, 4096)],
    "large": [(4096, 4096, 128, 32768)],
}

DEFAULT_DTYPES = ("float32", "float64")
