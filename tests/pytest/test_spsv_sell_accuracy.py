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

import pytest
import torch

from flagsparse import (
    flagsparse_spsv_analysis_sell,
    flagsparse_spsv_create_workspace,
    flagsparse_spsv_sell,
    flagsparse_spsv_solve_sell,
)
from flagsparse.sparse_operations import spsv as spsv_impl
from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import SPSV_N, TRIANGULAR_DTYPES, TRIANGULAR_DTYPE_IDS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _lower_triangular_csr(n, dtype, index_dtype, device):
    dense_cpu = torch.zeros((n, n), dtype=dtype)
    row_ptr = [0]
    cols = []
    values = []
    for row in range(n):
        for col in range(row + 1):
            if row == col:
                value = 2.0 + 0.05 * row
            else:
                value = 0.01 * ((row + col) % 7 + 1)
            dense_cpu[row, col] = value
            cols.append(col)
            values.append(value)
        row_ptr.append(len(cols))
    return (
        torch.tensor(values, dtype=dtype, device=device),
        torch.tensor(cols, dtype=index_dtype, device=device),
        torch.tensor(row_ptr, dtype=index_dtype, device=device),
        dense_cpu.to(device=device),
    )


def _csr_to_sell(values, cols, row_ptr, n_rows, slice_size):
    slice_size = int(slice_size)
    n_slices = (n_rows + slice_size - 1) // slice_size
    widths = []
    for slice_id in range(n_slices):
        row0 = slice_id * slice_size
        row1 = min(row0 + slice_size, n_rows)
        widths.append(
            max(
                int(row_ptr[row + 1].item() - row_ptr[row].item())
                for row in range(row0, row1)
            )
        )

    offsets = torch.zeros(n_slices + 1, dtype=row_ptr.dtype, device=row_ptr.device)
    if widths:
        increments = torch.tensor(
            [width * slice_size for width in widths],
            dtype=row_ptr.dtype,
            device=row_ptr.device,
        )
        offsets[1:] = torch.cumsum(increments, dim=0)

    padded_size = int(offsets[-1].item())
    sell_values = torch.zeros(padded_size, dtype=values.dtype, device=values.device)
    sell_cols = torch.full((padded_size,), -1, dtype=cols.dtype, device=cols.device)
    for slice_id in range(n_slices):
        row0 = slice_id * slice_size
        row1 = min(row0 + slice_size, n_rows)
        base = int(offsets[slice_id].item())
        for row in range(row0, row1):
            start = int(row_ptr[row].item())
            end = int(row_ptr[row + 1].item())
            count = end - start
            dst = base + torch.arange(count, device=values.device) * slice_size + row - row0
            sell_values[dst] = values[start:end]
            sell_cols[dst] = cols[start:end]
    return sell_values, sell_cols, offsets


def _reference_solve(dense, b, dtype):
    ref_dtype = torch.float64 if dtype == torch.float32 else dtype
    ref = torch.linalg.solve_triangular(
        dense.to(ref_dtype),
        b.to(ref_dtype).unsqueeze(1),
        upper=False,
    ).squeeze(1)
    return ref.to(dtype)


@pytest.mark.spsv_sell
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRIANGULAR_DTYPES, ids=TRIANGULAR_DTYPE_IDS)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("slice_size", [8, 32], ids=["slice8", "slice32"])
@pytest.mark.parametrize("alg_num", [1, 2], ids=["alg1", "alg2"])
def test_spsv_sell_matches_dense_and_descriptor_api(n, dtype, index_dtype, slice_size, alg_num):
    device = torch.device("cuda")
    values, cols, row_ptr, dense = _lower_triangular_csr(n, dtype, index_dtype, device)
    sell_values, sell_cols, offsets = _csr_to_sell(values, cols, row_ptr, n, slice_size)
    b = torch.linspace(0.25, 1.25, n, dtype=dtype, device=device)
    expected = _reference_solve(dense, b, dtype)

    direct = flagsparse_spsv_sell(
        sell_values,
        sell_cols,
        offsets,
        b,
        (n, n),
        slice_size=slice_size,
        alg_num=alg_num,
    )
    descr = flagsparse_spsv_analysis_sell(
        sell_values,
        sell_cols,
        offsets,
        (n, n),
        slice_size=slice_size,
        alg_num=alg_num,
    )
    workspace = flagsparse_spsv_create_workspace(descr)
    via_descr = flagsparse_spsv_solve_sell(descr, b, workspace=workspace)

    rtol, atol = close_tolerances(dtype)
    assert torch.allclose(direct, expected, rtol=rtol, atol=atol)
    assert torch.allclose(via_descr, expected, rtol=rtol, atol=atol)


@pytest.mark.spsv_sell
def test_spsv_sell_alg_num_contract():
    assert spsv_impl._normalize_spsv_sell_alg_num(1) == 1
    assert spsv_impl._normalize_spsv_sell_alg_num(2) == 2
    assert spsv_impl._resolve_spsv_sell_alg2_worker_count(3172) == 64
    assert spsv_impl._resolve_spsv_sell_alg2_worker_count(100, requested=48) == 48
    with pytest.raises(ValueError, match="alg_num must be 1 or 2"):
        spsv_impl._normalize_spsv_sell_alg_num(3)
    with pytest.raises(ValueError, match="worker count must be positive"):
        spsv_impl._resolve_spsv_sell_alg2_worker_count(100, requested=0)
