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

"""CPU-only checks for the public package surface."""

import flagsparse

EXPECTED_TOP_LEVEL = {
    "flagsparse_gather",
    "flagsparse_scatter",
    "flagsparse_spmv_csr",
    "flagsparse_spmv_csr_run",
    "list_spmv_csr_algorithms",
    "get_spmv_csr_algorithm_spec",
    "flagsparse_spmv_coo",
    "flagsparse_spmv_csc",
    "flagsparse_spmv_bsr",
    "prepare_spmv_csc",
    "prepare_spmv_bsr",
    "flagsparse_spmm_csr",
    "flagsparse_spmm_csr_run",
    "prepare_spmm_csr_route",
    "list_spmm_csr_algorithms",
    "get_spmm_csr_algorithm_spec",
    "flagsparse_spmm_coo",
    "flagsparse_spmm_bsr",
    "flagsparse_spmm_bell",
    "flagsparse_spmm_csc",
    "prepare_spmm_bsr_route",
    "prepare_spmm_bell_route",
    "prepare_spmm_csc_route",
    "flagsparse_spgemm_csr",
    "flagsparse_sddmm_csr",
    "flagsparse_spsv_csr",
    "flagsparse_spsv_coo",
    "flagsparse_spsv_sell",
    "flagsparse_spsv_analysis_sell",
    "flagsparse_spsv_solve_sell",
    "flagsparse_spsm_csr",
    "flagsparse_spsm_coo",
    "create_csr_matrix",
    "create_coo_matrix",
    "read_mtx_file",
}


def test_public_surface_exposes_core_entry_points():
    exported = set(dir(flagsparse))
    assert EXPECTED_TOP_LEVEL <= exported
    assert EXPECTED_TOP_LEVEL <= set(flagsparse.__all__)



def test_all_registered_spmv_spmm_routes_declare_float16():
    import torch
    for operator in ("spmv", "spmm"):
        for storage in ("csr", "coo", "csc"):
            query = getattr(flagsparse, f"list_{operator}_{storage}_algorithms")
            for op in ("non", "trans", "conj"):
                all_routes = set(query(op=op))
                half_routes = set(query(op=op, dtype=torch.float16))
                assert half_routes == all_routes, (operator, storage, op, all_routes - half_routes)



def test_coo_csc_p2_public_algorithm_queries():
    import torch
    from flagsparse import sparse_operations
    expected = {
        "spmv_coo": ("coo_segmented_stream_atomic",),
        "spmm_coo": ("coo_segmented_stream_panel_atomic",),
        "spmv_csc": ("csc_col_vector", "csc_col_split_reduce"),
        "spmm_csc": ("csc_col_kparallel", "csc_col_split_reduce"),
    }
    for operator, names in expected.items():
        for namespace in (flagsparse, sparse_operations):
            query = getattr(namespace, f"list_{operator}_algorithms")
            spec = getattr(namespace, f"get_{operator}_algorithm_spec")
            for name in names:
                assert name in query(op="trans", dtype=torch.float16)
                assert spec(name)["name"] == name
                if operator.endswith("csc"):
                    assert name not in query(op="non")
