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
