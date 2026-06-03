"""Shape / dtype grids for parametrized pytest accuracy suites."""

import torch

from tests.pytest.conftest import QUICK_MODE


if QUICK_MODE:
    SPMV_MN_SHAPES = [
        (1, 32),
    ]
    MNK_SHAPES = [
        (8, 4, 16),
    ]
    GATHER_SCATTER_SHAPES = [
        (512, 128),
    ]
    SPSV_N = [16]
    SPSM_N_RHS = [(16, 4)]
    SPGEMM_MNK_SHAPES = [(8, 10, 6)]
    SDDMM_MNK_SHAPES = [(8, 10, 8)]
else:
    SPMV_MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (128, 256),
    ]
    MNK_SHAPES = [
        (8, 4, 16),
        (64, 16, 96),
    ]
    GATHER_SCATTER_SHAPES = [
        (1024, 256),
        (4096, 512),
    ]
    SPSV_N = [16, 64]
    SPSM_N_RHS = [(16, 4), (64, 8)]
    SPGEMM_MNK_SHAPES = [(8, 10, 6), (48, 64, 32)]
    SDDMM_MNK_SHAPES = [(8, 10, 8), (64, 96, 16)]


def _bf16_ok():
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(fn()) if callable(fn) else False


_PRIMARY_FLOAT = [torch.float16, torch.float32]
FLOAT_DTYPES = _PRIMARY_FLOAT + ([torch.bfloat16] if _bf16_ok() else []) + [torch.float64]


def _dtype_node_id(value):
    """Short name for pytest node IDs (e.g. ``torch.float32`` -> ``float32``)."""
    return str(value).replace("torch.", "")


FLOAT_DTYPE_IDS = [_dtype_node_id(d) for d in FLOAT_DTYPES]
SPMV_COO_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
SPMV_COO_DTYPE_IDS = ("float32", "float64", "complex64", "complex128")
SPMM_FLOAT_DTYPES = FLOAT_DTYPES
SPMM_FLOAT_DTYPE_IDS = FLOAT_DTYPE_IDS
SPMM_OPT_DTYPES = (torch.float32, torch.float64)
SPMM_OPT_DTYPE_IDS = ("float32", "float64")
TRIANGULAR_DTYPES = (torch.float32, torch.float64)
TRIANGULAR_DTYPE_IDS = ("float32", "float64")
SPGEMM_DTYPES = (torch.float32, torch.float64)
SPGEMM_DTYPE_IDS = ("float32", "float64")
SDDMM_DTYPES = (torch.float32, torch.float64)
SDDMM_DTYPE_IDS = ("float32", "float64")
