"""Shape / dtype grids for parametrized tests (gather/scatter, CSR+COO SpMV, CSR SpSV)."""

import torch

from tests.pytest.conftest import QUICK_MODE


if QUICK_MODE:
    SPMV_MN_SHAPES = [
        (1, 32),
    ]
    GATHER_SCATTER_SHAPES = [
        (512, 128),
    ]
    SPSV_N = [16]
else:
    SPMV_MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (128, 256),
    ]
    GATHER_SCATTER_SHAPES = [
        (1024, 256),
        (4096, 512),
    ]
    SPSV_N = [16, 64]


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
SPMV_COO_DTYPES = (torch.float32, torch.float64)
SPMV_COO_DTYPE_IDS = ("float32", "float64")
