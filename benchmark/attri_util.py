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
