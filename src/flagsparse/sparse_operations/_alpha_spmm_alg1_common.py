"""Internal launch helpers for the experimental AlphaSparse CSR ALG1 port."""

from ._common import triton


def _select_alpha_spmm_alg1_warp_and_factor(n_dense_cols, order_row=True):
    """Mirror AlphaSparse csrspmm_rb_sr warp/factor selection."""
    n_dense_cols = int(n_dense_cols)
    if order_row:
        if n_dense_cols > 64:
            return 32, 4
        if n_dense_cols > 32:
            return 32, 2
        if n_dense_cols > 16:
            return 32, 1
        if n_dense_cols > 8:
            return 16, 1
        if n_dense_cols > 4:
            return 8, 1
        return 4, 1
    if n_dense_cols > 4:
        return 8, 1
    return 4, 1


def _build_alpha_spmm_alg1_launch_meta(
    n_rows,
    n_dense_cols,
    warp_size,
    factor,
    block_size=256,
):
    """Build the AlphaSparse ALG1 block geometry for row-major dense B/C."""
    block_size = int(block_size)
    warp_size = int(warp_size)
    factor = int(factor)
    block_rows = max(1, block_size // warp_size)
    block_cols = warp_size * factor
    return {
        "block_size": block_size,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "grid_m": int(triton.cdiv(int(n_rows), block_rows)),
        "grid_n": int(triton.cdiv(int(n_dense_cols), block_cols)),
    }
