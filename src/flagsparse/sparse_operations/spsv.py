"""Sparse triangular solve (SpSV) CSR/COO."""

from ._common import *

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
import time
import triton
import triton.language as tl

SUPPORTED_SPSV_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SUPPORTED_SPSV_INDEX_DTYPES = (torch.int32, torch.int64)
SPSV_NON_TRANS_SUPPORTED_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.int32),
    (torch.complex128, torch.int32),
    (torch.float32, torch.int64),
    (torch.float64, torch.int64),
    (torch.complex64, torch.int64),
    (torch.complex128, torch.int64),
)
SPSV_TRANS_SUPPORTED_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.int32),
    (torch.complex128, torch.int32),
    (torch.float32, torch.int64),
    (torch.float64, torch.int64),
    (torch.complex64, torch.int64),
    (torch.complex128, torch.int64),
)
def _spsv_env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


SPSV_PROMOTE_FP32_TO_FP64 = _spsv_env_flag("FLAGSPARSE_SPSV_PROMOTE_FP32_TO_FP64", "0")
SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64", "0"
)
SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128", "0"
)
_SPSV_CSR_PREPROCESS_CACHE = OrderedDict()
_SPSV_CSR_PREPROCESS_CACHE_SIZE = 8


@dataclass
class FlagSparseSpSVDescr:
    """Host-side analysis handle for Triton SpSV.

    This is the Triton/Python equivalent of the CUDA-side SpSV descriptor:
    it stores the analyzed matrix metadata, the selected solve route, and the
    workspace layout needed by the current implementation.
    """

    format: str
    canonical_format: str
    shape: tuple
    lower: bool
    unit_diagonal: bool
    fill_mode: str
    diag_type: str
    matrix_type: str
    index_base: int
    transpose_mode: str
    value_dtype: torch.dtype
    compute_dtype: torch.dtype
    index_dtype: torch.dtype
    solve_kind: str
    route_name: str
    storage_view: str
    buffer_size: int
    workspace_layout: tuple
    data: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    indptr: torch.Tensor = field(repr=False)
    solve_plan: dict = field(repr=False)


@dataclass
class FlagSparseSpSVWorkspace:
    """Caller-owned workspace object for Triton SpSV host APIs."""

    buffer_size: int
    layout: tuple
    device: torch.device
    buffers: dict = field(default_factory=dict, repr=False)
    prepared_solve_kind: str = ""
    prepared_signature: tuple | None = None


@dataclass
class FlagSparseSpSVHandle:
    """Host-side execution handle for Triton SpSV."""

    device: torch.device
    stream: object = None


@dataclass
class FlagSparseSpMatDescr:
    """Sparse matrix descriptor mirroring the CUDA SpMat inputs."""

    format: str
    shape: tuple
    values: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    indptr_or_col: torch.Tensor = field(repr=False)
    lower: bool = True
    unit_diagonal: bool = False
    diag_type: str = "non_unit"
    fill_mode: str = "lower"
    matrix_type: str = "triangular"
    index_base: int = 0


@dataclass
class FlagSparseDnVecDescr:
    """Dense vector descriptor mirroring the CUDA DnVec inputs."""

    values: torch.Tensor = field(repr=False)


def flagsparse_create_spsv_handle(device=None, stream=None):
    if device is None:
        device = torch.device("cuda")
    return FlagSparseSpSVHandle(device=torch.device(device), stream=stream)


def flagsparse_create_dnvec(values):
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 1:
        raise ValueError("DnVec values must be 1D")
    return FlagSparseDnVecDescr(values=values)


def flagsparse_create_spmat_csr(
    values,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    matrix_type="triangular",
    index_base=0,
):
    return FlagSparseSpMatDescr(
        format="csr",
        shape=(int(shape[0]), int(shape[1])),
        values=values,
        indices=indices,
        indptr_or_col=indptr,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        diag_type="unit" if unit_diagonal else "non_unit",
        fill_mode="lower" if lower else "upper",
        matrix_type=str(matrix_type),
        index_base=int(index_base),
    )


def flagsparse_create_spmat_coo(
    values,
    row,
    col,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    matrix_type="triangular",
    index_base=0,
):
    return FlagSparseSpMatDescr(
        format="coo",
        shape=(int(shape[0]), int(shape[1])),
        values=values,
        indices=row,
        indptr_or_col=col,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        diag_type="unit" if unit_diagonal else "non_unit",
        fill_mode="lower" if lower else "upper",
        matrix_type=str(matrix_type),
        index_base=int(index_base),
    )


def _clear_spsv_csr_preprocess_cache():
    _SPSV_CSR_PREPROCESS_CACHE.clear()


def _as_strided_contiguous(tensor):
    if tensor is None:
        return None
    if tensor.layout != torch.strided:
        out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        out.copy_(tensor)
        return out
    return tensor.contiguous()


def _complex_interleaved_view(tensor):
    tensor_strided = _as_strided_contiguous(tensor)
    return torch.view_as_real(tensor_strided).reshape(-1).contiguous()


def _attach_spsv_complex_plan_views(plan):
    kernel_data = plan.get("kernel_data")
    if kernel_data is None or not torch.is_complex(kernel_data):
        return plan
    plan["kernel_data_ri"] = _complex_interleaved_view(kernel_data)
    return plan


def _validate_spsv_non_trans_combo(data_dtype, index_dtype, fmt_name):
    """Validate NON_TRANS support matrix and keep error messages explicit."""
    if (data_dtype, index_dtype) in SPSV_NON_TRANS_SUPPORTED_COMBOS:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports NON_TRANS combinations: "
        "(float32, int32/int64), (float64, int32/int64), "
        "(complex64, int32/int64), (complex128, int32/int64)"
    )


def _validate_spsv_trans_combo(data_dtype, index_dtype, fmt_name):
    if (data_dtype, index_dtype) in SPSV_TRANS_SUPPORTED_COMBOS:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports TRANS/CONJ combinations: "
        "(float32, int32/int64), (float64, int32/int64), "
        "(complex64, int32/int64), (complex128, int32/int64)"
    )


def _normalize_spsv_transpose_mode(transpose):
    if isinstance(transpose, bool):
        return "T" if transpose else "N"
    token = str(transpose).strip().upper()
    if token in ("N", "NON", "NON_TRANS"):
        return "N"
    if token in ("T", "TRANS"):
        return "T"
    if token in ("C", "H", "CONJ", "CONJ_TRANS", "CONJUGATE_TRANSPOSE"):
        return "C"
    raise ValueError(
        "transpose must be bool or one of: "
        "N/NON/NON_TRANS, T/TRANS, C/H/CONJ/CONJ_TRANS/CONJUGATE_TRANSPOSE"
    )


def _prepare_spsv_inputs(data, indices, indptr, b, shape):
    """Validate and normalize inputs for sparse solve A x = b with CSR A."""
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, b)):
        raise TypeError("data, indices, indptr, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, b)):
        raise ValueError("data, indices, indptr, b must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")

    if data.dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")

    indices64 = indices.to(torch.int64).contiguous()
    indptr64 = indptr.to(torch.int64).contiguous()
    if indices64.numel() > 0 and int(indices64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(indices64.max().item())} exceeds Triton int32 kernel range"
        )
    if indptr64.numel() > 0:
        if int(indptr64[0].item()) != 0:
            raise ValueError("indptr[0] must be 0")
        if int(indptr64[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
            raise ValueError("indptr must be non-decreasing")
    if indices64.numel() > 0:
        if bool(torch.any(indices64 < 0).item()):
            raise IndexError("indices must be non-negative")
        max_idx = int(indices64.max().item())
        if max_idx >= n_cols:
            raise IndexError(f"indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        indices.dtype,
        indices64,
        indptr64,
        b.contiguous(),
        n_rows,
        n_cols,
    )

def _spsv_diag_eps_for_dtype(value_dtype):
    return 1e-12 if value_dtype in (torch.float64, torch.complex128) else 1e-6


def _tensor_cache_token(tensor):
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        int(tensor.numel()),
        storage_ptr,
        int(getattr(tensor, "_version", 0)),
    )


def _spsv_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsv_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _normalize_spsv_format(fmt):
    token = str(fmt).strip().lower()
    if token not in ("csr", "coo"):
        raise ValueError("format must be 'csr' or 'coo'")
    return token


def _normalize_spsv_storage_view(storage_view):
    if storage_view is None:
        return "csr_as_csc"
    token = str(storage_view).strip().lower()
    aliases = {
        "csr_as_csc": "csr_as_csc",
        "csc_view": "csr_as_csc",
        "reuse_csr_storage": "csr_as_csc",
    }
    if token not in aliases:
        raise ValueError(
            "storage_view must be one of: csr_as_csc, csc_view, reuse_csr_storage"
        )
    return aliases[token]


def _resolve_spsv_stream(handle, stream, device):
    resolved = stream
    if handle is not None:
        if not isinstance(handle, FlagSparseSpSVHandle):
            raise TypeError("handle must be a FlagSparseSpSVHandle or None")
        if torch.device(handle.device) != torch.device(device):
            raise ValueError("handle device must match the solve device")
        if resolved is None:
            resolved = handle.stream
    return resolved


def _coerce_spsv_alpha(alpha, dtype, device):
    if torch.is_tensor(alpha):
        alpha_tensor = alpha.to(device=device, dtype=dtype).reshape(-1)
        if alpha_tensor.numel() != 1:
            raise ValueError("alpha must be a scalar tensor")
        return alpha_tensor.reshape(())
    return torch.tensor(alpha, device=device, dtype=dtype)


def _workspace_entry(name, numel, dtype):
    return {
        "name": str(name),
        "numel": int(numel),
        "dtype": dtype,
        "bytes": int(numel) * int(torch.empty((), dtype=dtype).element_size()),
    }


def _workspace_size_bytes(layout):
    return int(sum(int(entry["bytes"]) for entry in layout))


def _spsv_effective_compute_dtype(value_dtype, trans_mode, compute_dtype=None):
    if compute_dtype is not None:
        if compute_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
            raise TypeError(
                "compute_dtype must be one of: float32, float64, complex64, complex128"
            )
        return compute_dtype
    if (
        value_dtype == torch.complex64
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128
    ):
        return torch.complex128
    if value_dtype == torch.float32 and SPSV_PROMOTE_FP32_TO_FP64:
        return torch.float64
    if (
        value_dtype == torch.float32
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64
    ):
        return torch.float64
    return value_dtype


def _build_spsv_workspace_layout(n_rows, solve_kind, value_dtype=None):
    n_rows = int(n_rows)
    if solve_kind == "csr_cw":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    if solve_kind == "transpose_cw":
        if value_dtype is None:
            raise ValueError("value_dtype is required for transpose_cw workspace sizing")
        return (
            _workspace_entry("residual", n_rows, value_dtype),
            _workspace_entry("indegree", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    raise ValueError(f"Unsupported SpSV solve kind for workspace sizing: {solve_kind}")


def _clone_spsv_plan(plan):
    cloned = dict(plan)
    matrix_stats = plan.get("matrix_stats")
    if matrix_stats is not None:
        cloned["matrix_stats"] = dict(matrix_stats)
    return cloned


def _alloc_spsv_workspace_buffers(layout, device):
    buffers = {}
    for entry in layout:
        buffers[entry["name"]] = torch.empty(
            int(entry["numel"]), dtype=entry["dtype"], device=device
        )
    return buffers


def _resolve_spsv_workspace(workspace, layout, device):
    if workspace is None:
        return _alloc_spsv_workspace_buffers(layout, device)
    if not isinstance(workspace, FlagSparseSpSVWorkspace):
        raise TypeError("workspace must be a FlagSparseSpSVWorkspace or None")
    if torch.device(workspace.device) != torch.device(device):
        raise ValueError("workspace device must match the solve device")
    if int(workspace.buffer_size) < _workspace_size_bytes(layout):
        raise ValueError("workspace buffer is smaller than the required SpSV size")
    required = {entry["name"]: entry for entry in layout}
    for name, entry in required.items():
        buf = workspace.buffers.get(name)
        if buf is None:
            workspace.buffers[name] = torch.empty(
                int(entry["numel"]), dtype=entry["dtype"], device=device
            )
            continue
        if buf.device != torch.device(device):
            raise ValueError(f"workspace buffer {name!r} is on the wrong device")
        if buf.dtype != entry["dtype"] or int(buf.numel()) < int(entry["numel"]):
            workspace.buffers[name] = torch.empty(
                int(entry["numel"]), dtype=entry["dtype"], device=device
            )
    return workspace.buffers


def _transpose_cw_preprocess_signature(
    solve_plan, n_rows, unit_diagonal, block_nnz_use, max_segments_use
):
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    return (
        "transpose_cw",
        int(n_rows),
        bool(solve_plan["lower_eff"]),
        bool(unit_diagonal),
        int(block_nnz_use),
        int(max_segments_use),
        _tensor_cache_token(kernel_indices32),
        _tensor_cache_token(kernel_indptr64),
    )


def flagsparse_spsv_buffer_size(
    shape,
    value_dtype,
    *,
    format="csr",
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    alpha=None,
    handle=None,
    vecX=None,
    vecY=None,
    storage_view="csr_as_csc",
):
    """Return the caller-managed workspace size for the current Triton SpSV route.

    This is the Triton host-side equivalent of the CUDA bufferSize query.
    The returned byte count matches the scratch buffers used by the current
    Triton implementation, rather than the raw CUDA ABI layout.
    """

    fmt = _normalize_spsv_format(format)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSV expects a square matrix, got shape={shape}")
    if value_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "value_dtype must be one of: float32, float64, complex64, complex128"
        )
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    storage_view = _normalize_spsv_storage_view(storage_view)
    compute_dtype = _spsv_effective_compute_dtype(
        value_dtype, trans_mode, compute_dtype=compute_dtype
    )
    route = _normalize_requested_spsv_route(solve_kind, trans_mode)
    if route is None:
        route = "transpose_cw" if trans_mode in ("T", "C") else "csr_cw"
    if trans_mode in ("T", "C") and storage_view != "csr_as_csc":
        raise ValueError("TRANS/CONJ SpSV only supports storage_view='csr_as_csc'")
    layout = _build_spsv_workspace_layout(n_rows, route, value_dtype=compute_dtype)
    return _workspace_size_bytes(layout)


def flagsparse_spsv_create_workspace(descr, device=None):
    """Allocate a caller-owned SpSV workspace object from an analysis descriptor."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if device is None:
        device = descr.data.device
    device = torch.device(device)
    buffers = _alloc_spsv_workspace_buffers(descr.workspace_layout, device)
    return FlagSparseSpSVWorkspace(
        buffer_size=int(descr.buffer_size),
        layout=tuple(descr.workspace_layout),
        device=device,
        buffers=buffers,
    )


def _csr_preprocess_cache_key(
    data, indices, indptr, shape, lower, trans_mode, unit_diagonal, storage_view="csr_as_csc"
):
    return (
        "csr_preprocess",
        trans_mode,
        bool(lower),
        bool(unit_diagonal),
        str(storage_view),
        int(shape[0]),
        int(shape[1]),
        _tensor_cache_token(data),
        _tensor_cache_token(indices),
        _tensor_cache_token(indptr),
    )


def _normalize_requested_spsv_route(solve_kind, trans_mode):
    if solve_kind is None:
        return None
    token = str(solve_kind).strip().lower()
    aliases = {
        "csr_cw": "csr_cw",
        "cw": "csr_cw" if trans_mode == "N" else "transpose_cw",
        "transpose_cw": "transpose_cw",
        "csc_cw": "transpose_cw",
    }
    route = aliases.get(token)
    if route is None:
        raise ValueError("solve_kind must be one of: csr_cw, transpose_cw")
    if trans_mode == "N" and route != "csr_cw":
        raise ValueError("NON_TRANS SpSV only supports solve_kind='csr_cw'")
    if trans_mode in ("T", "C") and route == "csr_cw":
        raise ValueError("TRANS/CONJ SpSV cannot use solve_kind='csr_cw'")
    return route


@triton.jit
def _spsv_csc_preprocess_kernel(
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_rows:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        row = tl.load(indices_ptr + offsets, mask=mask, other=0)
        if LOWER:
            dep_mask = mask & (row > col if UNIT_DIAG else row >= col)
        else:
            dep_mask = mask & (row < col if UNIT_DIAG else row <= col)
        tl.atomic_add(indegree_ptr + row, 1, mask=dep_mask)


def _sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=True):
    if data.numel() == 0:
        return data, indices64, indptr64
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    key = row_ids * max(1, n_cols)
    if lower:
        key = key + indices64
    else:
        key = key + (n_cols - 1 - indices64)
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order], indices64[order], indptr64


def _cw_rhs_bucket(n_rhs):
    if n_rhs <= 1:
        return 1
    if n_rhs <= 2:
        return 2
    if n_rhs <= 4:
        return 4
    if n_rhs <= 8:
        return 8
    if n_rhs <= 16:
        return 16
    return 32


def _snap_cw_worker_count(target, n_rows):
    if n_rows <= 0:
        return 1
    target = max(1, min(int(target), int(n_rows)))
    snapped = 1
    tier = 1
    while tier < target and tier < 4096:
        tier *= 2
        if tier <= target:
            snapped = tier
    return int(max(1, min(snapped, int(n_rows))))


def _cw_worker_count(n_rows, max_frontier, avg_nnz_per_row, n_rhs):
    if n_rows <= 0:
        return 1
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    if rhs_bucket == 1:
        target = min(n_rows, 32)
    else:
        target = max(32, min(n_rows, 512))
    if max_frontier > 0:
        target = min(target, max(4, min(n_rows, max_frontier * 2)))
    if avg_nnz_per_row > 8192:
        target = max(4, target // 8)
    elif avg_nnz_per_row > 4096:
        target = max(4, target // 4)
    elif avg_nnz_per_row > 2048:
        target = max(4, target // 2)
    elif avg_nnz_per_row > 1024:
        target = max(8, (target * 2) // 3)
    elif avg_nnz_per_row > 512:
        target = max(8, (target * 3) // 5)
    if rhs_bucket >= 16:
        target = max(4, target // 4)
    elif rhs_bucket >= 8:
        target = max(4, target // 2)
    elif rhs_bucket >= 4:
        target = max(4, (target * 3) // 4)
    return _snap_cw_worker_count(target, n_rows)


def _resolve_cw_worker_count(n_rows, matrix_stats, n_rhs, cached_worker_count=None):
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    max_frontier = int(matrix_stats.get("max_frontier", n_rows))
    avg_frontier = float(matrix_stats.get("avg_frontier", float(max_frontier)))
    frontier_ratio = float(matrix_stats.get("frontier_ratio", 1.0 if n_rows > 0 else 0.0))
    num_levels = int(matrix_stats.get("num_levels", 0))
    avg_nnz_per_row = float(matrix_stats.get("avg_nnz_per_row", 0.0))
    if cached_worker_count is not None and rhs_bucket == 1:
        target = int(max(1, min(int(cached_worker_count), int(max(n_rows, 1)))))
    else:
        target = _cw_worker_count(
            n_rows,
            max_frontier,
            avg_nnz_per_row,
            rhs_bucket,
        )
    if frontier_ratio < 0.01 or avg_frontier < 4.0:
        target = min(target, max(1, min(n_rows, 4)))
    elif frontier_ratio < 0.02 or avg_frontier < 8.0:
        target = min(target, max(1, min(n_rows, 8)))
    elif frontier_ratio < 0.05 or avg_frontier < 16.0:
        target = min(target, max(2, min(n_rows, 16)))
    if num_levels > max(1024, n_rows // 2):
        target = max(1, target // 2)
    if avg_nnz_per_row > 2048:
        target = max(1, target // 2)
    return _snap_cw_worker_count(
        target,
        n_rows,
    )


def _build_spsv_cw_matrix_stats(indptr64, n_rows):
    if indptr64.numel() <= 1:
        avg_nnz_per_row = 0.0
        max_nnz_per_row = 0
    else:
        row_lengths = indptr64[1:] - indptr64[:-1]
        avg_nnz_per_row = float(row_lengths.to(torch.float32).mean().item())
        max_nnz_per_row = int(row_lengths.max().item())
    return {
        "num_levels": 0,
        "max_frontier": int(n_rows),
        "avg_frontier": float(n_rows),
        "frontier_ratio": 1.0 if n_rows > 0 else 0.0,
        "avg_nnz_per_row": avg_nnz_per_row,
        "max_nnz_per_row": max_nnz_per_row,
        "n_rows": int(n_rows),
    }


def _prepare_spsv_csr_system(
    data,
    indices64,
    indptr64,
    n_rows,
    n_cols,
    lower,
    trans_mode,
    unit_diagonal,
    storage_view="csr_as_csc",
):
    if trans_mode == "N":
        data, indices64, indptr64 = _sort_csr_rows(
            data, indices64, indptr64, n_rows, n_cols, lower=lower
        )
        matrix_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
        default_block_nnz, default_max_segments = _auto_spsv_launch_config(indptr64)
        if lower:
            nontrans_variant = "csr_u_lo_cw" if unit_diagonal else "csr_n_lo_cw"
        else:
            nontrans_variant = "csr_u_up_cw" if unit_diagonal else "csr_n_up_cw"
        cw_plan = {
            "solve_kind": "csr_cw",
            "nontrans_variant": nontrans_variant,
            "kernel_data": data,
            "kernel_indices32": indices64.to(torch.int32),
            "kernel_indptr64": indptr64,
            "lower_eff": lower,
            "default_block_nnz": default_block_nnz,
            "default_max_segments": default_max_segments,
            "storage_view": "csr",
            "cw_worker_count": _cw_worker_count(
                n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
            ),
            "matrix_stats": matrix_stats,
            "route_name": nontrans_variant,
        }
        _attach_spsv_complex_plan_views(cw_plan)
        return cw_plan

    lower_eff = not lower
    storage_view = _normalize_spsv_storage_view(storage_view)
    if storage_view != "csr_as_csc":
        raise ValueError("TRANS/CONJ SpSV only supports storage_view='csr_as_csc'")
    data_eff = data
    indices_eff64 = indices64
    indptr_eff64 = indptr64
    matrix_stats = _build_spsv_cw_matrix_stats(indptr_eff64, n_rows)
    default_block_nnz, default_max_segments = _choose_transpose_family_launch_config(
        indptr_eff64
    )
    cw_plan = {
        "solve_kind": "transpose_cw",
        "kernel_data": data_eff,
        "kernel_indices32": indices_eff64.to(torch.int32),
        "kernel_indptr64": indptr_eff64,
        "lower_eff": lower_eff,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "cw_worker_count": _cw_worker_count(
            n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
        ),
        "matrix_stats": matrix_stats,
        "storage_view": storage_view,
        "route_name": "transpose_cw",
    }
    _attach_spsv_complex_plan_views(cw_plan)
    return cw_plan


def _resolve_spsv_csr_runtime(
    data,
    indices,
    indptr,
    b,
    shape,
    lower,
    transpose,
    unit_diagonal=False,
    storage_view="csr_as_csc",
):
    input_data = data
    input_indices = indices
    input_indptr = indptr
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    data, input_index_dtype, indices, indptr, b, n_rows, n_cols = _prepare_spsv_inputs(
        data, indices, indptr, b, shape
    )
    original_output_dtype = None
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "CSR")
    else:
        _validate_spsv_trans_combo(data.dtype, input_index_dtype, "CSR")

    preprocess_key = _csr_preprocess_cache_key(
        input_data,
        input_indices,
        input_indptr,
        (n_rows, n_cols),
        lower,
        trans_mode,
        unit_diagonal,
        _normalize_spsv_storage_view(storage_view),
    )
    cached = _spsv_cache_get(_SPSV_CSR_PREPROCESS_CACHE, preprocess_key)
    if cached is None:
        cached = _prepare_spsv_csr_system(
            data,
            indices,
            indptr,
            n_rows,
            n_cols,
            lower,
            trans_mode,
            unit_diagonal,
            storage_view=storage_view,
        )
        _spsv_cache_put(
            _SPSV_CSR_PREPROCESS_CACHE,
            preprocess_key,
            cached,
            _SPSV_CSR_PREPROCESS_CACHE_SIZE,
        )
    return (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        n_cols,
        cached,
    )


def _select_spsv_runtime_plan(solve_plan, trans_mode, requested_solve_kind=None):
    requested_route = _normalize_requested_spsv_route(requested_solve_kind, trans_mode)
    storage_view = str(solve_plan.get("storage_view", "csr"))
    if trans_mode in ("T", "C") and storage_view == "csr_as_csc":
        return _clone_spsv_plan(solve_plan)
    if trans_mode == "N" and requested_route in (None, "csr_cw"):
        return _clone_spsv_plan(solve_plan)
    if requested_route is not None:
        routed = _clone_spsv_plan(solve_plan)
        routed["solve_kind"] = requested_route
        routed["route_name"] = requested_route
        return routed
    return _clone_spsv_plan(solve_plan)


@triton.jit
def _publish_ready_flag_i32(flag_ptr, idx):
    """Approximate 'store result then publish ready' in Triton."""

    tl.atomic_add(flag_ptr + idx, 1)


@triton.jit
def _complex_atomic_add_interleaved(ptr_ri, idx, delta_re, delta_im, mask):
    """Complex atomicAdd equivalent for interleaved real/imag buffers."""

    tl.atomic_add(ptr_ri + idx * 2, delta_re, mask=mask)
    tl.atomic_add(ptr_ri + idx * 2 + 1, delta_im, mask=mask)


@triton.jit
def _propagate_real(residual_ptr, idx, delta, mask):
    """Publish a real contribution into shared residual state."""

    tl.atomic_add(residual_ptr + idx, delta, mask=mask)


@triton.jit
def _propagate_then_release_real(residual_ptr, indegree_ptr, idx, delta, mask):
    """Approximate 'write contribution then decrement dependency count'."""

    _propagate_real(residual_ptr, idx, delta, mask)
    tl.atomic_add(indegree_ptr + idx, -1, mask=mask)


@triton.jit
def _propagate_complex(residual_ri_ptr, idx, delta_re, delta_im, mask):
    """Publish a complex contribution into shared residual state."""

    _complex_atomic_add_interleaved(residual_ri_ptr, idx, delta_re, delta_im, mask)


@triton.jit
def _propagate_then_release_complex(residual_ri_ptr, indegree_ptr, idx, delta_re, delta_im, mask):
    """Complex propagation + dependency release for transpose-style solve."""

    _propagate_complex(residual_ri_ptr, idx, delta_re, delta_im, mask)
    tl.atomic_add(indegree_ptr + idx, -1, mask=mask)


@triton.jit
def _spsv_csr_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        ptr = start
        if USE_FP64_ACC:
            rhs = tl.load(b_ptr + row).to(tl.float64)
            tmp_sum = tl.zeros((), dtype=tl.float64)
        else:
            rhs = tl.load(b_ptr + row).to(tl.float32)
            tmp_sum = tl.zeros((), dtype=tl.float32)
        row_done = 0
        while row_done == 0:
            if UNIT_DIAG:
                if ptr >= end:
                    x_row = rhs - tmp_sum
                    x_row = tl.where(x_row == x_row, x_row, 0.0)
                    tl.store(x_ptr + row, x_row)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    stop_at_diag = (col >= row) if LOWER else (col <= row)
                    if stop_at_diag:
                        x_row = rhs - tmp_sum
                        x_row = tl.where(x_row == x_row, x_row, 0.0)
                        tl.store(x_ptr + row, x_row)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(data_ptr + ptr).to(tl.float64)
                            y_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(data_ptr + ptr).to(tl.float32)
                            y_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * y_dep
                        ptr += 1
            else:
                if ptr >= end:
                    x_row = rhs * 0
                    tl.store(x_ptr + row, x_row)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    if col == row:
                        if USE_FP64_ACC:
                            diag = tl.load(data_ptr + ptr).to(tl.float64)
                        else:
                            diag = tl.load(data_ptr + ptr).to(tl.float32)
                        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
                        x_row = (rhs - tmp_sum) / diag_safe
                        x_row = tl.where(x_row == x_row, x_row, 0.0)
                        tl.store(x_ptr + row, x_row)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(data_ptr + ptr).to(tl.float64)
                            y_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(data_ptr + ptr).to(tl.float32)
                            y_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * y_dep
                        ptr += 1
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    lane2 = tl.arange(0, 2)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        rhs_re = tl.load(b_ri_ptr + row * 2)
        rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
            tmp_re = tl.zeros((), dtype=tl.float64)
            tmp_im = tl.zeros((), dtype=tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)
            tmp_re = tl.zeros((), dtype=tl.float32)
            tmp_im = tl.zeros((), dtype=tl.float32)
        ptr = start
        row_done = 0
        while row_done == 0:
            if UNIT_DIAG:
                if ptr >= end:
                    x_re_out = rhs_re - tmp_re
                    x_im_out = rhs_im - tmp_im
                    x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                    x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                    out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                    tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    stop_at_diag = (col >= row) if LOWER else (col <= row)
                    if stop_at_diag:
                        x_re_out = rhs_re - tmp_re
                        x_im_out = rhs_im - tmp_im
                        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        a_re = tl.load(data_ri_ptr + ptr * 2)
                        a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                        else:
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                        tmp_re += a_re * x_re - a_im * x_im
                        tmp_im += a_re * x_im + a_im * x_re
                        ptr += 1
            else:
                if ptr >= end:
                    out_vals = tl.where(lane2 == 0, rhs_re * 0, rhs_im * 0)
                    tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    if col == row:
                        diag_re = tl.load(data_ri_ptr + ptr * 2)
                        diag_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            diag_re = diag_re.to(tl.float64)
                            diag_im = diag_im.to(tl.float64)
                        else:
                            diag_re = diag_re.to(tl.float32)
                            diag_im = diag_im.to(tl.float32)
                        num_re = rhs_re - tmp_re
                        num_im = rhs_im - tmp_im
                        den = diag_re * diag_re + diag_im * diag_im
                        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                        x_re_out = (num_re * diag_re + num_im * diag_im) / den_safe
                        x_im_out = (num_im * diag_re - num_re * diag_im) / den_safe
                        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        a_re = tl.load(data_ri_ptr + ptr * 2)
                        a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                        else:
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                        tmp_re += a_re * x_re - a_im * x_im
                        tmp_im += a_re * x_im + a_im * x_re
                        ptr += 1
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_transpose_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    residual_ptr,
    x_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        ready_value = 0 if UNIT_DIAG else 1
        dep_ready = tl.atomic_add(indegree_ptr + row, 0)
        while dep_ready != ready_value:
            dep_ready = tl.atomic_add(indegree_ptr + row, 0)

        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        rhs = tl.load(residual_ptr + row)
        if UNIT_DIAG:
            diag = rhs * 0 + 1.0
        else:
            diag = rhs * 0
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                offsets = idx + tl.arange(0, BLOCK_NNZ)
                mask = offsets < end
                a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
                dep_row = tl.load(indices_ptr + offsets, mask=mask, other=0)
                is_diag = dep_row == row
                diag = diag + tl.sum(tl.where(mask & is_diag, a, 0.0), axis=0)
        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
        x_row = rhs / diag_safe
        x_row = tl.where(x_row == x_row, x_row, 0.0)
        tl.store(x_ptr + row, x_row)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            if LOWER:
                target_mask = mask & (col > row)
            else:
                target_mask = mask & (col < row)
            _propagate_then_release_real(
                residual_ptr, indegree_ptr, col, -a * x_row, target_mask
            )
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_transpose_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    residual_ri_ptr,
    x_ri_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    CONJ_TRANS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    lane2 = tl.arange(0, 2)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        ready_value = 0 if UNIT_DIAG else 1
        dep_ready = tl.atomic_add(indegree_ptr + row, 0)
        while dep_ready != ready_value:
            dep_ready = tl.atomic_add(indegree_ptr + row, 0)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)

        rhs_re = tl.load(residual_ri_ptr + row * 2)
        rhs_im = tl.load(residual_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)

        if UNIT_DIAG:
            diag_re = rhs_re * 0 + 1.0
            diag_im = rhs_im * 0
        else:
            if USE_FP64_ACC:
                diag_re = tl.zeros((), dtype=tl.float64)
                diag_im = tl.zeros((), dtype=tl.float64)
            else:
                diag_re = tl.zeros((), dtype=tl.float32)
                diag_im = tl.zeros((), dtype=tl.float32)
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                offsets = idx + tl.arange(0, BLOCK_NNZ)
                mask = offsets < end
                dep_row = tl.load(indices_ptr + offsets, mask=mask, other=0)
                a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
                a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
                if CONJ_TRANS:
                    a_im = -a_im
                if USE_FP64_ACC:
                    a_re = a_re.to(tl.float64)
                    a_im = a_im.to(tl.float64)
                else:
                    a_re = a_re.to(tl.float32)
                    a_im = a_im.to(tl.float32)
                is_diag = dep_row == row
                diag_re = diag_re + tl.sum(tl.where(mask & is_diag, a_re, 0.0), axis=0)
                diag_im = diag_im + tl.sum(tl.where(mask & is_diag, a_im, 0.0), axis=0)

        den = diag_re * diag_re + diag_im * diag_im
        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
        x_re_out = (rhs_re * diag_re + rhs_im * diag_im) / den_safe
        x_im_out = (rhs_im * diag_re - rhs_re * diag_im) / den_safe
        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
            a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
            if CONJ_TRANS:
                a_im = -a_im
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
            if LOWER:
                target_mask = mask & (col > row)
            else:
                target_mask = mask & (col < row)
            prod_re = a_re * x_re_out - a_im * x_im_out
            prod_im = a_re * x_im_out + a_im * x_re_out
            _propagate_then_release_complex(
                residual_ri_ptr,
                indegree_ptr,
                col,
                -prod_re,
                -prod_im,
                target_mask,
            )
        logical_row = tl.atomic_add(row_counter_ptr, 1)


def _auto_spsv_launch_config(indptr, block_nnz=None, max_segments=None):
    if indptr.numel() <= 1:
        max_nnz_per_row = 0
    else:
        row_lengths = indptr[1:] - indptr[:-1]
        max_nnz_per_row = int(row_lengths.max().item())

    auto_block = block_nnz is None
    if block_nnz is None:
        if max_nnz_per_row <= 64:
            block_nnz_use = 64
        elif max_nnz_per_row <= 256:
            block_nnz_use = 128
        elif max_nnz_per_row <= 1024:
            block_nnz_use = 256
        elif max_nnz_per_row <= 4096:
            block_nnz_use = 512
        elif max_nnz_per_row <= 16384:
            block_nnz_use = 1024
        else:
            block_nnz_use = 2048
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be a positive integer")

    required_segments = max(
        (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
    )
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max(
                    (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
                )
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be a positive integer")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )

    return block_nnz_use, max_segments_use


def _triton_spsv_csr_cw_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    worker_count=None,
    matrix_stats=None,
    ready_in=None,
    row_counter_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return x
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    use_fp64_acc = data.dtype == torch.float64
    grid = (worker_count,)
    _spsv_csr_cw_kernel[grid](
        data,
        indices,
        indptr,
        b_vec,
        x,
        ready,
        row_counter,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_cw_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    worker_count=None,
    matrix_stats=None,
    data_ri_in=None,
    ready_in=None,
    row_counter_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return x

    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_cw_kernel_complex[grid](
        data_ri,
        indices,
        indptr,
        b_ri,
        x_ri,
        ready,
        row_counter,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_u_lo_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=True, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_lo_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=True, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_up_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=False, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_up_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=False, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_lo_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=True, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_lo_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=True, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_up_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=False, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_up_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=False, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_transpose_cw_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
    residual_in=None,
    indegree_in=None,
    row_counter_in=None,
    preprocessed=False,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    residual = residual_in if residual_in is not None else b_vec.clone()
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    residual.copy_(b_vec)
    row_counter.zero_()
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    if not preprocessed:
        _run_spsv_csc_preprocess(
            indices,
            indptr,
            indegree,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz_use=block_nnz_use,
            max_segments_use=max_segments_use,
        )
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_transpose_cw_kernel[grid](
        data,
        indices,
        indptr,
        indegree,
        residual,
        x,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_transpose_cw_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    conjugate=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
    data_ri_in=None,
    residual_in=None,
    indegree_in=None,
    row_counter_in=None,
    preprocessed=False,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    residual_work = residual_in if residual_in is not None else b_vec.contiguous().clone()
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    residual_work.copy_(b_vec.contiguous())
    row_counter.zero_()
    if not preprocessed:
        _run_spsv_csc_preprocess(
            indices,
            indptr,
            indegree,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz_use=block_nnz_use,
            max_segments_use=max_segments_use,
        )
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    residual_ri = torch.view_as_real(residual_work).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    if component_dtype == torch.float16:
        x_ri_work = torch.zeros((n_rows, 2), dtype=torch.float32, device=b_vec.device)
        x_ri = x_ri_work.reshape(-1).contiguous()
    else:
        x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_transpose_cw_kernel_complex[grid](
        data_ri,
        indices,
        indptr,
        indegree,
        residual_ri,
        x_ri,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        CONJ_TRANS=conjugate,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    if component_dtype == torch.float16:
        return torch.view_as_complex(x_ri_work.contiguous())
    return x


def _choose_transpose_family_launch_config(indptr, block_nnz=None, max_segments=None):
    if block_nnz is not None or max_segments is not None:
        return _auto_spsv_launch_config(indptr, block_nnz=block_nnz, max_segments=max_segments)

    if indptr.numel() <= 1:
        return 32, 1
    max_nnz_per_row = int((indptr[1:] - indptr[:-1]).max().item())
    for cand in (32, 64, 128, 256, 512, 1024):
        req = max((max_nnz_per_row + cand - 1) // cand, 1)
        if req <= 2048:
            return cand, req
    cand = 2048
    req = max((max_nnz_per_row + cand - 1) // cand, 1)
    return cand, req


def _run_spsv_csc_preprocess(
    indices,
    indptr,
    indegree,
    n_rows,
    *,
    lower,
    unit_diagonal,
    block_nnz_use,
    max_segments_use,
):
    indegree.zero_()
    if n_rows == 0:
        return indegree
    grid = (n_rows,)
    _spsv_csc_preprocess_kernel[grid](
        indices,
        indptr,
        indegree,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        UNIT_DIAG=unit_diagonal,
    )
    return indegree


def _prepare_spsv_coo_inputs(data, row, col, b, shape):
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, b)):
        raise ValueError("data, row, col, b must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if row.numel() != data.numel() or col.numel() != data.numel():
        raise ValueError("data, row, col must have the same length")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")

    if data.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise TypeError(
            "data dtype must be one of: float32, float64, complex64, complex128"
        )
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    input_index_dtype = (
        torch.int64
        if row.dtype == torch.int64 or col.dtype == torch.int64
        else torch.int32
    )
    row64 = row.to(torch.int64).contiguous()
    col64 = col.to(torch.int64).contiguous()
    if col64.numel() > 0 and int(col64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(col64.max().item())} exceeds Triton int32 kernel range"
        )
    if row64.numel() > 0:
        if bool(torch.any(row64 < 0).item()):
            raise IndexError("row indices must be non-negative")
        if bool(torch.any(col64 < 0).item()):
            raise IndexError("col indices must be non-negative")
        max_row = int(row64.max().item())
        max_col = int(col64.max().item())
        if max_row >= n_rows:
            raise IndexError(f"row indices out of range for n_rows={n_rows}")
        if max_col >= n_cols:
            raise IndexError(f"col indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        input_index_dtype,
        row64,
        col64,
        b.contiguous(),
        n_rows,
        n_cols,
    )


def _build_coo_row_ptr(row_sorted, n_rows):
    row_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=row_sorted.device)
    if row_sorted.numel() > 0:
        nnz_per_row = torch.bincount(row_sorted, minlength=n_rows)
        row_ptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return row_ptr


def _coo_order_for_spsv(data, row64, col64):
    if data.numel() == 0:
        return data, row64, col64
    key = row64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order], row64[order], col64[order]


def _coo2csr_for_spsv(data, row64, col64, n_rows, assume_ordered=False):
    nnz = data.numel()
    if nnz == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        indices = torch.empty(0, dtype=torch.int64, device=data.device)
        return data, indices, indptr

    if not assume_ordered:
        data, row64, col64 = _coo_order_for_spsv(data, row64, col64)

    indptr = _build_coo_row_ptr(row64, n_rows)
    indices = col64.to(torch.int64).contiguous()
    return data.contiguous(), indices, indptr


def _analyze_spsv_csr_descriptor(
    data,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
    format_name="csr",
    clear_cache=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    n_rows = int(shape[0])
    dummy_b = torch.empty(n_rows, dtype=data.dtype, device=data.device)
    (
        matrix_data,
        _dummy_b,
        _original_output_dtype,
        trans_mode,
        n_rows,
        n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        dummy_b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        storage_view=storage_view,
    )
    input_index_dtype = indices.dtype
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    compute_dtype = _spsv_effective_compute_dtype(
        matrix_data.dtype, trans_mode, compute_dtype=compute_dtype
    )
    layout = _build_spsv_workspace_layout(
        n_rows, solve_plan["solve_kind"], value_dtype=compute_dtype
    )
    if workspace is not None:
        _resolve_spsv_workspace(workspace, layout, matrix_data.device)
    return FlagSparseSpSVDescr(
        format=_normalize_spsv_format(format_name),
        canonical_format="csr",
        shape=(int(shape[0]), int(shape[1])),
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        fill_mode="lower" if lower else "upper",
        diag_type="unit" if unit_diagonal else "non_unit",
        matrix_type="triangular",
        index_base=0,
        transpose_mode=trans_mode,
        value_dtype=matrix_data.dtype,
        compute_dtype=compute_dtype,
        index_dtype=input_index_dtype,
        solve_kind=solve_plan["solve_kind"],
        route_name=str(solve_plan.get("route_name", solve_plan["solve_kind"])),
        storage_view=str(solve_plan.get("storage_view", "csr")),
        buffer_size=_workspace_size_bytes(layout),
        workspace_layout=layout,
        data=matrix_data,
        indices=indices.contiguous(),
        indptr=indptr.contiguous(),
        solve_plan=_clone_spsv_plan(solve_plan),
    )


def flagsparse_spsv_analysis_csr(
    data,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
    clear_cache=False,
):
    """Analyze a CSR SpSV problem and return a reusable Triton descriptor."""

    return _analyze_spsv_csr_descriptor(
        data,
        indices,
        indptr,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        handle=handle,
        workspace=workspace,
        storage_view=storage_view,
        format_name="csr",
        clear_cache=clear_cache,
    )


def flagsparse_spsv_analysis_coo(
    data,
    row,
    col,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
):
    """Analyze a COO SpSV problem by canonicalizing COO into CSR first."""

    dummy_b = torch.empty(int(shape[0]), dtype=data.dtype, device=data.device)
    data, _input_index_dtype, row64, col64, _b, n_rows, n_cols = _prepare_spsv_coo_inputs(
        data, row, col, dummy_b, shape
    )
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, row.dtype, "COO")
    else:
        _validate_spsv_trans_combo(data.dtype, row.dtype, "COO")
    data_csr, indices_csr, indptr_csr = _coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    return _analyze_spsv_csr_descriptor(
        data_csr,
        indices_csr,
        indptr_csr,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        handle=handle,
        workspace=workspace,
        storage_view=storage_view,
        format_name="coo",
        clear_cache=False,
    )


def _execute_spsv_csr_plan(
    data,
    b,
    solve_plan,
    trans_mode,
    n_rows,
    *,
    alpha=1,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    original_output_dtype=None,
    compute_dtype=None,
    handle=None,
    stream=None,
):
    solve_plan = _clone_spsv_plan(solve_plan)
    solve_kind = solve_plan["solve_kind"]
    kernel_data = solve_plan["kernel_data"]
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    default_block_nnz = solve_plan["default_block_nnz"]
    default_max_segments = solve_plan["default_max_segments"]
    cw_worker_count = solve_plan.get("cw_worker_count")
    nontrans_variant = solve_plan.get("nontrans_variant", "csr_n_lo_cw")
    lower_eff = solve_plan["lower_eff"]
    matrix_stats = solve_plan.get("matrix_stats", {})
    kernel_indices = kernel_indices32
    kernel_indptr = kernel_indptr64
    compute_dtype = _spsv_effective_compute_dtype(
        data.dtype, trans_mode, compute_dtype=compute_dtype
    )
    data_in = kernel_data
    b_in = b
    if compute_dtype != data.dtype:
        data_in = kernel_data.to(compute_dtype)
        b_in = b.to(compute_dtype)
    alpha_in = _coerce_spsv_alpha(alpha, compute_dtype, b.device)
    b_in = b_in * alpha_in
    solve_stream = _resolve_spsv_stream(handle, stream, b.device)

    if solve_kind == "transpose_cw":
        if block_nnz is None and max_segments is None:
            block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        else:
            block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
                kernel_indptr, block_nnz=block_nnz, max_segments=max_segments
            )
        vec_real = _triton_spsv_csr_transpose_cw_vector
        vec_complex = _triton_spsv_csr_transpose_cw_vector_complex
    elif solve_kind == "csr_cw":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        nontrans_real_wrappers = {
            "csr_u_lo_cw": _triton_spsv_csr_u_lo_cw_vector,
            "csr_n_lo_cw": _triton_spsv_csr_n_lo_cw_vector,
            "csr_u_up_cw": _triton_spsv_csr_u_up_cw_vector,
            "csr_n_up_cw": _triton_spsv_csr_n_up_cw_vector,
        }
        nontrans_complex_wrappers = {
            "csr_u_lo_cw": _triton_spsv_csr_u_lo_cw_vector_complex,
            "csr_n_lo_cw": _triton_spsv_csr_n_lo_cw_vector_complex,
            "csr_u_up_cw": _triton_spsv_csr_u_up_cw_vector_complex,
            "csr_n_up_cw": _triton_spsv_csr_n_up_cw_vector_complex,
        }
        vec_real = nontrans_real_wrappers[nontrans_variant]
        vec_complex = nontrans_complex_wrappers[nontrans_variant]
    else:
        raise RuntimeError(f"unexpected SpSV solve kind: {solve_kind}")
    diag_eps = _spsv_diag_eps_for_dtype(compute_dtype)

    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    worker_count_use = cw_worker_count
    matrix_stats_use = dict(matrix_stats)
    if solve_kind in ("csr_cw", "transpose_cw"):
        worker_count_use = _resolve_cw_worker_count(
            n_rows,
            matrix_stats_use,
            1,
            cached_worker_count=cw_worker_count,
        )
    complex_kernel_data_ri = None
    if torch.is_complex(data_in):
        if compute_dtype == solve_plan["kernel_data"].dtype:
            complex_kernel_data_ri = solve_plan.get("kernel_data_ri")
        if complex_kernel_data_ri is None:
            complex_kernel_data_ri = _complex_interleaved_view(data_in)
    workspace_buffers = _resolve_spsv_workspace(
        workspace,
        _build_spsv_workspace_layout(n_rows, solve_kind, value_dtype=compute_dtype),
        b.device,
    )
    ready_buf = workspace_buffers.get("ready")
    residual_buf = workspace_buffers.get("residual")
    indegree_buf = workspace_buffers.get("indegree")
    row_counter_buf = workspace_buffers.get("row_counter")
    transpose_preprocessed = False
    if solve_kind == "transpose_cw":
        transpose_sig = _transpose_cw_preprocess_signature(
            solve_plan,
            n_rows,
            unit_diagonal,
            block_nnz_use,
            max_segments_use,
        )
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            transpose_preprocessed = (
                workspace.prepared_solve_kind == "transpose_cw"
                and workspace.prepared_signature == transpose_sig
            )
        if not transpose_preprocessed:
            _run_spsv_csc_preprocess(
                kernel_indices,
                kernel_indptr,
                indegree_buf,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
            )
            transpose_preprocessed = True
            if isinstance(workspace, FlagSparseSpSVWorkspace):
                workspace.prepared_solve_kind = "transpose_cw"
                workspace.prepared_signature = transpose_sig
    stream_ctx = (
        torch.cuda.stream(solve_stream)
        if solve_stream is not None
        else nullcontext()
    )
    with stream_ctx:
        if torch.is_complex(data_in):
            if solve_kind == "transpose_cw":
                x = vec_complex(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                conjugate=(trans_mode == "C"),
                block_nnz=block_nnz,
                max_segments=max_segments,
                diag_eps=diag_eps,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                data_ri_in=complex_kernel_data_ri,
                residual_in=residual_buf,
                indegree_in=indegree_buf,
                row_counter_in=row_counter_buf,
                preprocessed=transpose_preprocessed,
                )
            else:
                x = vec_complex(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                diag_eps=diag_eps,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                data_ri_in=complex_kernel_data_ri,
                ready_in=ready_buf,
                row_counter_in=row_counter_buf,
                )
        else:
            if solve_kind == "transpose_cw":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                block_nnz=block_nnz,
                max_segments=max_segments,
                diag_eps=diag_eps,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                residual_in=residual_buf,
                indegree_in=indegree_buf,
                row_counter_in=row_counter_buf,
                preprocessed=transpose_preprocessed,
                )
            else:
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                diag_eps=diag_eps,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                ready_in=ready_buf,
                row_counter_in=row_counter_buf,
                )
    target_dtype = original_output_dtype if original_output_dtype is not None else data.dtype
    if x.dtype != target_dtype:
        x = x.to(target_dtype)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out

    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsv_solve_csr(
    descr,
    b,
    *,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    handle=None,
    stream=None,
):
    """Solve a previously analyzed CSR SpSV problem."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if descr.canonical_format != "csr":
        raise ValueError("descr must reference a CSR-canonicalized SpSV analysis")
    if not torch.is_tensor(b):
        raise TypeError("b must be a torch.Tensor")
    if not b.is_cuda:
        raise ValueError("b must be a CUDA tensor")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")
    if int(b.numel()) != int(descr.shape[0]):
        raise ValueError(f"b length must equal n_rows={descr.shape[0]}")
    if b.dtype != descr.value_dtype:
        raise TypeError("b dtype must match the analyzed matrix dtype")
    return _execute_spsv_csr_plan(
        descr.data,
        b.contiguous(),
        descr.solve_plan,
        descr.transpose_mode,
        int(descr.shape[0]),
        alpha=alpha,
        unit_diagonal=descr.unit_diagonal,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        original_output_dtype=descr.value_dtype,
        compute_dtype=compute_dtype if compute_dtype is not None else descr.compute_dtype,
        handle=handle,
        stream=stream,
    )


def flagsparse_spsv_solve_coo(
    descr,
    b,
    *,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    handle=None,
    stream=None,
):
    """Solve a previously analyzed COO SpSV problem via its CSR canonical form."""

    return flagsparse_spsv_solve_csr(
        descr,
        b,
        alpha=alpha,
        compute_dtype=compute_dtype,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        handle=handle,
        stream=stream,
    )


def _materialize_spsv_workspace_state(descr, workspace=None):
    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    buffers = _resolve_spsv_workspace(
        workspace, descr.workspace_layout, descr.data.device
    )
    solve_kind = descr.solve_kind
    if solve_kind == "csr_cw":
        ready = buffers.get("ready")
        row_counter = buffers.get("row_counter")
        if ready is not None:
            ready.zero_()
        if row_counter is not None:
            row_counter.zero_()
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = ""
            workspace.prepared_signature = None
    elif solve_kind == "transpose_cw":
        residual = buffers.get("residual")
        indegree = buffers.get("indegree")
        row_counter = buffers.get("row_counter")
        block_nnz_use = int(descr.solve_plan["default_block_nnz"])
        max_segments_use = int(descr.solve_plan["default_max_segments"])
        preprocess_sig = _transpose_cw_preprocess_signature(
            descr.solve_plan,
            int(descr.shape[0]),
            bool(descr.unit_diagonal),
            block_nnz_use,
            max_segments_use,
        )
        if residual is not None:
            residual.zero_()
        if indegree is not None:
            _run_spsv_csc_preprocess(
                descr.solve_plan["kernel_indices32"],
                descr.solve_plan["kernel_indptr64"],
                indegree,
                int(descr.shape[0]),
                lower=bool(descr.solve_plan["lower_eff"]),
                unit_diagonal=bool(descr.unit_diagonal),
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
            )
        if row_counter is not None:
            row_counter.zero_()
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = "transpose_cw"
            workspace.prepared_signature = preprocess_sig
    else:
        raise RuntimeError(f"unexpected SpSV solve kind: {solve_kind}")
    if workspace is None:
        return FlagSparseSpSVWorkspace(
            buffer_size=int(descr.buffer_size),
            layout=tuple(descr.workspace_layout),
            device=descr.data.device,
            buffers=buffers,
            prepared_solve_kind="transpose_cw" if solve_kind == "transpose_cw" else "",
            prepared_signature=preprocess_sig if solve_kind == "transpose_cw" else None,
        )
    return workspace


def flagsparse_spsv_preprocess_csr(descr, *, workspace=None):
    """Materialize caller-managed workspace for a CSR SpSV descriptor."""

    return _materialize_spsv_workspace_state(descr, workspace=workspace)


def flagsparse_spsv_preprocess_coo(descr, *, workspace=None):
    """Materialize caller-managed workspace for a COO SpSV descriptor."""

    return _materialize_spsv_workspace_state(descr, workspace=workspace)


def flagsparse_spsv_buffer_size_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    storage_view="csr_as_csc",
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    return flagsparse_spsv_buffer_size(
        matA.shape,
        matA.values.dtype,
        format=matA.format,
        transpose=opA,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        alpha=alpha,
        handle=handle,
        vecX=vecX,
        vecY=vecY,
        storage_view=storage_view,
    )


def flagsparse_spsv_analysis_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    workspace=None,
    storage_view="csr_as_csc",
    clear_cache=False,
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    if matA.format == "csr":
        return flagsparse_spsv_analysis_csr(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            solve_kind=solve_kind,
            compute_dtype=compute_dtype,
            handle=handle,
            workspace=workspace,
            storage_view=storage_view,
            clear_cache=clear_cache,
        )
    if matA.format == "coo":
        return flagsparse_spsv_analysis_coo(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            solve_kind=solve_kind,
            compute_dtype=compute_dtype,
            handle=handle,
            workspace=workspace,
            storage_view=storage_view,
        )
    raise ValueError("matA.format must be 'csr' or 'coo'")


def flagsparse_spsv_solve_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    descr=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    workspace=None,
    stream=None,
    storage_view="csr_as_csc",
    block_nnz=None,
    max_segments=None,
    return_time=False,
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    out_tensor = None if vecY is None else vecY.values
    if matA.format == "csr":
        return flagsparse_spsv_csr(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            vecX.values,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out_tensor,
            return_time=return_time,
            descr=descr,
            workspace=workspace,
            solve_kind=solve_kind,
            handle=handle,
            stream=stream,
            storage_view=storage_view,
        )
    if matA.format == "coo":
        return flagsparse_spsv_coo(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            vecX.values,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out_tensor,
            return_time=return_time,
            descr=descr,
            workspace=workspace,
            solve_kind=solve_kind,
            handle=handle,
            stream=stream,
            storage_view=storage_view,
        )
    raise ValueError("matA.format must be 'csr' or 'coo'")


def flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    descr=None,
    workspace=None,
    solve_kind=None,
    handle=None,
    stream=None,
    storage_view="csr_as_csc",
):
    """Sparse triangular solve using Triton CSR CW kernels.

    Current support matrix:
    - NON_TRANS: float32/float64/complex64/complex128 with int32/int64 indices
    - TRANS/CONJ: float32/float64/complex64/complex128 with int32/int64 indices
    """
    if descr is not None:
        if not isinstance(descr, FlagSparseSpSVDescr):
            raise TypeError("descr must be a FlagSparseSpSVDescr or None")
        return flagsparse_spsv_solve_csr(
            descr,
            b,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            workspace=workspace,
            handle=handle,
            stream=stream,
        )
    (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        storage_view=storage_view,
    )
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    return _execute_spsv_csr_plan(
        data,
        b,
        solve_plan,
        trans_mode,
        n_rows,
        alpha=alpha,
        unit_diagonal=unit_diagonal,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        original_output_dtype=original_output_dtype,
        compute_dtype=compute_dtype,
        handle=handle,
        stream=stream,
    )


def _analyze_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
    )
    if return_time:
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0


def flagsparse_spsv_coo(
    data,
    row,
    col,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    descr=None,
    workspace=None,
    solve_kind=None,
    handle=None,
    stream=None,
    storage_view="csr_as_csc",
):
    """COO SpSV by canonicalizing COO into CSR, then reusing CSR SpSV."""
    if descr is not None:
        if not isinstance(descr, FlagSparseSpSVDescr):
            raise TypeError("descr must be a FlagSparseSpSVDescr or None")
        return flagsparse_spsv_solve_coo(
            descr,
            b,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            workspace=workspace,
            handle=handle,
            stream=stream,
        )
    data, input_index_dtype, row64, col64, b, n_rows, n_cols = _prepare_spsv_coo_inputs(
        data, row, col, b, shape
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")

    trans_mode = _normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "COO")
    else:
        _validate_spsv_trans_combo(data.dtype, input_index_dtype, "COO")
    data_csr, indices_csr, indptr_csr = _coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    return flagsparse_spsv_csr(
        data_csr,
        indices_csr,
        indptr_csr,
        b,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        alpha=alpha,
        compute_dtype=compute_dtype,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        solve_kind=solve_kind,
        handle=handle,
        stream=stream,
        storage_view=storage_view,
    )
