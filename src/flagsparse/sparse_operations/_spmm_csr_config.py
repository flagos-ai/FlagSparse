"""Pure Python contracts for the native CSR SpMM extensions (no GPU imports)."""

from dataclasses import asdict, dataclass

NEW_ALGORITHMS = (
    "csr_row_tile", "csr_row_kparallel", "csr_split_nnz_reduce",
    "csr_adaptive_tile_split",
)
BACKENDS = ("cuda", "rocm", "metax", "mthreads", "ascend", "xpu", "gcu", "mlu")
DTYPES = ("float32", "float64", "complex64", "complex128")
IMPLEMENTATION_VERSION = 1
CONFIG_FIELDS = frozenset((
    "tile_rows", "tile_k", "tile_n", "block_k", "block_n", "panels",
    "segment_nnz", "reduce_block_size", "num_warps", "num_stages",
    "workspace_bytes", "short_row_threshold", "split_row_threshold",
))
# Entries require recorded device evidence; unmeasured defaults are not tuning results.
ARCH_PROFILES = {}
BACKEND_PROFILES = {name: {} for name in BACKENDS}


@dataclass(frozen=True)
class BackendCaps:
    backend: str
    target: str = "unknown"
    arch: str = "unknown"
    subgroup_width: int = 0
    max_threads_per_block: int = 0
    legal_num_warps: tuple = ()
    fp64: object = None
    int64: object = None
    reduction: object = None
    stable_sort: object = None
    scan: object = None


def algorithm_spec(name):
    if name not in NEW_ALGORITHMS:
        raise ValueError(f"unknown new CSR SpMM algorithm {name!r}")
    return dict(name=name, ops=("non", "trans", "conj"), value_dtypes=DTYPES,
                layouts=("row", "col", "strided"), backends=BACKENDS,
                index_dtypes=("int32", "int64"), execution_column_dtype="int32",
                compute_dtype="native_component", implementation_version=IMPLEMENTATION_VERSION,
                transpose_strategy="per_run_csr_rebuild", validation="unverified")


def capability_reason(caps, dtype, *, op="non", algorithm="csr_row_tile"):
    if caps.backend not in BACKENDS:
        return f"unknown backend {caps.backend}"
    if not caps.subgroup_width or not caps.max_threads_per_block or not caps.legal_num_warps:
        return f"launch capabilities unknown for {caps.backend}/{caps.arch}"
    required = ["int64", "reduction"]
    if dtype in ("float64", "complex128"):
        required.append("fp64")
    if op != "non" or algorithm in ("csr_split_nnz_reduce", "csr_adaptive_tile_split"):
        required.extend(("stable_sort", "scan"))
    for key in required:
        if getattr(caps, key) is not True:
            return f"{key} capability unavailable or unknown for {caps.backend}/{caps.arch}"
    return None


def resolve_config(algorithm, dtype, n, layout, caps, overrides=None, *, op="non"):
    algorithm_spec(algorithm)
    if dtype not in DTYPES:
        raise TypeError(f"{algorithm} does not support {dtype}")
    reason = capability_reason(caps, dtype, op=op, algorithm=algorithm)
    if reason:
        raise NotImplementedError(reason)
    cfg = dict(tile_rows=4, tile_k=8, tile_n=8 if dtype == "complex128" else 16,
               block_k=32, block_n=16 if dtype == "complex128" else 32,
               panels=1, segment_nnz=1024, reduce_block_size=256,
               num_warps=4, num_stages=1, workspace_bytes=256 * 1024 * 1024,
               short_row_threshold=32 if layout == "row" and n <= 32 else 16,
               split_row_threshold=4096 if n >= 128 else 2048)
    source = "conservative"
    rejections = []
    if cfg["num_warps"] not in caps.legal_num_warps:
        rejections.append("default num_warps=4 is unsupported by target")
        cfg["num_warps"] = min(caps.legal_num_warps)
    for key, profiles in (
        (caps.backend, BACKEND_PROFILES),
        ((caps.backend, algorithm), BACKEND_PROFILES),
        ((caps.backend, caps.arch), ARCH_PROFILES),
        ((caps.backend, caps.arch, algorithm), ARCH_PROFILES),
    ):
        profile = profiles.get(key, {})
        if profile:
            candidate = {**cfg, **profile}
            try:
                validate_config(candidate, caps)
            except ValueError as exc:
                rejections.append(str(exc))
            else:
                cfg, source = candidate, str(key)
    supplied = dict(overrides or {})
    unknown = set(supplied) - set(cfg)
    if unknown:
        raise ValueError(f"unknown CSR SpMM configuration keys: {sorted(unknown)}")
    cfg.update(supplied)
    if "segment_nnz" in supplied and "split_row_threshold" not in supplied:
        cfg["split_row_threshold"] = cfg["segment_nnz"] * (4 if n >= 128 else 2)
    validate_config(cfg, caps)
    return cfg, dict(config_source="explicit" if supplied else source,
                     config_rejections=rejections, backend_caps=asdict(caps))


def validate_config(cfg, caps):
    if set(cfg) != CONFIG_FIELDS:
        raise ValueError(f"invalid configuration fields: {sorted(set(cfg) ^ CONFIG_FIELDS)}")
    for key, value in cfg.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**63:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("tile_rows", "tile_k", "tile_n", "block_k", "block_n", "panels", "reduce_block_size"):
        if cfg[key] & (cfg[key] - 1):
            raise ValueError(f"{key} must be a power of two")
    if not 2 <= cfg["reduce_block_size"] <= 256:
        raise ValueError("reduce_block_size must be between 2 and 256")
    if cfg["num_warps"] not in caps.legal_num_warps:
        raise ValueError("num_warps is unsupported by target")
    if cfg["num_warps"] * caps.subgroup_width > caps.max_threads_per_block:
        raise ValueError("launch exceeds target thread limit")
    if cfg["num_stages"] != 1:
        raise ValueError("initial CSR SpMM profiles support num_stages=1 only")
    if cfg["short_row_threshold"] >= cfg["split_row_threshold"]:
        raise ValueError("short_row_threshold must be less than split_row_threshold")
    if cfg["workspace_bytes"] < 48:
        raise ValueError("workspace_bytes must hold at least three complex128 elements")


def workspace_geometry(n, element_bytes, cfg):
    """Bound numeric partial/reduction buffers, including an allocation transition."""
    if n <= 0 or element_bytes <= 0:
        raise ValueError("workspace geometry requires positive N and element size")
    wave = min(n, cfg["block_n"], cfg["workspace_bytes"] // (3 * element_bytes))
    if wave < 1:
        raise ValueError("workspace cannot hold one output column and reduction buffers")
    capacity = cfg["workspace_bytes"] // (3 * wave * element_bytes)
    return wave, capacity
