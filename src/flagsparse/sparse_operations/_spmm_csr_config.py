"""Pure Python contracts for the native CSR SpMM extensions (no GPU imports)."""

from dataclasses import asdict, dataclass

NEW_ALGORITHMS = (
    "csr_row_tile", "csr_row_kparallel", "csr_split_nnz_reduce",
    "csr_adaptive_tile_split", "csr_row_panel",
)
BACKENDS = ("cuda", "rocm", "metax", "mthreads", "ascend", "xpu", "gcu", "mlu")
DTYPES = ("float16", "float32", "float64", "complex64", "complex128")
IMPLEMENTATION_VERSION = 2
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
    fp32_atomic: object = None
    fp64_atomic: object = None


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
    if algorithm == "csr_row_panel":
        panel_n = 16 if dtype == "complex128" else 32
        cfg.update(tile_rows=4 if dtype == "complex128" else 8,
                   tile_n=min(panel_n, 1 << (max(1, n) - 1).bit_length()),
                   num_warps=2, panel_accumulators=2)
    source = "conservative"
    rejections = []
    if cfg["num_warps"] not in caps.legal_num_warps:
        rejections.append(f"default num_warps={cfg['num_warps']} is unsupported by target")
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
                if set(profile) - set(cfg):
                    raise ValueError(f"profile has fields not used by {algorithm}: {sorted(set(profile) - set(cfg))}")
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
    fields = CONFIG_FIELDS | ({"panel_accumulators"} if "panel_accumulators" in cfg else set())
    if set(cfg) != fields:
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
    if cfg.get("panel_accumulators", 2) not in (1, 2, 4):
        raise ValueError("panel_accumulators must be 1, 2 or 4")
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


# COO shares backend capability contracts; kernels remain in the COO modules.
COO_ALGORITHMS = ("coo_segmented_atomic", "coo_rowrun_subgroup",
                  "coo_segmented_panel_atomic", "coo_rowrun_panel")
COO_ARCH_PROFILES = {}
COO_BACKEND_PROFILES = {name: {} for name in BACKENDS}


def resolve_coo_config(alg, dtype, caps, config=None, n=1):
    if alg not in COO_ALGORITHMS:
        raise ValueError(f"unknown COO extension {alg!r}")
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if caps.backend not in BACKENDS or not caps.legal_num_warps or not caps.subgroup_width or caps.int64 is not True:
        raise NotImplementedError(f"COO launch/index capabilities unknown: {caps}")
    atomic = "atomic" in alg
    overrides = dict(config or {})
    cfg = dict(num_warps=4 if atomic else 2, num_stages=1)
    if atomic:
        cfg.update(block_nnz=128 if alg == "coo_segmented_atomic" else 32,
                   local_reduce="segment")
        if alg == "coo_segmented_panel_atomic":
            cfg["block_n"] = min(16, (1 << (max(1, n) - 1).bit_length()))
    elif alg == "coo_rowrun_subgroup":
        cfg.update(lanes_per_row=8, rows_per_program=1)
    else:
        cfg.update(tile_rows=4, tile_n=min(16, (1 << (max(1, n) - 1).bit_length())))
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"unsupported {alg} configuration fields: {sorted(unknown)}")
    if cfg["num_warps"] not in caps.legal_num_warps:
        cfg["num_warps"] = max((w for w in caps.legal_num_warps if w <= cfg["num_warps"]),
                               default=min(caps.legal_num_warps))
    profile = COO_BACKEND_PROFILES.get(caps.backend, {}).get(alg, {})
    arch_profile = COO_ARCH_PROFILES.get((caps.backend, caps.arch), {}).get(alg, {})
    rejected = []
    accepted_keys = set()
    for source, candidate in (("backend", profile), ("architecture", arch_profile)):
        for key, value in candidate.items():
            valid = key in cfg and (
                value in ("segment", "none") if key == "local_reduce" else
                type(value) is int and value > 0 and not value & (value - 1))
            valid = valid and (key != "num_warps" or value in caps.legal_num_warps)
            valid = valid and (key != "num_stages" or value == 1)
            valid = valid and (key != "lanes_per_row" or value <= caps.subgroup_width)
            if valid:
                cfg[key] = value
                accepted_keys.add(key)
            else:
                rejected.append(f"{source}: illegal {key}={value!r}")
    cfg.update(overrides)
    for key, value in cfg.items():
        if key == "local_reduce":
            if value not in ("segment", "none"):
                raise ValueError("local_reduce must be segment or none")
        elif type(value) is not int or value <= 0 or value & (value - 1):
            raise ValueError(f"{key} must be a positive power of two")
    if (cfg["num_warps"] not in caps.legal_num_warps or cfg["num_stages"] != 1
            or cfg["num_warps"] * caps.subgroup_width > caps.max_threads_per_block):
        raise ValueError("illegal COO launch configuration")
    if alg == "coo_rowrun_subgroup" and "rows_per_program" not in overrides and "rows_per_program" not in accepted_keys:
        cfg["rows_per_program"] = cfg["num_warps"] * caps.subgroup_width // cfg["lanes_per_row"]
    if cfg.get("lanes_per_row", 1) > caps.subgroup_width:
        raise ValueError("lanes_per_row exceeds backend subgroup width")
    if dtype in ("float64", "complex128") and caps.fp64 is not True:
        raise NotImplementedError("COO FP64 capability unavailable or unknown")
    if atomic:
        feature = "fp64_atomic" if dtype in ("float64", "complex128") else "fp32_atomic"
        supported = getattr(caps, feature)
        if supported is not True:
            raise NotImplementedError(f"{feature} unavailable or unknown on {caps.backend}")
        if cfg["local_reduce"] == "segment" and caps.scan is not True:
            raise NotImplementedError("COO segmented scan capability unknown")
    elif caps.stable_sort is not True or caps.reduction is not True:
        raise NotImplementedError("COO stable sort/reduction capability unknown")
    return cfg, dict(backend=caps.backend, target=caps.target, arch=caps.arch,
                     config_source="explicit" if overrides else ("architecture" if arch_profile else "backend" if profile else "conservative"),
                     validation="unverified", config=cfg, rejected_profiles=rejected, implementation_version=1,
                     timing_contract_version=2)


CSC_ALGORITHMS = {
    "csc_col_subgroup": ("trans", "conj"),
    "csc_col_tile_atomic": ("non",),
    "csc_col_panel": ("trans", "conj"),
    "csc_col_tile_panel_atomic": ("non",),
}
CSC_ARCH_PROFILES = {}
CSC_BACKEND_PROFILES = {}


def resolve_csc_config(algorithm, dtype, n, caps, overrides=None, *, op="non"):
    """Resolve native CSC launch policy without importing an accelerator runtime."""
    if algorithm not in CSC_ALGORITHMS:
        raise ValueError(f"unknown CSC algorithm {algorithm!r}")
    if op not in CSC_ALGORITHMS[algorithm]:
        raise ValueError(f"{algorithm} does not support op={op}")
    if dtype not in DTYPES:
        raise TypeError(f"unsupported CSC dtype {dtype}")
    if caps.backend not in BACKENDS or not caps.legal_num_warps or not caps.subgroup_width:
        raise NotImplementedError("CSC launch capabilities unavailable or unknown")
    required = ["int64"]
    atomic = "atomic" in algorithm
    if dtype in ("float64", "complex128"):
        required.append("fp64")
    if atomic:
        required.append("fp64_atomic" if dtype in ("float64", "complex128") else "fp32_atomic")
    else:
        required.append("reduction")
    for key in required:
        if getattr(caps, key) is not True:
            raise NotImplementedError(f"CSC {key} capability unavailable for {caps.backend}/{caps.arch}")
    warps = 4 if atomic else 2
    if warps not in caps.legal_num_warps:
        warps = max((w for w in caps.legal_num_warps if w <= warps), default=min(caps.legal_num_warps))
    panel = "panel" in algorithm
    bn = 8 if dtype == "complex128" and not atomic else 16
    cfg = dict(num_warps=warps, num_stages=1)
    if algorithm == "csc_col_subgroup":
        cfg.update(lanes_per_column=8, columns_per_program=warps * caps.subgroup_width // 8)
    elif algorithm == "csc_col_tile_atomic":
        cfg.update(columns_per_program=4, block_nnz=32)
    elif algorithm == "csc_col_panel":
        cfg.update(columns_per_program=4, block_n=min(bn, 1 << (max(1, n)-1).bit_length()), panel_accumulators=2)
    else:
        cfg.update(columns_per_program=2, block_nnz=16, block_n=min(bn, 1 << (max(1, n)-1).bit_length()))
    allowed = set(cfg)
    def validate(candidate):
        if set(candidate) != allowed:
            raise ValueError(f"unsupported CSC config fields: {sorted(set(candidate)-allowed)}")
        for key, value in candidate.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            if key != "num_stages" and value & (value-1):
                raise ValueError(f"{key} must be a power of two")
        if candidate["num_warps"] not in caps.legal_num_warps:
            raise ValueError("num_warps is not supported by this target")
        if candidate["num_warps"] * caps.subgroup_width > caps.max_threads_per_block:
            raise ValueError("launch exceeds max_threads_per_block")
        if candidate.get("lanes_per_column", 1) > caps.subgroup_width:
            raise ValueError("lanes_per_column exceeds subgroup width")
        if candidate.get("panel_accumulators", 1) not in (1, 2):
            raise ValueError("panel_accumulators must be 1 or 2")
    source, rejected = "conservative", []
    for key, table in ((caps.backend, CSC_BACKEND_PROFILES), ((caps.backend, algorithm), CSC_BACKEND_PROFILES),
                       ((caps.backend, caps.arch, algorithm), CSC_ARCH_PROFILES)):
        if key in table:
            candidate = dict(cfg, **table[key])
            try:
                validate(candidate)
            except ValueError as exc:
                rejected.append(str(exc))
            else:
                cfg, source = candidate, str(key)
    explicit = dict(overrides or {})
    for key, value in explicit.items():
        if key not in allowed or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid CSC configuration {key}={value!r}")
    if algorithm == "csc_col_subgroup" and "columns_per_program" not in explicit:
        cfg["columns_per_program"] = explicit.get("num_warps", cfg["num_warps"]) * caps.subgroup_width // explicit.get("lanes_per_column", cfg["lanes_per_column"])
    cfg.update(explicit)
    validate(cfg)
    return cfg, dict(config_source="explicit" if explicit else source, config_rejections=rejected,
                     backend=caps.backend, target=caps.target, arch=caps.arch,
                     subgroup_width=caps.subgroup_width, validation="unverified")
