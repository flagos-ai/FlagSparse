"""CPU-only CSR SpMV algorithm contracts and backend launch profiles."""

from copy import deepcopy
from dataclasses import dataclass

IMPLEMENTATION_VERSION = 9
NEW_ALGORITHMS = ("row_tile", "row_vector", "row_split_reduce", "row_adaptive_split", "row_subgroup_stream", "csr_nnz_tile_reduce", "csr_subgroup_pair_mixed", "csr_rowgroup_stream_reduce")
TARGETED_ALGORITHMS = ("csr_row_refill_reduce", "csr_transpose_sharded_atomic")
NEW_ALGORITHMS += TARGETED_ALGORITHMS
SHORT_ROW_ALGORITHMS = ("csr_row_scalar_strip", "csr_row_static_tile", "csr_row_scalar_bound", "csr_row_scalar_tail_reduce")
LONGROW_ALGORITHMS = ("csr_longrow_queue_atomic", "csr_longrow_grid_atomic")
ROWQUEUE_ALGORITHMS = ("csr_short_rowqueue_vector", "csr_short_rowqueue_strided_vector")
COOPERATIVE_ALGORITHMS = ("csr_longrow_cooperative_reduce",)
THREE_TIER_ALGORITHMS = ("csr_three_tier_queue_reduce",)
NEW_ALGORITHMS += SHORT_ROW_ALGORITHMS + LONGROW_ALGORITHMS + ROWQUEUE_ALGORITHMS + COOPERATIVE_ALGORITHMS + ("csr_nnz_local_reduce_atomic",)
NON_ONLY_ALGORITHMS = SHORT_ROW_ALGORITHMS + LONGROW_ALGORITHMS + ROWQUEUE_ALGORITHMS + COOPERATIVE_ALGORITHMS
NEW_ALGORITHMS += THREE_TIER_ALGORITHMS
NON_ONLY_ALGORITHMS += THREE_TIER_ALGORITHMS
LEGACY_ALGORITHMS = ("legacy_rowpar", "legacy_segbin", "legacy_bucket_vector")
ALGORITHMS = LEGACY_ALGORITHMS + NEW_ALGORITHMS
BACKENDS = ("cuda", "rocm", "metax", "mthreads", "ascend")
VALUE_DTYPES = ("float16", "bfloat16", "float32", "float64", "complex64", "complex128")


class BackendCapabilityUnavailable(NotImplementedError):
    """Known missing/unknown device capability, distinct from an implementation fault."""


@dataclass(frozen=True)
class BackendCaps:
    backend: str
    arch: str = "unknown"
    target: str = "unknown"
    subgroup_width: int = 0
    max_threads_per_block: int = 0
    fp64: bool = False
    int64: bool = False
    reduction: bool = False
    scan: bool = False
    fp32_atomic: object = None
    fp64_atomic: object = None
    int32_atomic: object = None


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def set_known_backends(backends):
    """Synchronize legacy-route backend names with the shared runtime registry."""
    global BACKENDS
    normalized = tuple(dict.fromkeys(str(name) for name in backends if name))
    if not normalized:
        raise ValueError("CSR SpMV backend registry must not be empty")
    BACKENDS = normalized


def normalize_alg(alg):
    name = "auto" if alg is None else str(alg).strip().lower()
    if name == "base":
        name = "auto"
    if name not in ("auto",) + ALGORITHMS:
        raise ValueError(
            f"unknown CSR SpMV algorithm {name!r}; expected auto or {ALGORITHMS}"
        )
    return name


def compute_dtype(alg, dtype):
    name = dtype_name(dtype)
    if name in ("float16", "bfloat16"):
        return "float32"
    if name == "float32" and alg in NEW_ALGORITHMS + ("legacy_rowpar",):
        return "float64"
    return name


def algorithm_spec(alg):
    name = normalize_alg(alg)
    if name == "auto":
        raise ValueError("auto is a selection policy, not a concrete algorithm")
    new = name in NEW_ALGORITHMS
    return {
        "name": name,
        "implementation_version": 2 if name == "csr_nnz_local_reduce_atomic" else 1 if name in NON_ONLY_ALGORITHMS else IMPLEMENTATION_VERSION,
        "ops": ("non",) if name in NON_ONLY_ALGORITHMS else ("trans", "conj") if name == "csr_transpose_sharded_atomic" else ("non", "trans", "conj"),
        "value_dtypes": ("float16", "bfloat16") if name in ("csr_nnz_tile_reduce", "csr_subgroup_pair_mixed") else VALUE_DTYPES,
        "index_dtypes": (
            ("int32",) if name == "legacy_bucket_vector" else ("int32", "int64")
        ),
        "indptr_dtypes": ("int32", "int64"),
        "backends": ("cuda", "rocm") if new else BACKENDS,
        "runtime_process": name not in SHORT_ROW_ALGORITHMS + ("csr_transpose_sharded_atomic",),
        "runtime_process_ops": ("non",) if name in LONGROW_ALGORITHMS + ROWQUEUE_ALGORITHMS + COOPERATIVE_ALGORITHMS + THREE_TIER_ALGORITHMS else () if name in SHORT_ROW_ALGORITHMS + ("csr_transpose_sharded_atomic",) else (
            ("non", "trans", "conj")
            if name
            in ("row_split_reduce", "row_adaptive_split", "legacy_bucket_vector", "csr_nnz_tile_reduce")
            else ("trans", "conj")
        ),
        "transpose_strategy": "unsupported" if name in NON_ONLY_ALGORITHMS else "native_csr_sharded_scatter" if name == "csr_transpose_sharded_atomic" else "per_run_csr_rebuild",
        "cross_call_plan_cache": False,
        "capability_requirements": ("int64", "reduction", "scan", "int32_atomic") if name in ROWQUEUE_ALGORITHMS + COOPERATIVE_ALGORITHMS + THREE_TIER_ALGORITHMS else ("int64", "reduction", "scan", "int32_atomic", "atomic_by_compute_dtype") if name in LONGROW_ALGORITHMS else ("int64", "reduction", "scan", "atomic_by_compute_dtype") if name == "csr_nnz_local_reduce_atomic" else ("int64", "reduction", "atomic_by_compute_dtype") if name == "csr_transpose_sharded_atomic" else ("int64", "reduction", "scan") if name in ("csr_rowgroup_stream_reduce", "csr_row_refill_reduce") else ("int64", "reduction") if new else (),
        "compute_dtype": "input_dependent",
        "compute_dtype_by_input": {dt: compute_dtype(name, dt) for dt in VALUE_DTYPES},
        "half_output_dtypes": {
            dt: (dt, "float32")
            for dt in ("float16", "bfloat16")
        },
        "config_candidates": ({"rows_per_program": (64, 128), "short_row_threshold": (16, 32, 64),
                               "long_row_threshold": (128, 256, 512, 1024), "cooperative_parts": (1, 2, 4, 8)}
                              if name in THREE_TIER_ALGORITHMS else {"long_row_threshold": (256, 512, 1024, 2048, 4096),
                               "cooperative_parts": (1, 2, 4, 8, 16), "grid_blocks": (64, 128, 256),
                               "block_nnz": (64, 128, 256)} if name in COOPERATIVE_ALGORITHMS else
                              {"rows_per_program": (32, 64, 128), "short_row_threshold": (0, 8, 16, 32, 64, 128, 256),
                               "block_nnz": (64, 128, 256), "grid_blocks": (128, 256, 512, 1024)}
                              if name == "csr_short_rowqueue_strided_vector" else
                              {"rows_per_program": (32, 64, 128), "short_row_threshold": (0, 8, 16, 32, 64, 128, 256),
                               "block_nnz": (64, 128, 256)} if name in ROWQUEUE_ALGORITHMS else
                              {"block_nnz": (128, 256, 512, 1024), "owner_search": ("bounded", "full"),
                               "single_row_fastpath": (True, False), "atomic_mode": ("boundary", "all")}
                              if name == "csr_nnz_local_reduce_atomic" else {"rows_per_program": (32, 64, 128), "scalar_steps": (0, 4, 8, 16, 32),
                               "block_nnz": (8, 16, 32, 64)} if name == "csr_row_scalar_tail_reduce" else
                              {"rows_per_program": (32, 64, 128)} if name in ("csr_row_scalar_strip", "csr_row_scalar_bound") else
                              {"rows_per_program": (4, 8, 16), "block_nnz": (8, 16, 32, 64)} if name == "csr_row_static_tile" else
                              {"rows_per_program": (8, 16, 32), "block_nnz": (64, 128, 256)}
                              if name == "csr_rowgroup_stream_reduce" else {"prefetch_chunks": (1, 2), "lanes_per_row": (2, 4, 8, 16, 32, 64), "row_tiles_per_program": (1, 2, 4, 8)}
                              if name == "csr_subgroup_pair_mixed" else {"lanes_per_row": (2, 4, 8, 16, 32, 64),
                               "row_tiles_per_program": (1, 2, 4, 8)}
                              if name == "row_subgroup_stream" else {"plan_mode": ("compact", "fixed")}
                              if name == "csr_nnz_tile_reduce" else {}),
    }


def list_algorithms(op=None, dtype=None, backend=None):
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    result = []
    for name in ALGORITHMS:
        spec = algorithm_spec(name)
        if op is not None and op not in spec["ops"]:
            continue
        if dtype is not None and dtype_name(dtype) not in spec["value_dtypes"]:
            continue
        if backend is not None and backend not in spec["backends"]:
            continue
        result.append(name)
    return tuple(result)


def validate_support(alg, op, dtype, indices_dtype, indptr_dtype, caps):
    spec = algorithm_spec(alg)
    if alg in ("csr_rowgroup_stream_reduce", "csr_row_refill_reduce", "csr_nnz_local_reduce_atomic") and not caps.scan:
        raise BackendCapabilityUnavailable("rowgroup reduction requires scan capability")
    if alg in ("csr_transpose_sharded_atomic", "csr_nnz_local_reduce_atomic") + LONGROW_ALGORITHMS:
        feature = "fp64_atomic" if compute_dtype(alg, dtype) in ("float64", "complex128") else "fp32_atomic"
        if getattr(caps, feature) is not True:
            raise BackendCapabilityUnavailable(f"{alg} requires known {feature} capability")
    if alg in LONGROW_ALGORITHMS + ROWQUEUE_ALGORITHMS + COOPERATIVE_ALGORITHMS + THREE_TIER_ALGORITHMS and (caps.int32_atomic is not True or not caps.scan):
        raise BackendCapabilityUnavailable("long-row queue requires known integer atomic and scan capabilities")
    for value, key in (
        (op, "ops"),
        (dtype_name(dtype), "value_dtypes"),
        (dtype_name(indices_dtype), "index_dtypes"),
        (dtype_name(indptr_dtype), "indptr_dtypes"),
        (caps.backend, "backends"),
    ):
        if value not in spec[key]:
            raise NotImplementedError(f"CSR SpMV {alg} does not support {key}={value}")
    if alg in NEW_ALGORITHMS and not (
        caps.int64
        and caps.reduction
        and caps.subgroup_width in (32, 64)
        and caps.max_threads_per_block > 0
    ):
        raise BackendCapabilityUnavailable(
            f"CSR SpMV {alg}: unverified capabilities for {caps.backend}/{caps.arch}"
        )
    if (
        alg in NEW_ALGORITHMS
        and compute_dtype(alg, dtype) in ("float64", "complex128")
        and not caps.fp64
    ):
        raise BackendCapabilityUnavailable(
            f"CSR SpMV {alg}: FP64 capability required for {dtype}"
        )


def _defaults(caps):
    return {
        "short_row_threshold": 32,
        "split_row_threshold": 1024,
        "process": {"block_size": 256, "num_warps": 4},
        "row_tile": {
            "rows_per_program": 2 * caps.subgroup_width // 8,
            "lanes_per_row": 8,
            "num_warps": 2,
            "loop_num_stages": 1,
        },
        "row_subgroup_stream": {
            "rows_per_program": 2 * caps.subgroup_width // 8,
            "lanes_per_row": 8, "num_warps": 2,
            "row_tiles_per_program": 4, "loop_num_stages": 1,
        },
        "row_vector": {"block_nnz": 128, "num_warps": 2, "loop_num_stages": 1},
        "row_split_reduce": {
            "segment_nnz": 1024,
            "block_nnz": 128,
            "num_warps": 2,
            "reduce_block_size": 256,
            "reduce_num_warps": 4,
            "loop_num_stages": 1,
        },
    }


# Architecture-specific overrides may be added only with a recorded benchmark.
# On gfx936, 64 independent rows in a single wave hide FP64 reduction latency
# better than the conservative 16-row/two-wave default on the delivery corpus.
ARCH_PROFILES = {
    ("rocm", "gfx936"): {
        "row_tile": {"rows_per_program": 64, "num_warps": 1},
    },
}


def _merge(base, overrides):
    if not isinstance(overrides, dict):
        raise TypeError("CSR SpMV config must be a dictionary")
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"unknown CSR SpMV config field {key!r}")
        if isinstance(base[key], dict):
            _merge(base[key], value)
        else:
            if key == "plan_mode":
                if value not in ("compact", "fixed"):
                    raise ValueError("plan_mode must be compact or fixed")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"config {key} must be an integer")
            base[key] = value


def _validate_config(resolved, caps):
    if (
        resolved["short_row_threshold"] < 0
        or resolved["split_row_threshold"] <= resolved["short_row_threshold"]
    ):
        raise ValueError("require 0 <= short_row_threshold < split_row_threshold")
    for section in ("row_tile", "row_vector", "row_subgroup_stream", "row_split_reduce", "process"):
        for key, value in resolved[section].items():
            if value <= 0:
                raise ValueError(f"{section}.{key} must be positive")
            if key != "loop_num_stages" and value & (value - 1):
                raise ValueError(f"{section}.{key} must be a power of two")
            if key.endswith("num_warps"):
                if (
                    value not in (1, 2, 4, 8, 16, 32)
                    or value * caps.subgroup_width > caps.max_threads_per_block
                ):
                    raise ValueError(f"{section}.{key} exceeds backend launch limits")
                if caps.backend == "rocm" and value > 8:
                    raise ValueError("ROCm CSR SpMV profile limits num_warps to 8")
    tile = resolved["row_tile"]
    if tile["lanes_per_row"] > caps.subgroup_width:
        raise ValueError("lanes_per_row exceeds subgroup width")
    if tile["rows_per_program"] * tile["lanes_per_row"] > 65536:
        raise ValueError("row_tile exceeds the supported tile size")
    stream = resolved["row_subgroup_stream"]
    if stream["lanes_per_row"] > caps.subgroup_width:
        raise ValueError("row_subgroup_stream lanes_per_row exceeds subgroup width")
    if stream["rows_per_program"] * stream["lanes_per_row"] > 65536:
        raise ValueError("row_subgroup_stream exceeds the supported tile size")
    if "csr_subgroup_pair_mixed" in resolved:
        pair = resolved["csr_subgroup_pair_mixed"]
        if pair["prefetch_chunks"] not in (1, 2):
            raise ValueError("prefetch_chunks must be 1 or 2")
        if pair["loop_num_stages"] != 1 or pair["row_tiles_per_program"] not in (1, 2, 4, 8):
            raise ValueError("mixed subgroup requires stage=1 and 1/2/4/8 row tiles")
        for key, value in pair.items():
            if type(value) is not int or value <= 0 or value & (value - 1):
                raise ValueError(f"invalid mixed subgroup {key}")
        if (pair["lanes_per_row"] > caps.subgroup_width or
                pair["rows_per_program"] * pair["lanes_per_row"] > 65536 or
                pair["num_warps"] not in (1, 2, 4, 8) or
                pair["num_warps"] * caps.subgroup_width > caps.max_threads_per_block):
            raise ValueError("mixed subgroup exceeds target launch/tile limits")
    split = resolved["row_split_reduce"]
    if split["reduce_block_size"] < 2:
        raise ValueError("reduce_block_size must be >= 2 to make reduction progress")
    if (
        max(
            split["block_nnz"],
            split["reduce_block_size"],
            resolved["row_vector"]["block_nnz"],
            resolved["process"]["block_size"],
        )
        > 65536
    ):
        raise ValueError("block size exceeds the supported tile size")


def short_rowqueue_capacity(rows, nnz, config):
    """Every queued row owns at least S+1 NNZ; no GPU count readback needed."""
    if type(rows) is not int or type(nnz) is not int or rows < 0 or nnz < 0:
        raise ValueError("row-queue dimensions must be nonnegative integers")
    capacity = min(rows, nnz // (config["short_row_threshold"] + 1))
    if capacity > 2**31 - 1:
        raise ValueError("row-queue count exceeds the supported int32 counter range")
    descriptor_bytes = capacity * 8 + 4
    if descriptor_bytes > config["workspace_limit_bytes"]:
        raise ValueError(f"row-queue workspace requires {descriptor_bytes} bytes")
    return dict(row_queue_capacity=capacity, descriptor_peak_bytes=descriptor_bytes,
                numeric_workspace_bytes=0, workspace_peak_bytes=descriptor_bytes)


def cooperative_queue_capacity(rows, nnz, config, dtype):
    """Bound row IDs and fixed partial slots from dimensions, without device reads."""
    if type(rows) is not int or type(nnz) is not int or rows < 0 or nnz < 0:
        raise ValueError("cooperative queue dimensions must be nonnegative integers")
    capacity = min(rows, nnz // (config["long_row_threshold"] + 1))
    if capacity > 2**31 - 1:
        raise ValueError("cooperative queue count exceeds the supported int32 counter range")
    dt = dtype_name(dtype)
    if dt not in VALUE_DTYPES:
        raise ValueError(f"unsupported cooperative dtype {dt}")
    component_bytes = 8 if dt in ("float32", "float64", "complex128") else 4
    components = 2 if dt.startswith("complex") else 1
    slots = capacity * config["cooperative_parts"]
    numeric = slots * components * component_bytes
    descriptors = capacity * 8 + 4
    if numeric + descriptors > config["workspace_limit_bytes"]:
        raise ValueError(f"cooperative workspace requires {numeric + descriptors} bytes")
    return dict(row_queue_capacity=capacity, partial_slot_capacity=slots,
                numeric_workspace_bytes=numeric, descriptor_peak_bytes=descriptors,
                workspace_peak_bytes=numeric + descriptors)


def three_tier_queue_capacity(rows, nnz, config, dtype):
    """Conservative capacities, without reading either GPU queue count."""
    if any(type(v) is not int or v < 0 for v in (rows, nnz)):
        raise ValueError("three-tier queue dimensions must be nonnegative integers")
    medium = min(rows, nnz // (config["short_row_threshold"] + 1))
    long = min(rows, nnz // (config["long_row_threshold"] + 1))
    if max(medium, long) > 2**31 - 1:
        raise ValueError("three-tier queue count exceeds the supported int32 counter range")
    dt = dtype_name(dtype)
    if dt not in VALUE_DTYPES:
        raise ValueError(f"unsupported three-tier dtype {dt}")
    component_bytes = 8 if dt in ("float32", "float64", "complex128") else 4
    components = 2 if dt.startswith("complex") else 1
    slots = long * config["cooperative_parts"]
    numeric = slots * component_bytes * components
    descriptors = 8 + 8 * (medium + long)
    if numeric + descriptors > config["workspace_limit_bytes"]:
        raise ValueError(f"three-tier workspace requires {numeric + descriptors} bytes")
    return dict(medium_queue_capacity=medium, long_queue_capacity=long,
                partial_slot_capacity=slots, numeric_workspace_bytes=numeric,
                descriptor_peak_bytes=descriptors, workspace_peak_bytes=numeric + descriptors)


def resolve_config(alg, caps, config=None, *, return_rejections=False, mean_row_nnz=0):
    if config is not None and not isinstance(config, dict):
        raise TypeError("CSR SpMV config must be a dictionary")
    if alg in THREE_TIER_ALGORITHMS:
        if (caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64)
                or not caps.scan or caps.int32_atomic is not True):
            raise BackendCapabilityUnavailable("three-tier queue requires known scan/integer-atomic/launch capabilities")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal three-tier queue launch")
        cfg = dict(rows_per_program=128, short_row_threshold=32, long_row_threshold=1024,
                   block_nnz=128, cooperative_parts=4, long_grid_blocks=128,
                   producer_num_warps=max((w for w in warps if w <= 2), default=min(warps)),
                   num_warps=max((w for w in warps if w <= 4), default=min(warps)),
                   num_stages=1, workspace_limit_bytes=268435456)
        supplied = config or {}
        if set(supplied) - set(cfg):
            raise ValueError(f"unused three-tier config fields: {sorted(set(supplied) - set(cfg))}")
        cfg.update(supplied)
        if (any(type(v) is not int for v in cfg.values()) or
                cfg["rows_per_program"] not in (32, 64, 128) or
                cfg["short_row_threshold"] not in (0, 8, 16, 32, 64, 128) or
                not cfg["short_row_threshold"] < cfg["long_row_threshold"] <= 1048576 or
                cfg["block_nnz"] not in (64, 128, 256) or
                cfg["cooperative_parts"] not in (1, 2, 4, 8, 16) or
                not 1 <= cfg["long_grid_blocks"] <= 1024 or
                cfg["producer_num_warps"] not in warps or cfg["num_warps"] not in warps or
                cfg["num_stages"] != 1 or cfg["workspace_limit_bytes"] <= 0):
            raise ValueError("invalid three-tier queue configuration")
        result = (cfg, "explicit" if supplied else "three-tier-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg in COOPERATIVE_ALGORITHMS:
        if (caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64)
                or not caps.scan or caps.int32_atomic is not True):
            raise BackendCapabilityUnavailable("cooperative queue requires known scan/integer-atomic/launch capabilities")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal cooperative queue launch")
        cfg = dict(rows_per_program=8, lanes_per_row=16, long_row_threshold=1024,
                   cooperative_parts=8, grid_blocks=128, block_nnz=128,
                   producer_num_warps=max((w for w in warps if w <= 2), default=min(warps)),
                   num_warps=max((w for w in warps if w <= 4), default=min(warps)),
                   num_stages=1, workspace_limit_bytes=268435456)
        supplied = config or {}
        if set(supplied) - set(cfg):
            raise ValueError(f"unused cooperative config fields: {sorted(set(supplied) - set(cfg))}")
        cfg.update(supplied)
        if (any(type(v) is not int for v in cfg.values()) or
                cfg["rows_per_program"] not in (4, 8, 16, 32) or
                cfg["lanes_per_row"] not in (2, 4, 8, 16, 32, 64) or
                cfg["lanes_per_row"] > caps.subgroup_width or
                not 0 <= cfg["long_row_threshold"] <= 1048576 or
                cfg["cooperative_parts"] not in (1, 2, 4, 8, 16) or
                not 1 <= cfg["grid_blocks"] <= 1024 or cfg["block_nnz"] not in (32, 64, 128, 256) or
                cfg["num_warps"] not in warps or cfg["producer_num_warps"] not in warps or
                cfg["num_stages"] != 1 or cfg["workspace_limit_bytes"] <= 0):
            raise ValueError("invalid cooperative queue configuration")
        result = (cfg, "explicit" if supplied else "longrow-cooperative-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg == "csr_short_rowqueue_strided_vector":
        supplied = dict(config or {})
        grid_blocks = supplied.pop("grid_blocks", 256)
        if type(grid_blocks) is not int or not 1 <= grid_blocks <= 1024:
            raise ValueError("grid_blocks must be an integer between 1 and 1024")
        supplied.setdefault("short_row_threshold", 128)
        cfg, _, rejected = resolve_config("csr_short_rowqueue_vector", caps, supplied,
                                          return_rejections=True)
        cfg["grid_blocks"] = grid_blocks
        result = (cfg, "explicit" if config else "bounded-rowqueue-conservative-v1")
        return result + (rejected,) if return_rejections else result
    if alg in ROWQUEUE_ALGORITHMS:
        if (caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64)
                or not caps.scan or caps.int32_atomic is not True):
            raise BackendCapabilityUnavailable("row queue requires known scan/integer-atomic/launch capabilities")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal row-queue launch")
        cfg = dict(rows_per_program=64, short_row_threshold=32, block_nnz=128,
                   producer_num_warps=max((w for w in warps if w <= 2), default=min(warps)),
                   num_warps=max((w for w in warps if w <= 4), default=min(warps)),
                   num_stages=1, workspace_limit_bytes=268435456)
        supplied = config or {}
        if set(supplied) - set(cfg):
            raise ValueError(f"unused row-queue config fields: {sorted(set(supplied) - set(cfg))}")
        cfg.update(supplied)
        if (any(type(v) is not int for v in cfg.values()) or
                cfg["rows_per_program"] not in (32, 64, 128) or
                cfg["short_row_threshold"] not in (0, 8, 16, 32, 64, 128, 256) or
                cfg["block_nnz"] not in (64, 128, 256) or cfg["num_stages"] != 1 or
                cfg["num_warps"] not in warps or cfg["producer_num_warps"] not in warps or
                cfg["workspace_limit_bytes"] <= 0):
            raise ValueError("invalid row-queue configuration")
        result = (cfg, "explicit" if supplied else "short-rowqueue-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg == "csr_longrow_grid_atomic":
        if config and "worker_programs" in config:
            raise ValueError("worker_programs is unused by the static long-row grid")
        cfg, source, rejected = resolve_config("csr_longrow_queue_atomic", caps, config, return_rejections=True)
        cfg.pop("worker_programs")
        result = (cfg, "explicit" if config else "long-row-grid-conservative-v1")
        return result + (rejected,) if return_rejections else result
    if alg == "csr_nnz_local_reduce_atomic":
        if caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64) or not caps.scan:
            raise BackendCapabilityUnavailable("NNZ local reduction requires known scan/launch capabilities")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal NNZ local reduction launch")
        cfg = dict(block_nnz=256, owner_search="bounded", single_row_fastpath=True,
                   atomic_mode="boundary", num_warps=max((w for w in warps if w <= 4), default=min(warps)), num_stages=1)
        supplied = config or {}
        if set(supplied) - set(cfg):
            raise ValueError(f"unused NNZ local config fields: {sorted(set(supplied) - set(cfg))}")
        cfg.update(supplied)
        if (any(type(cfg[k]) is not int for k in ("block_nnz", "num_warps", "num_stages")) or
                cfg["block_nnz"] not in (128, 256, 512, 1024) or cfg["num_warps"] not in warps or
                cfg["num_stages"] != 1 or type(cfg["single_row_fastpath"]) is not bool or
                cfg["owner_search"] not in ("bounded", "full") or cfg["atomic_mode"] not in ("boundary", "all")):
            raise ValueError("invalid NNZ local reduction configuration")
        result = (cfg, "explicit" if supplied else "nnz-local-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg in SHORT_ROW_ALGORITHMS:
        if caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64):
            raise BackendCapabilityUnavailable("short-row launch capabilities unknown")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal short-row launch")
        tail = alg == "csr_row_scalar_tail_reduce"
        scalar = alg in ("csr_row_scalar_strip", "csr_row_scalar_bound") or tail
        cfg = dict(rows_per_program=64 if scalar else 8,
                   num_warps=max((w for w in warps if w <= 2), default=min(warps)), num_stages=1)
        if not scalar:
            cfg["block_nnz"] = 16
        if tail:
            cfg.update(scalar_steps=8, block_nnz=32)
        supplied = config or {}
        if set(supplied) - set(cfg):
            raise ValueError(f"unused {alg} config fields: {sorted(set(supplied) - set(cfg))}")
        cfg.update(supplied)
        if any(type(v) is not int for v in cfg.values()):
            raise ValueError("short-row configuration values must be integers")
        if (cfg["num_warps"] not in warps or cfg["num_stages"] != 1 or
                cfg["rows_per_program"] not in ((32, 64, 128) if scalar else (4, 8, 16)) or
                ((not scalar or tail) and cfg["block_nnz"] not in (8, 16, 32, 64)) or
                (tail and cfg["scalar_steps"] not in (0, 4, 8, 16, 32))):
            raise ValueError("invalid short-row tile or launch configuration")
        result = (cfg, "explicit" if supplied else "scalar-tail-conservative-v1" if tail else "short-row-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg == "csr_longrow_queue_atomic":
        if caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64):
            raise BackendCapabilityUnavailable("long-row queue launch capabilities unknown")
        if caps.int32_atomic is not True or not caps.scan:
            raise BackendCapabilityUnavailable("long-row queue requires integer atomic and scan")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal queue launch")
        warp = max((w for w in warps if w <= 4), default=min(warps))
        cfg = dict(long_row_threshold=1024, segment_nnz=1024, rows_per_program=8,
                   short_block_nnz=32, block_nnz=128, producer_num_warps=warp,
                   num_warps=warp, worker_programs=128, num_stages=1,
                   workspace_limit_bytes=268435456)
        _merge(cfg, config or {})
        if (any(v <= 0 for v in cfg.values()) or cfg["num_stages"] != 1 or
                cfg["producer_num_warps"] not in warps or cfg["num_warps"] not in warps or
                cfg["rows_per_program"] not in (4, 8, 16) or
                cfg["short_block_nnz"] not in (8, 16, 32, 64) or
                cfg["block_nnz"] not in (32, 64, 128, 256) or
                cfg["worker_programs"] not in (32, 64, 128, 256) or
                any(cfg[k] & (cfg[k] - 1) for k in ("long_row_threshold", "segment_nnz")) or
                cfg["segment_nnz"] % cfg["block_nnz"]):
            raise ValueError("invalid long-row queue configuration")
        result = (cfg, "explicit" if config else "long-row-queue-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg in TARGETED_ALGORITHMS:
        if caps.backend not in ("cuda", "rocm") or caps.subgroup_width not in (32, 64):
            raise BackendCapabilityUnavailable("targeted CSR launch capabilities unknown")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps:
            raise BackendCapabilityUnavailable("no legal targeted CSR launch")
        refill = alg == "csr_row_refill_reduce"
        preferred = 2 if refill else 4
        warp = max((w for w in warps if w <= preferred), default=min(warps))
        cfg = (dict(rows_per_program=warp * caps.subgroup_width // 8,
                    lanes_per_row=8, window_rows=128, num_warps=warp, num_stages=1)
               if refill else dict(output_shards=4, rows_per_program=4, block_nnz=32,
                                   num_warps=warp, num_stages=1, workspace_limit_bytes=268435456))
        supplied = config or {}
        if set(supplied) - set(cfg):
            raise ValueError(f"unused {alg} config fields: {sorted(set(supplied) - set(cfg))}")
        cfg.update(supplied)
        if any(type(v) is not int for v in cfg.values()):
            raise ValueError("targeted CSR configuration values must be integers")
        if cfg["num_warps"] not in warps or cfg["num_stages"] != 1:
            raise ValueError("invalid targeted CSR launch configuration")
        if refill:
            if not caps.scan:
                raise BackendCapabilityUnavailable("refill requires integer scan")
            if "rows_per_program" not in supplied:
                cfg["rows_per_program"] = cfg["num_warps"] * caps.subgroup_width // cfg["lanes_per_row"] if cfg["lanes_per_row"] > 0 else 0
            r, v, w = (cfg[k] for k in ("rows_per_program", "lanes_per_row", "window_rows"))
            if r not in (4, 8, 16, 32) or v not in (2, 4, 8, 16, 32, 64) or v > caps.subgroup_width:
                raise ValueError("invalid refill row/lane tile")
            if not r <= w <= 1024 or w % r:
                raise ValueError("window_rows must be a row-tile multiple in [R,1024]")
        elif (cfg["output_shards"] not in (1, 2, 4, 8) or cfg["rows_per_program"] not in (1, 2, 4, 8)
              or cfg["block_nnz"] not in (8, 16, 32, 64) or cfg["workspace_limit_bytes"] <= 0):
            raise ValueError("invalid sharded scatter configuration")
        result = (cfg, "explicit" if supplied else "targeted-conservative-v1")
        return result + ([],) if return_rejections else result
    if alg == "csr_rowgroup_stream_reduce":
        if not caps.scan:
            raise BackendCapabilityUnavailable("rowgroup reduction requires scan capability")
        warps = [w for w in (1, 2, 4, 8) if w * caps.subgroup_width <= caps.max_threads_per_block]
        if not warps or caps.subgroup_width not in (32, 64):
            raise BackendCapabilityUnavailable("rowgroup launch capabilities unknown")
        resolved = dict(rows_per_program=16, block_nnz=128,
                        num_warps=max(w for w in warps if w <= 4), num_stages=1)
        _merge(resolved, config or {})
        if (resolved["rows_per_program"] not in (8, 16, 32)
                or resolved["block_nnz"] not in (64, 128, 256)
                or resolved["num_warps"] not in warps or resolved["num_stages"] != 1):
            raise ValueError("invalid rowgroup tile or launch configuration")
        result = (resolved, "explicit" if config else "conservative")
        return result + ([],) if return_rejections else result
    if alg == "csr_nnz_tile_reduce":
        if caps.backend not in ("cuda", "rocm") or not (caps.int64 and caps.reduction and caps.subgroup_width in (32, 64)):
            raise BackendCapabilityUnavailable("NNZ tiles require verified CUDA/ROCm scan and index capabilities")
        resolved = dict(block_nnz=128, num_warps=4, num_stages=1,
                        reduce_block_size=256, workspace_bytes=256 * 1024 * 1024, plan_mode="compact")
        _merge(resolved, config or {})
        for key in ("block_nnz", "reduce_block_size", "num_warps"):
            value = resolved[key]
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{key} must be a positive power of two")
        if not 2 <= resolved["reduce_block_size"] <= 256 or resolved["block_nnz"] > 4096:
            raise ValueError("NNZ tile/reduction block exceeds supported limits")
        if resolved["num_warps"] not in (1, 2, 4, 8) or resolved["num_warps"] * caps.subgroup_width > caps.max_threads_per_block:
            raise ValueError("illegal NNZ tile launch")
        if resolved["num_stages"] != 1 or resolved["workspace_bytes"] < 36:
            raise ValueError("NNZ tiles require stage=1 and at least 36 workspace bytes")
        if resolved["plan_mode"] == "fixed" and resolved["reduce_block_size"] < 4:
            raise ValueError("fixed reduction requires reduce_block_size >= 4")
        source = "explicit" if config else "conservative-nnz-v1"
        return (resolved, source, []) if return_rejections else (resolved, source)
    if config is not None and not isinstance(config, dict):
        raise TypeError("CSR SpMV config must be a dictionary")
    if alg not in NEW_ALGORITHMS:
        if config:
            raise ValueError(
                "legacy CSR SpMV routes do not accept new-algorithm config"
            )
        return ({}, "legacy", []) if return_rejections else ({}, "legacy")
    if (
        caps.backend not in ("cuda", "rocm")
        or caps.subgroup_width not in (32, 64)
        or not (caps.int64 and caps.reduction and caps.max_threads_per_block > 0)
    ):
        raise BackendCapabilityUnavailable(
            f"no verified CSR SpMV profile for {caps.backend}/{caps.arch}"
        )
    candidates = []
    profile = ARCH_PROFILES.get((caps.backend, caps.arch))
    if alg == "csr_subgroup_pair_mixed":
        profile = {alg: profile[alg]} if profile and alg in profile else None
    if profile is not None:
        candidates.append((f"{caps.backend}:{caps.arch}", profile))
    candidates.append((f"{caps.backend}:conservative-v1", {}))
    rejections = []
    resolved = None
    source = None
    for name, overrides in candidates:
        candidate = _defaults(caps)
        try:
            _merge(candidate, overrides)
            _validate_config(candidate, caps)
        except (ValueError, TypeError) as exc:
            rejections.append({"profile": name, "reason": str(exc)})
            continue
        resolved, source = candidate, name
        break
    if resolved is None:
        # Internal candidate may shrink launch groups to available resources.
        # Explicit user values below are always validated without modification.
        resolved = _defaults(caps)
        available = caps.max_threads_per_block // caps.subgroup_width
        warps = max((w for w in (1, 2, 4, 8) if w <= available), default=0)
        if not warps:
            raise BackendCapabilityUnavailable(f"no legal launch profile: {rejections}")
        for section in ("row_tile", "row_vector", "row_subgroup_stream", "row_split_reduce", "process"):
            resolved[section]["num_warps"] = min(resolved[section]["num_warps"], warps)
        resolved["row_split_reduce"]["reduce_num_warps"] = min(4, warps)
        _validate_config(resolved, caps)
        source = f"{caps.backend}:resource-conservative-v1"
    if alg == "csr_subgroup_pair_mixed":
        if config is not None and set(config) - {alg}:
            raise ValueError("mixed subgroup config must use its own algorithm section")
        resolved[alg] = dict(resolved["row_subgroup_stream"], prefetch_chunks=2)
        resolved[alg]["rows_per_program"] = resolved[alg]["num_warps"] * caps.subgroup_width // 8
    if alg == "row_subgroup_stream":
        stream = resolved[alg]
        lanes = next((v for bound, v in ((4, 2), (8, 4), (16, 8), (32, 16), (64, 32))
                      if mean_row_nnz < bound), 64)
        stream["lanes_per_row"] = min(lanes, caps.subgroup_width)
        stream["rows_per_program"] = stream["num_warps"] * caps.subgroup_width // stream["lanes_per_row"]
    if config is not None:
        _merge(resolved, config)
        _validate_config(resolved, caps)
        if alg in ("row_subgroup_stream", "csr_subgroup_pair_mixed"):
            override = config.get(alg, {})
            stream = resolved[alg]
            if "rows_per_program" not in override and any(k in override for k in ("lanes_per_row", "num_warps")):
                stream["rows_per_program"] = stream["num_warps"] * caps.subgroup_width // stream["lanes_per_row"]
            _validate_config(resolved, caps)
        source = "explicit"
    result = (deepcopy(resolved), source)
    return (*result, rejections) if return_rejections else result


def assert_route_match(prepared_alg, prepared_config, alg=None, config=None, caps=None, mean_row_nnz=0):
    if alg is not None and normalize_alg(alg) not in ("auto", prepared_alg):
        raise ValueError(f"alg does not match prepared.alg={prepared_alg}")
    if config is not None:
        candidate, _ = resolve_config(prepared_alg, caps, config, mean_row_nnz=mean_row_nnz)
        if candidate != prepared_config:
            raise ValueError("config does not match prepared.config")


def is_index_compatibility_error(exc):
    """Never reinterpret OOM, invalid memory access, or arbitrary failures as index errors."""
    text = str(exc).lower()
    return (
        isinstance(exc, (RuntimeError, TypeError, NotImplementedError))
        and any(
            token in text for token in ("int64", "i64", "64-bit index", "64 bit index")
        )
        and any(
            token in text
            for token in (
                "unsupported",
                "not supported",
                "cannot lower",
                "not implemented",
            )
        )
    )


def longrow_queue_capacity(rows, nnz, config, component_bytes, channels):
    """Host-shape upper bounds; no tensor reads or runtime queue count synchronization."""
    if rows < 0 or nnz < 0 or component_bytes not in (4, 8) or channels not in (1, 2):
        raise ValueError("invalid queue shape or component type")
    long_capacity = min(rows, nnz // (config["long_row_threshold"] + 1))
    jobs = nnz // config["segment_nnz"] + long_capacity if long_capacity else 0
    limit = (1 << 31) - 1
    if long_capacity > limit or jobs + config["worker_programs"] > limit:
        raise ValueError("queue capacity exceeds safe int32 atomic counter range")
    descriptor_bytes = jobs * 24 + long_capacity * 8 + 12
    numeric_bytes = long_capacity * component_bytes * channels
    if descriptor_bytes + numeric_bytes > config["workspace_limit_bytes"]:
        raise ValueError("long-row queue workspace exceeds workspace_limit_bytes")
    return dict(long_row_capacity=long_capacity, job_capacity=jobs,
                descriptor_bytes=descriptor_bytes, numeric_workspace_bytes=numeric_bytes,
                workspace_peak_bytes=descriptor_bytes + numeric_bytes)
