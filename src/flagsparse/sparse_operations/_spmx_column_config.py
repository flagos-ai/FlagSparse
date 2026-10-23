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

"""COO/CSC extension launch policies; backend identities come from main."""

from . import _spmm_csr_config as _csr_policy

DTYPES = _csr_policy.DTYPES
# COO shares backend capability contracts; kernels remain in the COO modules.
COO_ALGORITHMS = ("coo_segmented_atomic", "coo_rowrun_subgroup",
                  "coo_segmented_panel_atomic", "coo_rowrun_panel",
                  "coo_segmented_stream_atomic", "coo_segmented_stream_panel_atomic")
COO_ARCH_PROFILES = {}
COO_BACKEND_PROFILES = {name: {} for name in _csr_policy.BACKENDS}


def resolve_coo_config(alg, dtype, caps, config=None, n=1):
    if alg not in COO_ALGORITHMS:
        raise ValueError(f"unknown COO extension {alg!r}")
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if caps.backend not in _csr_policy.BACKENDS or not caps.legal_num_warps or not caps.subgroup_width or caps.int64 is not True:
        raise _csr_policy.BackendCapabilityUnavailable(f"COO launch/index capabilities unknown: {caps}")
    atomic = alg in ("coo_segmented_atomic", "coo_segmented_panel_atomic",
                     "coo_segmented_stream_atomic", "coo_segmented_stream_panel_atomic")
    overrides = dict(config or {})
    cfg = dict(num_warps=4 if atomic else 2, num_stages=1)
    if atomic:
        cfg.update(block_nnz=128 if alg in ("coo_segmented_atomic", "coo_segmented_stream_atomic") else 32,
                   local_reduce="segment")
        if alg in ("coo_segmented_panel_atomic", "coo_segmented_stream_panel_atomic"):
            cfg["block_n"] = min(16, (1 << (max(1, n) - 1).bit_length()))
    elif alg == "coo_rowrun_subgroup":
        cfg.update(lanes_per_row=8, rows_per_program=1)
    else:
        cfg.update(tile_rows=4, tile_n=min(16, (1 << (max(1, n) - 1).bit_length())))
    if alg in ("coo_segmented_stream_atomic", "coo_segmented_stream_panel_atomic"):
        cfg["blocks_per_program"] = 2
    if alg in ("coo_segmented_panel_atomic", "coo_segmented_stream_panel_atomic"):
        cfg["conversion_mode"] = "load" if alg == "coo_segmented_stream_panel_atomic" else "materialize"
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
                value in ("load", "materialize") if key == "conversion_mode" else
                type(value) is int and value > 0 and not value & (value - 1))
            valid = valid and (key != "blocks_per_program" or value in (1, 2, 4, 8))
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
        if key == "conversion_mode":
            if value not in ("load", "materialize"):
                raise ValueError("conversion_mode must be load or materialize")
        elif key == "local_reduce":
            if value not in ("segment", "none"):
                raise ValueError("local_reduce must be segment or none")
        elif type(value) is not int or value <= 0 or value & (value - 1):
            raise ValueError(f"{key} must be a positive power of two")
    if cfg.get("blocks_per_program", 1) not in (1, 2, 4, 8):
        raise ValueError("blocks_per_program must be 1, 2, 4 or 8")
    if (cfg["num_warps"] not in caps.legal_num_warps or cfg["num_stages"] != 1
            or cfg["num_warps"] * caps.subgroup_width > caps.max_threads_per_block):
        raise ValueError("illegal COO launch configuration")
    if alg == "coo_rowrun_subgroup" and "rows_per_program" not in overrides and "rows_per_program" not in accepted_keys:
        cfg["rows_per_program"] = cfg["num_warps"] * caps.subgroup_width // cfg["lanes_per_row"]
    if cfg.get("lanes_per_row", 1) > caps.subgroup_width:
        raise ValueError("lanes_per_row exceeds backend subgroup width")
    if dtype in ("float64", "complex128") and caps.fp64 is not True:
        raise _csr_policy.BackendCapabilityUnavailable("COO FP64 capability unavailable or unknown")
    if atomic:
        feature = "fp64_atomic" if dtype in ("float64", "complex128") else "fp32_atomic"
        supported = getattr(caps, feature)
        if supported is not True:
            raise _csr_policy.BackendCapabilityUnavailable(f"{feature} unavailable or unknown on {caps.backend}")
        if cfg["local_reduce"] == "segment" and caps.scan is not True:
            raise _csr_policy.BackendCapabilityUnavailable("COO segmented scan capability unknown")
    elif caps.stable_sort is not True or caps.reduction is not True:
        raise _csr_policy.BackendCapabilityUnavailable("COO stable sort/reduction capability unknown")
    return cfg, dict(backend=caps.backend, target=caps.target, arch=caps.arch,
                     config_source="explicit" if overrides else ("architecture" if arch_profile else "backend" if profile else "conservative"),
                     validation="unverified", config=cfg, rejected_profiles=rejected, implementation_version=2,
                     timing_contract_version=2)


CSC_ALGORITHMS = {
    "csc_col_vector": ("trans", "conj"),
    "csc_col_kparallel": ("trans", "conj"),
    "csc_col_split_reduce": ("trans", "conj"),
    "csc_col_subgroup": ("trans", "conj"),
    "csc_col_tile_atomic": ("non",),
    "csc_col_panel": ("trans", "conj"),
    "csc_col_tile_panel_atomic": ("non",),
}
CSC_ARCH_PROFILES = {}
CSC_BACKEND_PROFILES = {}


def resolve_csc_config(algorithm, dtype, n, caps, overrides=None, *, op="non", kind="spmm"):
    """Resolve native CSC launch policy without importing an accelerator runtime."""
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("config must be a dictionary")
    if algorithm not in CSC_ALGORITHMS:
        raise ValueError(f"unknown CSC algorithm {algorithm!r}")
    if op not in CSC_ALGORITHMS[algorithm]:
        raise ValueError(f"{algorithm} does not support op={op}")
    if dtype not in DTYPES:
        raise TypeError(f"unsupported CSC dtype {dtype}")
    if caps.backend not in _csr_policy.BACKENDS or not caps.legal_num_warps or not caps.subgroup_width:
        raise _csr_policy.BackendCapabilityUnavailable("CSC launch capabilities unavailable or unknown")
    required = ["int64"]
    atomic = algorithm in ("csc_col_tile_atomic", "csc_col_tile_panel_atomic")
    if dtype in ("float64", "complex128"):
        required.append("fp64")
    if atomic:
        required.append("fp64_atomic" if dtype in ("float64", "complex128") else "fp32_atomic")
    else:
        required.append("reduction")
    for key in required:
        if getattr(caps, key) is not True:
            raise _csr_policy.BackendCapabilityUnavailable(f"CSC {key} capability unavailable for {caps.backend}/{caps.arch}")
    if algorithm == "csc_col_split_reduce" and caps.scan is not True:
        raise _csr_policy.BackendCapabilityUnavailable("CSC GPU prefix-plan capability unavailable")
    warps = 4 if atomic or algorithm in ("csc_col_kparallel", "csc_col_split_reduce") else 2
    if warps not in caps.legal_num_warps:
        warps = max((w for w in caps.legal_num_warps if w <= warps), default=min(caps.legal_num_warps))
    bn = 8 if dtype == "complex128" and not atomic else 16
    cfg = dict(num_warps=warps, num_stages=1)
    if algorithm in ("csc_col_vector", "csc_col_kparallel", "csc_col_split_reduce"):
        cfg.update(block_nnz=128 if kind == "spmv" or algorithm == "csc_col_vector" else 32,
                   block_n=1 if kind == "spmv" or algorithm == "csc_col_vector" else min(bn, 1 << (max(1, n)-1).bit_length()))
        if algorithm == "csc_col_split_reduce":
            cfg.update(segment_nnz=1024, reduce_block_size=256, workspace_bytes=256 * 1024 * 1024)
    elif algorithm == "csc_col_subgroup":
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
            if key not in ("num_stages", "workspace_bytes") and value & (value-1):
                raise ValueError(f"{key} must be a power of two")
        if candidate["num_stages"] != 1:
            raise ValueError("CSC num_stages must be 1")
        if algorithm == "csc_col_split_reduce" and (not 2 <= candidate["reduce_block_size"] <= 256 or candidate["workspace_bytes"] < 64):
            raise ValueError("split needs 2 <= reduce_block_size <= 256 and workspace_bytes >= 64")
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
