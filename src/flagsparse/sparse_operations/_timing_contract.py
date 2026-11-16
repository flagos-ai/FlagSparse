"""Keep full operator measurements independent of optional phase instrumentation."""
from functools import wraps


def separate_timing_diagnostics(run):
    """Collect phase timings in another invocation, preserving the public tuple API.

    The decorated implementation retains its normal untimed execution and full
    GPU timer. Only callers requesting phase metadata pay for a diagnostic run.
    CPU/GPU primary values always come from the same uninstrumented invocation.
    """
    @wraps(run)
    def measured(*args, **kwargs):
        if not kwargs.get("timing", False):
            return run(*args, **kwargs)
        primary = dict(kwargs, timing=False)
        if not kwargs.get("return_meta", False):
            return run(*args, **primary)
        primary.update(return_time=True, return_meta=True)
        result, elapsed, meta = run(*args, **primary)
        diagnostics = dict(kwargs, timing=True, return_time=False, return_meta=True)
        _, phases = run(*args, **diagnostics)
        for key in ("process_gpu_ms", "compute_ms", "symbolic_ms"):
            if key in phases:
                meta[key] = phases[key]
        meta["phase_diagnostics"] = "separately_measured"
        meta["op_total_ms"] = elapsed
        return (result, elapsed, meta) if kwargs.get("return_time", False) else (result, meta)
    return measured
