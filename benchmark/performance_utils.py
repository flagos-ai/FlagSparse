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

"""Shared pytest benchmark utilities for FlagSparse performance suites."""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional


DEFAULT_METRICS = ("latency_base", "latency", "speedup")


@dataclass(frozen=True)
class BenchmarkMetrics:
    """One benchmark measurement summary."""

    op: str
    dtype: str
    shape: str
    latency_base: float
    latency: float
    speedup: float
    extras: Mapping[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "op": self.op,
            "dtype": self.dtype,
            "shape": self.shape,
            "latency_base": self.latency_base,
            "latency": self.latency,
            "speedup": self.speedup,
        }
        row.update(self.extras)
        return row


class PerformanceBenchmark:
    """Base class for pytest-style performance benchmarks.

    Subclasses should provide input construction and callables for the baseline
    and FlagSparse implementation. Timing boundaries should contain only the
    operator invocation, not data construction.
    """

    default_metrics = DEFAULT_METRICS

    def __init__(self, warmup: int = 5, iterations: int = 20):
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.warmup = warmup
        self.iterations = iterations

    @staticmethod
    def synchronize(device=None):
        try:
            import torch
        except Exception:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def measure(self, fn: Callable[[], object], *, device=None) -> float:
        for _ in range(self.warmup):
            fn()
        self.synchronize(device)

        samples: List[float] = []
        for _ in range(self.iterations):
            self.synchronize(device)
            start = time.perf_counter()
            fn()
            self.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
        return statistics.median(samples)

    def compare(
        self,
        *,
        op: str,
        dtype: str,
        shape: str,
        baseline: Callable[[], object],
        candidate: Callable[[], object],
        device=None,
        extras: Optional[Mapping[str, float]] = None,
    ) -> BenchmarkMetrics:
        latency_base = self.measure(baseline, device=device)
        latency = self.measure(candidate, device=device)
        speedup = latency_base / latency if latency > 0 else float("inf")
        return BenchmarkMetrics(
            op=op,
            dtype=dtype,
            shape=shape,
            latency_base=latency_base,
            latency=latency,
            speedup=speedup,
            extras=extras or {},
        )


def write_metrics_csv(path: Path, rows: Iterable[BenchmarkMetrics]):
    materialized = [row.as_row() for row in rows]
    fieldnames = list(DEFAULT_METRICS)
    prefix = ["op", "dtype", "shape"]
    extra_fields = sorted(
        {key for row in materialized for key in row.keys()}
        - set(prefix)
        - set(fieldnames)
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*prefix, *fieldnames, *extra_fields])
        writer.writeheader()
        writer.writerows(materialized)


def read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def two_level_average_speedup(rows: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    """Average speedup by shape within dtype, then average across dtypes."""
    by_dtype_shape: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        dtype = str(row["dtype"])
        shape = str(row["shape"])
        speedup = float(row["speedup"])
        by_dtype_shape.setdefault(dtype, {}).setdefault(shape, []).append(speedup)

    dtype_averages = {}
    for dtype, by_shape in by_dtype_shape.items():
        shape_averages = [statistics.mean(values) for values in by_shape.values()]
        dtype_averages[dtype] = statistics.mean(shape_averages)

    overall = statistics.mean(dtype_averages.values()) if dtype_averages else 0.0
    return {"overall": overall, **dtype_averages}
