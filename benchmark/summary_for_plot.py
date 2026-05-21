"""Summarize benchmark CSV/log records for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.performance_utils import read_metrics_csv, two_level_average_speedup


def summarize(path: Path):
    rows = read_metrics_csv(path)
    return two_level_average_speedup(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    args = parser.parse_args(argv)

    summary = summarize(args.result_file)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
