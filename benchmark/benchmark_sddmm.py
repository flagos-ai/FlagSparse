"""Run SDDMM benchmark (.mtx batch). From project root: python benchmark/benchmark_sddmm.py <dir/> --csv results.csv."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from tests.test_sddmm import main

if __name__ == "__main__":
    main()
