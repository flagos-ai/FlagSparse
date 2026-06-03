"""Run SpSV benchmark. From project root: python benchmark/benchmark_spsv.py [--synthetic | --csv-csr out.csv]."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from tests.test_spsv import main


if __name__ == "__main__":
    main()
