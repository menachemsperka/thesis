"""Legacy entry point — delegates to run_cross_benchmark_comparison.py."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_cross_benchmark_comparison import main

if __name__ == "__main__":
    main()
