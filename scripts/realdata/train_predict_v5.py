"""Entry point: train PredictV5 recurrent time-series expert."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realdata_experts.predict_v5.train import main


if __name__ == "__main__":
    main()
