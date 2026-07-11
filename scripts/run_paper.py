#!/usr/bin/env python3
"""Run the trading agent in paper mode.

  uv run python scripts/run_paper.py --once             # one cycle (live Base prices)
  uv run python scripts/run_paper.py --simulated --once # offline, synthetic data
  uv run python scripts/run_paper.py                    # scheduled loop
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
