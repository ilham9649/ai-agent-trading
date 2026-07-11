#!/usr/bin/env python3
"""Leveraged multi-TA vs B&H with HONEST perp funding (charged on FULL notional).

Research finding: real perp funding runs ~10-15%/yr on BTC/ETH and ~20-40%/yr on
SOL/BNB in bull markets, charged on full notional — routinely omitted from retail
backtests. This re-tests leverage net of those real costs. Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402
from trading_agent.strategy.multi_ta_signal import multi_ta_signal  # noqa: E402

TOTAL = 3000
THRESHOLD = 4
LEVERAGES = [1.0, 2.0, 3.0]
FUNDING_LEVELS = [0.08, 0.20, 0.35]  # annual funding on full notional
FEE = 0.001
LIQ_FLOOR = 0.05


def simulate(close, sig, leverage, funding):
    rets = np.diff(close) / close[:-1]
    pos = sig[1:] * leverage
    funding_drag = pos * (funding / 365.0)          # perp funding on FULL notional when long
    toggle = np.abs(np.diff(np.insert(sig, 0, 0.0)))
    fee_drag = toggle[1:] * leverage * FEE
    daily = pos * rets - funding_drag - fee_drag
    eq = np.insert(np.cumprod(1.0 + daily), 0, 1.0)
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100), bool(eq.min() <= LIQ_FLOOR)


def main() -> int:
    base = load_settings()
    extra = [UniverseItem(symbol="SOL-USDC", base="SOL", quote="USDC", coingecko_id="solana", decimals=6),
             UniverseItem(symbol="BNB-USDC", base="BNB", quote="USDC", coingecko_id="binancecoin", decimals=18)]
    base.universe = list(base.universe) + extra
    data = {}
    for sym in base.symbols:
        df = binance_series(base, sym, interval="1d", total=TOTAL)
        data[sym] = (df["close"].to_numpy(dtype=float), multi_ta_signal(df, THRESHOLD).to_numpy())
    bh = {sym: (data[sym][0][-1] / data[sym][0][0] - 1) * 100 for sym in data}

    for fund in FUNDING_LEVELS:
        print(f"\n=== funding {fund*100:.0f}%/yr on full notional ===")
        print(f"{'asset':11} {'L':>4} {'return':>9} {'B&H':>8} {'maxDD':>6} {'ruin?':>5} {'beats?':>6}")
        for sym, (close, sig) in data.items():
            for L in LEVERAGES:
                ret, dd, ruin = simulate(close, sig, L, fund)
                beat = ret > bh[sym] and not ruin
                print(f"{sym:11} {L:>3.0f}x {ret:+8.0f}% {bh[sym]:+7.0f}% {dd:5.0f}% "
                      f"{'YES' if ruin else 'no':>5} {'YES' if beat else '':>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
