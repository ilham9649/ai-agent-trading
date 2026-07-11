#!/usr/bin/env python3
"""Vol-targeted LEVERAGED multi-TA: exposure = clip(target_vol/realized_vol, 1, L_max)
when the multi-TA signal is bullish, 0 when flat. Auto-deleverages in high-vol (crash-
prone) regimes -> aims to beat B&H on ALL majors AND survive (vs ruin-risky fixed 2x).
Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402
from trading_agent.strategy.multi_ta_signal import multi_ta_signal  # noqa: E402

TOTAL = 3000
THRESHOLD = 4
TARGETS = [(60, 2.0), (80, 2.5), (100, 3.0)]  # (target_annual_vol%, L_max)
BORROW = 0.08
FEE = 0.001
LIQ_FLOOR = 0.05


def simulate(close, sig, target_vol, l_max):
    rets = np.diff(close) / close[:-1]
    realvol = pd.Series(close).pct_change().rolling(20).std().to_numpy()
    ann_vol = realvol * np.sqrt(365)
    # weight per bar from PRIOR bar's vol (no lookahead)
    weight = np.clip(target_vol / np.roll(ann_vol, 1), 1.0, l_max)
    weight[0] = 1.0
    pos = sig[1:] * weight[1:]  # prior-bar signal x prior-bar-vol weight
    borrow_drag = np.clip(pos - 1.0, 0.0, None) * (BORROW / 365.0)
    toggle = np.abs(np.diff(np.insert(sig, 0, 0.0)))
    fee_drag = toggle[1:] * FEE  # fee on the unleveraged toggle (conservative)
    daily = pos * rets - borrow_drag - fee_drag
    eq = np.insert(np.cumprod(1.0 + daily), 0, 1.0)
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100), bool(eq.min() <= LIQ_FLOOR)


def main() -> int:
    base = load_settings()
    extra = [UniverseItem(symbol="SOL-USDC", base="SOL", quote="USDC", coingecko_id="solana", decimals=6),
             UniverseItem(symbol="BNB-USDC", base="BNB", quote="USDC", coingecko_id="binancecoin", decimals=18)]
    base.universe = list(base.universe) + extra
    print(f"Vol-targeted leveraged multi-TA (t{THRESHOLD})\n")
    for tv, lmax in TARGETS:
        print(f"=== target_vol={tv}%  L_max={lmax}x ===")
        print(f"{'asset':11} {'return':>9} {'B&H':>8} {'maxDD':>6} {'ruin?':>5} {'beats?':>6}")
        all_beat = True
        for sym in base.symbols:
            df = binance_series(base, sym, interval="1d", total=TOTAL)
            close = df["close"].to_numpy(dtype=float)
            sig = multi_ta_signal(df, THRESHOLD).to_numpy()
            ret, dd, ruin = simulate(close, sig, tv / 100.0, lmax)
            bh = (close[-1] / close[0] - 1) * 100
            beat = ret > bh and not ruin
            all_beat = all_beat and beat
            print(f"{sym:11} {ret:+8.0f}% {bh:+7.0f}% {dd:5.0f}% {'YES' if ruin else 'no':>5} {'YES' if beat else 'no':>6}")
        print(f"  -> beats B&H on ALL majors without ruin: {'YES' if all_beat else 'no'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
