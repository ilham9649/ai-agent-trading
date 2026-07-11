#!/usr/bin/env python3
"""Leveraged-trend test: can RISK-CONTROLLED LEVERAGE beat buy-and-hold in ABSOLUTE dollars?

Honest premise: in a secular bull, the only way to exceed 100%-long B&H in dollars is
to hold MORE than 100% (leverage). So we test long Lx when close>SMA(200), flat when it
breaks — the trend filter still dodges crashes, leverage amplifies bull upside. We model
realistic drag: borrow financing on the leveraged slice + taker fees on exposure changes.

Reports ABSOLUTE return vs B&H, plus the max drawdown / liquidation risk (the price).
NOTE: leverage needs a margin venue (Aave/Compound on Base, or perps) — not the spot
CowSwap layer; this is a feasibility/edge test, paper-only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402

TOTAL = 3000
SMA = 200
BORROW_RATE = 0.08   # 8%/yr on the borrowed slice (on-chain USDC/ETH borrow ~4-8%; perp long funding similar)
FEE_BPS = 10         # 0.1% taker
LEVERAGES = [1, 2, 3]


def simulate(close: np.ndarray, leverage: float, trend: bool = True) -> dict:
    """Long `leverage`x (always-on if trend=False; when close>SMA if trend=True), flat otherwise."""
    n = len(close)
    sma = pd.Series(close).rolling(SMA).mean().to_numpy()
    rets = np.diff(close) / close[:-1]
    eq = 1.0
    exposure = 0.0
    curve = [1.0]
    for i in range(1, n):
        sig = (close[i - 1] > sma[i - 1]) if (trend and not np.isnan(sma[i - 1])) else (not trend)
        # warmup before SMA: if trend mode and no SMA yet, stay flat
        target = leverage if (sig if trend else True) else 0.0
        if not trend:
            target = leverage  # always-on leveraged B&H
        delta = abs(target - exposure)
        if delta > 1e-9:
            eq *= (1 - delta * FEE_BPS / 10000.0)
        exposure = target
        borrowed = max(0.0, exposure - 1.0)
        eq *= (1 + exposure * rets[i - 1] - borrowed * BORROW_RATE / 365.0)
        if eq <= 0:
            return dict(final=0.0, ret=-100.0, cagr=-100.0, sharpe=0.0, dd=100.0, liquidated=True)
        curve.append(eq)
    curve = np.array(curve)
    r = curve[1:] / curve[:-1] - 1
    years = (n - 1) / 365
    ret = (curve[-1] - 1) * 100
    cagr = (curve[-1] ** (1 / years) - 1) * 100 if years > 0 else 0
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(365) if r.std(ddof=1) > 0 else 0
    rmax = np.maximum.accumulate(curve)
    dd = abs(((curve - rmax) / rmax).min() * 100)
    return dict(final=curve[-1], ret=ret, cagr=cagr, sharpe=sharpe, dd=dd, liquidated=False)


def main() -> int:
    base = load_settings()
    syms = base.symbols
    series = {sym: binance_series(base, sym, interval="1d", total=TOTAL) for sym in syms}
    print(f"Leveraged-trend test (~{TOTAL} daily bars), borrow {BORROW_RATE*100:.0f}%/yr, fee {FEE_BPS}bps\n")

    for sym in syms:
        close = series[sym]["close"].to_numpy(dtype=float)
        bh = (close[-1] / close[0] - 1) * 100
        print(f"### {sym}  —  buy-and-hold: +{bh:.0f}%")
        print(f"{'mode':22} {'return':>10} {'CAGR':>7} {'sharpe':>7} {'maxDD':>7} {'liq?':>5} {'beats B&H?':>10}")
        for L in LEVERAGES:
            res = simulate(close, float(L), trend=True)
            tag = f"trend {L}x"
            print(f"{tag:22} {res['ret']:+9.0f}% {res['cagr']:+6.1f}% {res['sharpe']:7.2f} "
                  f"{res['dd']:6.1f}% {'YES' if res['liquidated'] else 'no':>5} {'<<< BEATS' if res['ret'] > bh else '':>10}")
        # always-on leveraged B&H (pure leverage, no timing) for reference
        for L in LEVERAGES:
            res = simulate(close, float(L), trend=False)
            tag = f"always {L}x (no timing)"
            print(f"{tag:22} {res['ret']:+9.0f}% {res['cagr']:+6.1f}% {res['sharpe']:7.2f} "
                  f"{res['dd']:6.1f}% {'YES' if res['liquidated'] else 'no':>5}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
