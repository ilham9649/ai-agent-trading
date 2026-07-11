#!/usr/bin/env python3
"""Online Portfolio Selection — Exponential Gradient (Helmbold et al. 1998) + trend gate.

EG rebalances toward recent winners (w *= exp(eta*r/(w·r)), normalized) — long-only,
unlevered. 'EG + trend gate' multiplies weights by a basket trend gate (flat in market
downtrends) to cut drawdown. arxiv-grounded OLPS baseline. Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402

ASSETS = ["WETH-USDC", "cbBTC-USDC", "SOL-USDC", "BNB-USDC"]
TOTAL = 3000
FEE = 0.001
ETAS = [0.05, 0.2, 0.5]


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def eg_backtest(rel, gate=None, eta=0.2):
    T, N = rel.shape
    w = np.ones(N) / N
    prev_w_eff = w.copy()
    eq = 1.0
    curve = [1.0]
    for t in range(T):
        g = 1.0 if gate is None else float(gate[t])
        w_eff = w * g
        if w_eff.sum() > 0:
            w_eff = w_eff / w_eff.sum()
        turnover = np.abs(w_eff - prev_w_eff).sum()
        wealth = (w_eff @ rel[t]) if w_eff.sum() > 0 else 1.0  # flat = cash (no loss)
        eq *= wealth * (1 - turnover * FEE)
        curve.append(eq)
        prev_w_eff = w_eff
        denom = max(w @ rel[t], 1e-12)
        w = w * np.exp(eta * rel[t] / denom)
        w = w / w.sum()
    return np.array(curve)


def main() -> int:
    base = load_settings()
    base.universe = list(base.universe) + [
        UniverseItem(symbol="SOL-USDC", base="SOL", quote="USDC", coingecko_id="solana", decimals=6),
        UniverseItem(symbol="BNB-USDC", base="BNB", quote="USDC", coingecko_id="binancecoin", decimals=18)]
    series = {s: binance_series(base, s, interval="1d", total=TOTAL) for s in ASSETS}
    minlen = min(len(series[s]) for s in ASSETS)
    for s in ASSETS:
        series[s] = series[s].tail(minlen).reset_index(drop=True)
    T, N = minlen - 1, len(ASSETS)
    rel = np.column_stack([series[s]["close"].to_numpy(float)[1:] / series[s]["close"].to_numpy(float)[:-1]
                           for s in ASSETS])
    norm = np.column_stack([series[s]["close"].to_numpy(float) / series[s]["close"].to_numpy(float)[0]
                            for s in ASSETS]).mean(axis=1)
    gate_full = (pd.Series(norm) > pd.Series(norm).rolling(200).mean()).astype(float)
    gate = gate_full.shift(1).fillna(0.0).to_numpy()[1:]  # (T,) prior-bar gate

    bh = [metrics(np.insert(np.cumprod(rel[:, i]), 0, 1.0)) for i in range(N)]
    print(f"OLPS Exponential Gradient over {minlen} bars. B&H: ETH {bh[0][0]:+.0f}%/{bh[0][1]:.0f}% "
          f"BTC {bh[1][0]:+.0f}%/{bh[1][1]:.0f}% SOL {bh[2][0]:+.0f}% BNB {bh[3][0]:+.0f}%\n")
    print(f"{'method':22} {'eta':>5} {'return':>9} {'maxDD':>6} {'beats BTC?':>10}")
    for eta in ETAS:
        r, dd = metrics(eg_backtest(rel, eta=eta))
        print(f"{'EG (always invested)':22} {eta:>5.2f} {r:+8.0f}% {dd:5.1f}% {'YES' if r > bh[1][0] else 'no':>10}")
    for eta in ETAS:
        r, dd = metrics(eg_backtest(rel, gate=gate, eta=eta))
        print(f"{'EG + trend gate':22} {eta:>5.2f} {r:+8.0f}% {dd:5.1f}% {'YES' if r > bh[1][0] else 'no':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
