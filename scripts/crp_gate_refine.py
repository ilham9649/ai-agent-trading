#!/usr/bin/env python3
"""Refine the crash gate to cut the recent-regime (OOS) whipsaw drawdown.

Variants:
  dd20        : flat when drawdown>20% (re-enters at <20% -> whipsaws on crash bounces)
  hysteresis  : exit at dd>20%, re-enter only at dd<5% (no bounce re-entries)
  dd20+sma    : exit when dd>20% AND basket<200-SMA (confirmed bear only)
Tests full + OOS return/DD. Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402

ASSETS = ["cbBTC-USDC", "WETH-USDC", "SOL-USDC", "BNB-USDC", "XRP-USDC",
          "ADA-USDC", "AVAX-USDC", "DOGE-USDC", "LINK-USDC", "DOT-USDC"]
TOTAL = 3000
FEE = 0.001
BAND = 0.10


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def gates(closes):
    norm = (closes / closes[0]).mean(axis=1)
    bkt = pd.Series(norm)
    dd = ((bkt - bkt.cummax()) / bkt.cummax()).to_numpy()
    sma = bkt.rolling(200).mean().to_numpy()
    T = len(dd)
    prev_dd = np.r_[0.0, dd[:-1]]
    g_dd20 = np.where(prev_dd > -0.20, 1.0, 0.0)
    below = np.where(np.isnan(sma), False, norm < sma)
    prev_below = np.r_[False, below[:-1]]
    crash = (prev_dd <= -0.20) & prev_below
    g_ddsm = np.where(crash, 0.0, 1.0)
    # hysteresis (stateful)
    g_hyst = np.ones(T)
    flat = False
    for t in range(T):
        pd_ = prev_dd[t]
        if flat:
            if pd_ > -0.05:
                flat = False
        else:
            if pd_ < -0.20:
                flat = True
        g_hyst[t] = 0.0 if flat else 1.0
    return {"dd20": g_dd20, "hysteresis": g_hyst, "dd20+sma": g_ddsm}


def run(closes, gate):
    N = closes.shape[1]
    tgt = np.ones(N) / N
    rel = closes[1:] / closes[:-1]
    gate = gate[1:] if len(gate) == closes.shape[0] else gate
    w = tgt.copy(); invested = bool(gate[0]); wealth = 1.0; curve = [1.0]
    for t in range(rel.shape[0]):
        want = bool(gate[t])
        if want and not invested:
            wealth *= (1 - FEE); w = tgt * wealth; invested = True
        if (not want) and invested:
            wealth *= (1 - FEE); invested = False
        if invested:
            w = w * rel[t]; wv = w.sum(); wn = w / wv
            if np.abs(wn - tgt).max() > BAND:
                to = np.abs(wn - tgt).sum(); wv *= (1 - to * FEE); w = tgt * wv
            wealth = wv
        curve.append(wealth)
    return metrics(np.array(curve))


def main() -> int:
    base = load_settings()
    base.universe = list(base.universe) + [
        UniverseItem(symbol=s, base=s.split("-")[0], quote="USDC", coingecko_id=s.lower(), decimals=6)
        for s in ASSETS if s not in base.symbols]
    series = {s: binance_series(base, s, interval="1d", total=TOTAL) for s in ASSETS}
    minlen = min(len(series[s]) for s in ASSETS)
    for s in ASSETS:
        series[s] = series[s].tail(minlen).reset_index(drop=True)
    closes = np.column_stack([series[s]["close"].to_numpy(float) for s in ASSETS])
    cut = minlen // 2

    print("Crash-gate refinement (10-asset basket): full / OOS return & DD\n")
    print(f"{'gate':12} {'FULL':>16} {'OOS':>16}")
    for gname in ["dd20", "hysteresis", "dd20+sma"]:
        gf = gates(closes)[gname]
        fr, fdd = run(closes, gf)
        go = gates(closes[cut:])[gname]
        orr, odd = run(closes[cut:], go)
        print(f"{gname:12} {fr:+7.0f}%/{fdd:4.1f}DD {orr:+7.0f}%/{odd:4.1f}DD")
    # B&H basket reference
    N = closes.shape[1]; tgt = np.ones(N) / N
    bh_full = metrics(np.cumprod(np.insert(((closes[1:]/closes[:-1])*tgt).sum(1)+(1-tgt.sum()), 0, 1.0)))
    bh_oos = metrics(np.cumprod(np.insert(((closes[cut+1:]/closes[cut:-1])*tgt).sum(1)+(1-tgt.sum()), 0, 1.0)))
    print(f"{'B&H basket':12} {bh_full[0]:+7.0f}%/{bh_full[1]:4.1f}DD {bh_oos[0]:+7.0f}%/{bh_oos[1]:4.1f}DD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
