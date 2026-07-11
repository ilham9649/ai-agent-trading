#!/usr/bin/env python3
"""OOS-validate the improved strategy (10-asset basket, equal-weight, dd20 crash gate):
split history in half, confirm it beats B&H basket out-of-sample with low DD. Paper-only.
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
THR = 0.20


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def run(closes, thr):
    N = closes.shape[1]
    tgt = np.ones(N) / N
    rel = closes[1:] / closes[:-1]
    T = rel.shape[0]
    norm = (closes / closes[0]).mean(axis=1)
    bkt = pd.Series(norm)
    dd = ((bkt - bkt.cummax()) / bkt.cummax()).to_numpy()
    gate = np.where(np.r_[0.0, dd[:-1]] > -thr, 1.0, 0.0)
    w = tgt.copy(); invested = bool(gate[0]); wealth = 1.0; curve = [1.0]
    for t in range(T):
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
    eq = np.array(curve)
    bh = np.cumprod(np.insert((rel * tgt).sum(axis=1) + (1 - tgt.sum()), 0, 1.0))
    return metrics(eq), metrics(bh)


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

    print("Gate-threshold sweep (10-asset, equal-weight): full / OOS return & DD vs B&H basket\n")
    print(f"{'thr':>5} {'FULL strat':>16} {'OOS strat':>16} {'OOS B&H':>14}")
    for thr in [0.20, 0.25, 0.30, 0.35]:
        (fr, fdd), _ = run(closes, thr)
        (orr, odd), (obr, obdd) = run(closes[cut:], thr)
        print(f"{int(thr*100):>4}% {fr:+7.0f}%/{fdd:4.1f}DD {orr:+7.0f}%/{odd:4.1f}DD   {obr:+6.0f}%/{obdd:4.1f}DD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
