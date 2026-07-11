#!/usr/bin/env python3
"""Re-entry rule test — fix the 'wait forever for -5% of peak' problem.

Exit at dd>20%; compare RE-ENTRY rules (stateful):
  dd5     : re-enter at dd>-5%   (too tight — waits for near-peak after a deep crash)
  dd25    : re-enter at dd>-25%
  new60hi : re-enter on a new 60-day high (recovery confirmed, fires DURING recovery)
  sma50   : re-enter when basket > 50-day SMA (short-term uptrend)
Reports return / DD / %time-invested (full + OOS). Paper-only.
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
EXIT_DD = 0.20


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def build_gate(closes, reentry):
    norm = (closes / closes[0]).mean(axis=1)
    bkt = pd.Series(norm)
    dd = ((bkt - bkt.cummax()) / bkt.cummax()).to_numpy()
    sma50 = bkt.rolling(50).mean().to_numpy()
    hi60 = bkt.rolling(60).max().to_numpy()
    T = len(dd)
    prev_dd = np.r_[0.0, dd[:-1]]
    prev_sma = np.r_[np.nan, sma50[:-1]]
    prev_hi = np.r_[np.nan, hi60[:-1]]
    prev_norm = np.r_[norm[0], norm[:-1]]
    gate = np.ones(T)
    flat = False
    for t in range(T):
        if flat:
            cond = (prev_dd[t] > -0.05) if reentry == "dd5" else \
                   (prev_dd[t] > -0.25) if reentry == "dd25" else \
                   (prev_norm[t] >= prev_hi[t]) if reentry == "new60hi" else \
                   (prev_norm[t] > prev_sma[t]) if reentry == "sma50" else True
            if cond and not np.isnan(prev_sma[t]):
                flat = False
        else:
            if prev_dd[t] < -EXIT_DD:
                flat = True
        gate[t] = 0.0 if flat else 1.0
    return gate


def run(closes, gate):
    N = closes.shape[1]
    tgt = np.ones(N) / N
    rel = closes[1:] / closes[:-1]
    g = gate[1:]
    w = tgt.copy(); invested = bool(g[0]); wealth = 1.0; curve = [1.0]
    for t in range(rel.shape[0]):
        want = bool(g[t])
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
    return metrics(eq), float(np.mean(g))


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
    print(f"Re-entry rules (exit at dd>{int(EXIT_DD*100)}%), 10-asset basket\n")
    print(f"{'reentry':9} | {'FULL ret/DD':>14} {'%inv':>5} | {'OOS ret/DD':>14} {'%inv':>5}")
    for reentry in ["dd5", "dd25", "new60hi", "sma50"]:
        gf = build_gate(closes, reentry)
        (fr, fdd), finv = run(closes, gf)
        go = build_gate(closes[cut:], reentry)
        (orr, odd), oinv = run(closes[cut:], go)
        print(f"{reentry:9} | {fr:+7.0f}%/{fdd:4.1f}DD {finv*100:4.0f}% | {orr:+7.0f}%/{odd:4.1f}DD {oinv*100:4.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
