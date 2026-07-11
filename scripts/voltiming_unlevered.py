#!/usr/bin/env python3
"""UNLEVERED vol-scaled trend (cap 1x): exposure = clip(target_vol/realized_vol, 0, 1) * signal.
Stays fully invested in calm trends, scales DOWN (never up past 1x) in high-vol regimes,
flat when trend breaks. Targets low DD while keeping more upside than the flat 1/N portfolio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402
from trading_agent.strategy.multi_ta_signal import multi_ta_signal  # noqa: E402

ASSETS = ["WETH-USDC", "cbBTC-USDC", "SOL-USDC", "BNB-USDC"]
TOTAL = 3000
FEE = 0.001
TARGETS = [0.30, 0.50, 0.80]


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def run(w, rets):
    pr = (w * rets).sum(axis=1)
    to = np.abs(np.diff(w, axis=0, prepend=w[:1])).sum(axis=1)
    return np.insert(np.cumprod(1 + pr - to * FEE), 0, 1.0)


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
    sig = np.column_stack([multi_ta_signal(series[s], 4).to_numpy()[:T] for s in ASSETS])
    rets = np.column_stack([np.diff(series[s]["close"].to_numpy(float)) / series[s]["close"].to_numpy(float)[:-1]
                            for s in ASSETS])
    # annualized realized vol per asset per bar (prior bar, no lookahead)
    ann_vol = np.column_stack([
        (pd.Series(series[s]["close"]).pct_change().rolling(20).std().to_numpy()[:T] * np.sqrt(365))
        for s in ASSETS])

    btc = run(np.eye(N)[1][None, :] * np.ones((T, 1)), rets)
    btc_ret, btc_dd = metrics(btc)
    eth = run(np.eye(N)[0][None, :] * np.ones((T, 1)), rets)
    eth_ret, eth_dd = metrics(eth)
    print(f"Unlevered vol-scaled trend portfolio (cap 1x). B&H: ETH {eth_ret:+.0f}%/{eth_dd:.0f}%, "
          f"BTC {btc_ret:+.0f}%/{btc_dd:.0f}%\n")
    print(f"{'target_vol':>10} {'alloc':>10} {'return':>9} {'maxDD':>6} {'beats ETH?':>10} {'beats BTC?':>10}")
    for tv in TARGETS:
        for mode, alloc in [("1/N+volscale", None), ("equalactive+volscale", "active")]:
            wvol = np.clip(tv / np.maximum(ann_vol, 1e-9), 0, 1) * sig
            if mode.startswith("1/N"):
                w = wvol / N
            else:
                rs = wvol.sum(axis=1, keepdims=True)
                w = np.where(rs > 0, wvol / np.maximum(rs, 1), 0)
            eq = run(w, rets)
            r, dd = metrics(eq)
            print(f"{tv*100:>9.0f}% {mode:>10} {r:+8.0f}% {dd:5.1f}% "
                  f"{'YES' if r > eth_ret else 'no':>10} {'YES' if r > btc_ret else 'no':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
