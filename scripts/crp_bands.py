#!/usr/bin/env python3
"""CRP with rebalance bands (the rebalancing premium) — arxiv's most-robust OLPS choice.

Hold fixed target weights across BTC/ETH/SOL/BNB; rebalance back ONLY when a weight
drifts past +/-band. 'Buy low, sell high' across high-vol/low-corr assets harvests the
rebalancing premium. Tests if it beats buy-and-hold (basket + singles) at lower DD.
Unlevered, long-only. Paper-only.
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
BANDS = [0.03, 0.05, 0.10]


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def crp(rets, target, band):
    w = target.copy()
    curve = [1.0]
    for t in range(rets.shape[0]):
        w = w * rets[t]
        wealth = w.sum()
        w_now = w / wealth
        if np.abs(w_now - target).max() > band:
            turnover = np.abs(w_now - target).sum()
            wealth *= (1 - turnover * FEE)
            w = target * wealth
        curve.append(wealth)
    return np.array(curve)


def crp_gated(rets, target, band, gate):
    """CRP-bands but flat (cash) when the prior-bar trend gate is off -> crash avoidance."""
    w = target.copy()
    invested = bool(gate[0])
    wealth = 1.0
    curve = [1.0]
    for t in range(rets.shape[0]):
        want = bool(gate[t])
        if want and not invested:
            wealth *= (1 - FEE)
            w = target * wealth
            invested = True
        if (not want) and invested:
            wealth *= (1 - FEE)
            invested = False
        if invested:
            w = w * rets[t]
            wv = w.sum()
            w_now = w / wv
            if np.abs(w_now - target).max() > band:
                turnover = np.abs(w_now - target).sum()
                wv *= (1 - turnover * FEE)
                w = target * wv
            wealth = wv
        curve.append(wealth)
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
    N = len(ASSETS)
    rel = np.column_stack([series[s]["close"].to_numpy(float)[1:] / series[s]["close"].to_numpy(float)[:-1]
                           for s in ASSETS])
    eq_target = np.ones(N) / N
    vol = np.array([pd.Series(series[s]["close"]).pct_change().rolling(90).std().to_numpy()[-1] for s in ASSETS])
    ivol_target = (1 / vol) / (1 / vol).sum()

    bh_basket = crp(rel, eq_target, band=1.0)  # band=1.0 => never rebalances => buy-and-hold basket
    basket_ret, basket_dd = metrics(bh_basket)
    singles = {ASSETS[i]: metrics(np.cumprod(rel[:, i])) for i in range(N)}
    print(f"CRP rebalance-bands over {minlen} bars. B&H basket {basket_ret:+.0f}%/{basket_dd:.0f}DD; "
          f"BTC {singles['cbBTC-USDC'][0]:+.0f}%, SOL {singles['SOL-USDC'][0]:+.0f}%, BNB {singles['BNB-USDC'][0]:+.0f}%\n")
    print(f"{'target':14} {'band':>5} {'return':>9} {'maxDD':>6} {'beats basket?':>13} {'beats BTC?':>10}")
    for tgt_name, tgt in [("equal-weight", eq_target), ("inverse-vol", ivol_target)]:
        for band in BANDS:
            eq = crp(rel, tgt, band)
            r, dd = metrics(eq)
            print(f"{tgt_name:14} {band*100:>4.0f}% {r:+8.0f}% {dd:5.1f}% "
                  f"{'YES' if r > basket_ret else 'no':>13} {'YES' if r > singles['cbBTC-USDC'][0] else 'no':>10}")

    # CRP-bands + trend gate (rebalancing premium + crash avoidance)
    norm = np.column_stack([series[s]["close"].to_numpy(float) / series[s]["close"].to_numpy(float)[0]
                            for s in ASSETS]).mean(axis=1)
    gate = (pd.Series(norm) > pd.Series(norm).rolling(200).mean()).astype(float).shift(1).fillna(0.0).to_numpy()
    gate = gate[:rel.shape[0]]
    print("\n--- CRP-bands + trend gate (rebalancing premium + crash avoidance) ---")
    print(f"{'target':14} {'band':>5} {'return':>9} {'maxDD':>6} {'beats basket?':>13} {'beats BTC?':>10}")
    for tgt_name, tgt in [("equal-weight", eq_target), ("inverse-vol", ivol_target)]:
        for band in [0.05, 0.10]:
            eq = crp_gated(rel, tgt, band, gate)
            r, dd = metrics(eq)
            print(f"{tgt_name:14} {band*100:>4.0f}% {r:+8.0f}% {dd:5.1f}% "
                  f"{'YES' if r > basket_ret else 'no':>13} {'YES' if r > singles['cbBTC-USDC'][0] else 'no':>10}")

    # CRP-bands + CRASH-ONLY gate (flat only in severe drawdowns, not normal pullbacks)
    basket = pd.Series(norm)
    dd = (basket - basket.cummax()) / basket.cummax()  # <=0
    print("\n--- CRP-bands + CRASH-ONLY gate (equal-weight, band 10%) ---")
    print(f"{'crash thresh':>12} {'return':>9} {'maxDD':>6} {'beats basket?':>13} {'beats BTC?':>10}")
    for thr in [0.20, 0.25, 0.30]:
        cgate = (dd.shift(1) > -thr).astype(float).fillna(0.0).to_numpy()[:rel.shape[0]]
        eq = crp_gated(rel, eq_target, 0.10, cgate)
        r, ddv = metrics(eq)
        print(f"{thr*100:>11.0f}% {r:+8.0f}% {ddv:5.1f}% "
              f"{'YES' if r > basket_ret else 'no':>13} {'YES' if r > singles['cbBTC-USDC'][0] else 'no':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
