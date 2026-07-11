#!/usr/bin/env python3
"""GENERALIZATION test: CRP rebalance-bands + crash-only gate on a DIFFERENT basket
(XRP/ADA/AVAX/DOGE/LINK) — none of the original BTC/ETH/SOL/BNB. Does the rebalancing-
premium + crash-gate edge beat these alts' buy-and-hold with lower DD too? Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402

ASSETS = ["XRP-USDC", "ADA-USDC", "AVAX-USDC", "DOGE-USDC", "LINK-USDC"]
TOTAL = 3000
FEE = 0.001


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def crp_gated(rets, target, band, gate):
    w = target.copy()
    invested = bool(gate[0])
    wealth = 1.0
    curve = [1.0]
    for t in range(rets.shape[0]):
        want = bool(gate[t])
        if want and not invested:
            wealth *= (1 - FEE); w = target * wealth; invested = True
        if (not want) and invested:
            wealth *= (1 - FEE); invested = False
        if invested:
            w = w * rets[t]
            wv = w.sum(); w_now = w / wv
            if np.abs(w_now - target).max() > band:
                to = np.abs(w_now - target).sum(); wv *= (1 - to * FEE); w = target * wv
            wealth = wv
        curve.append(wealth)
    return np.array(curve)


def main() -> int:
    base = load_settings()
    base.universe = list(base.universe) + [
        UniverseItem(symbol=s, base=s.split("-")[0], quote="USDC", coingecko_id=s.lower(), decimals=6)
        for s in ASSETS]
    series = {s: binance_series(base, s, interval="1d", total=TOTAL) for s in ASSETS}
    minlen = min(len(series[s]) for s in ASSETS)
    for s in ASSETS:
        series[s] = series[s].tail(minlen).reset_index(drop=True)
    N = len(ASSETS)
    rel = np.column_stack([series[s]["close"].to_numpy(float)[1:] / series[s]["close"].to_numpy(float)[:-1]
                           for s in ASSETS])
    eq_target = np.ones(N) / N
    norm = np.column_stack([series[s]["close"].to_numpy(float) / series[s]["close"].to_numpy(float)[0]
                            for s in ASSETS]).mean(axis=1)
    basket = pd.Series(norm)
    dd = (basket - basket.cummax()) / basket.cummax()

    bh_basket = crp_gated(rel, eq_target, 1.0, np.ones(rel.shape[0]))  # never rebalance => B&H basket
    br, bdd = metrics(bh_basket)
    singles = {ASSETS[i]: metrics(np.cumprod(rel[:, i])) for i in range(N)}
    print(f"CRP-bands on a DIFFERENT basket {ASSETS} over {minlen} bars.")
    print(f"B&H basket {br:+.0f}%/{bdd:.0f}DD; singles: " +
          " ".join(f"{s} {singles[s][0]:+.0f}%/{singles[s][1]:.0f}DD" for s in ASSETS) + "\n")
    print(f"{'variant':30} {'return':>9} {'maxDD':>6} {'beats basket?':>13} {'beats best single?':>18}")
    # always-invested CRP (rebalancing premium)
    for band in [0.05, 0.10]:
        eq = crp_gated(rel, eq_target, band, np.ones(rel.shape[0]))
        r, d = metrics(eq)
        best_single = max(singles[s][0] for s in ASSETS)
        print(f"{'CRP-bands inv (band %d%%)' % (band*100):30} {r:+8.0f}% {d:5.1f}% "
              f"{'YES' if r > br else 'no':>13} {'YES' if r > best_single else 'no':>18}")
    # CRP + crash-only gate
    for thr in [0.20, 0.25]:
        cgate = (dd.shift(1) > -thr).astype(float).fillna(0.0).to_numpy()[:rel.shape[0]]
        eq = crp_gated(rel, eq_target, 0.10, cgate)
        r, d = metrics(eq)
        best_single = max(singles[s][0] for s in ASSETS)
        print(f"{'CRP + crash-gate %d%%' % (thr*100):30} {r:+8.0f}% {d:5.1f}% "
              f"{'YES' if r > br else 'no':>13} {'YES' if r > best_single else 'no':>18}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
