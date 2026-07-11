#!/usr/bin/env python3
"""IMPROVE the CRP strategy: higher return AND lower drawdown.

Levers (mechanism-based, not grid-fit):
  1) larger diversified basket (10 alts) -> lower idiosyncratic DD
  2) inverse-vol risk-parity weighting -> tilt to calmer assets -> lower DD
  3) smarter crash gate: a VOLATILITY-SPIKE detector (realized vol > rolling p90) exits
     crashes EARLIER than waiting for a 20% drawdown -> less damage + fewer whipsaws.
Baseline to beat: CRP + crash-gate(20%) = +890% / 40.6% DD (4-asset). Paper-only.
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


def crp_gated(rets, target, gate):
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
            if np.abs(w_now - target).max() > BAND:
                to = np.abs(w_now - target).sum(); wv *= (1 - to * FEE); w = target * wv
            wealth = wv
        curve.append(wealth)
    return np.array(curve)


def main() -> int:
    base = load_settings()
    base.universe = list(base.universe) + [
        UniverseItem(symbol=s, base=s.split("-")[0], quote="USDC", coingecko_id=s.lower(), decimals=6)
        for s in ASSETS if s not in base.symbols]
    series = {s: binance_series(base, s, interval="1d", total=TOTAL) for s in ASSETS}
    minlen = min(len(series[s]) for s in ASSETS)
    for s in ASSETS:
        series[s] = series[s].tail(minlen).reset_index(drop=True)
    N = len(ASSETS)
    closes = np.column_stack([series[s]["close"].to_numpy(float) for s in ASSETS])
    rel = closes[1:] / closes[:-1]
    T = rel.shape[0]

    eq_w = np.ones(N) / N
    vol0 = np.std(np.diff(np.log(closes[:90]), axis=0), axis=0)  # first-90d vol, no lookahead
    ivol_w = (1 / vol0) / (1 / vol0).sum()

    norm = (closes / closes[0]).mean(axis=1)
    bkt = pd.Series(norm)
    dd = ((bkt - bkt.cummax()) / bkt.cummax()).to_numpy()
    realvol = bkt.pct_change().rolling(10).std().to_numpy()
    rp90 = pd.Series(realvol).rolling(252, min_periods=60).quantile(0.90).to_numpy()

    # gates (gate[t] uses info through t-1)
    ones = np.ones(T)
    g_dd20 = np.where(np.r_[0.0, dd[:-1]] > -0.20, 1.0, 0.0)
    g_vol90 = np.where(np.r_[np.nan, realvol[:-1]] <= np.r_[np.nan, rp90[:-1]], 1.0, 0.0)
    g_vol90[0] = 1.0
    g_comb = np.where((np.r_[0.0, dd[:-1]] > -0.15) | (np.r_[np.nan, realvol[:-1]] <= np.r_[np.nan, rp90[:-1]]),
                      1.0, 0.0)
    g_comb[0] = 1.0

    bh = crp_gated(rel, eq_w, ones)
    br, bdd = metrics(bh)
    print(f"10-asset basket over {minlen} bars. B&H basket {br:+.0f}%/{bdd:.0f}DD. "
          f"Baseline to beat: +890% / 40.6%DD (4-asset crash-gate).\n")
    print(f"{'weighting':12} {'gate':10} {'return':>9} {'maxDD':>6} {'Calmar':>7} {'beats baseline?':>15}")
    best = None
    for wname, tgt in [("equal", eq_w), ("inv-vol", ivol_w)]:
        for gname, g in [("none", ones), ("dd20", g_dd20), ("vol90", g_vol90), ("comb", g_comb)]:
            eq = crp_gated(rel, tgt, g)
            r, d = metrics(eq)
            cal = r / d if d > 0 else 0
            beat = (r > 890 and d < 40.6)
            star = "  <<<" if beat else ""
            print(f"{wname:12} {gname:10} {r:+8.0f}% {d:5.1f}% {cal:7.2f} {'YES' if beat else '':>15}{star}")
            if beat and (best is None or cal > best[2]):
                best = (wname, gname, cal, r, d)
    if best:
        print(f"\nBest improvement: {best[0]}+{best[1]} -> {best[3]:+.0f}% / {best[4]:.1f}%DD (Calmar {best[2]:.2f})")
    else:
        print("\nNo config beat baseline on BOTH return and DD (frontier).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
