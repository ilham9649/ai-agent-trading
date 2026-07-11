#!/usr/bin/env python3
"""Test CRP rebalance-bands + crash-gate on STOCKS (tokenized-stock proxy = underlying
equity price). Prediction: weaker than crypto — stocks have lower vol + higher correlation,
so the rebalancing premium is smaller. Two baskets: high-beta growth (crypto-like) and
diversified sector ETFs (lower vol). Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

FEE = 0.001
BASKETS = {
    "high-beta growth [TSLA,NVDA,AMD,META,NFLX]": ["TSLA", "NVDA", "AMD", "META", "NFLX"],
    "sector ETFs [XLK,XLE,XLF,XLV,XLI]": ["XLK", "XLE", "XLF", "XLV", "XLI"],
}


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
    for name, tickers in BASKETS.items():
        try:
            closes = pd.DataFrame({t: yf.Ticker(t).history(period="10y", interval="1d", auto_adjust=True)["Close"]
                                  for t in tickers}).dropna()
        except Exception as exc:
            print(f"{name}: fetch failed {exc!r}"); continue
        rel = (closes / closes.shift(1)).dropna().to_numpy()
        N = len(tickers)
        eq_target = np.ones(N) / N
        norm = (closes / closes.iloc[0]).mean(axis=1).to_numpy()
        dd = pd.Series(norm)
        dd = ((dd - dd.cummax()) / dd.cummax()).to_numpy()

        bh = crp_gated(rel, eq_target, 1.0, np.ones(rel.shape[0]))  # B&H basket (no rebalance)
        br, bdd = metrics(bh)
        singles = {tickers[i]: metrics(np.cumprod(rel[:, i])) for i in range(N)}
        best_single = max(singles[t][0] for t in tickers)
        print(f"\n### {name}  ({len(closes)} days)")
        print(f"    B&H basket {br:+.0f}%/{bdd:.0f}DD | singles: " +
              " ".join(f"{t} {singles[t][0]:+.0f}%/{singles[t][1]:.0f}DD" for t in tickers))
        print(f"    {'variant':28} {'return':>9} {'maxDD':>6} {'beats basket?':>13} {'beats best single?':>18}")
        for band in [0.10]:
            eq = crp_gated(rel, eq_target, band, np.ones(rel.shape[0]))
            r, d = metrics(eq)
            print(f"    {'CRP-bands inv':28} {r:+8.0f}% {d:5.1f}% "
                  f"{'YES' if r > br else 'no':>13} {'YES' if r > best_single else 'no':>18}")
        for thr in [0.20, 0.25]:
            cgate = np.where(np.r_[0.0, dd[:-1]] > -thr, 1.0, 0.0)  # prior-bar gate
            eq = crp_gated(rel, eq_target, 0.10, cgate)
            r, d = metrics(eq)
            print(f"    {'CRP + crash-gate %d%%' % (thr*100):28} {r:+8.0f}% {d:5.1f}% "
                  f"{'YES' if r > br else 'no':>13} {'YES' if r > best_single else 'no':>18}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
