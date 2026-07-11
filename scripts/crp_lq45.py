#!/usr/bin/env python3
"""Test CRP-bands + crash-gate (dd20 exit, new-60-day-high re-entry) on LQ45 blue-chips
(Indonesia) vs equal-weight basket B&H and IHSG (^JKSE). Free data via yfinance (.JK).
Paper-only / research.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

LQ45 = ["BBCA.JK", "BBRI.JK", "BBNI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
        "UNVR.JK", "TPIA.JK", "INDF.JK", "ICBP.JK", "KLBF.JK", "ADRO.JK"]
PERIOD = "10y"
FEE = 0.001
BAND = 0.10
EXIT_DD = 0.20


def fetch(tickers):
    cols = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=PERIOD, interval="1d", auto_adjust=True)
            s = df["Close"].dropna()
            if len(s) > 500:
                cols[t] = s
        except Exception as e:
            print(f"  skip {t}: {e!r}")
    return pd.DataFrame(cols).dropna()


def metrics(eq):
    rmax = np.maximum.accumulate(eq)
    return (eq[-1] / eq[0] - 1) * 100, abs(((eq - rmax) / rmax).min() * 100)


def gate_new60(closes):
    norm = (closes / closes[0]).mean(axis=1)
    bkt = pd.Series(norm)
    dd = ((bkt - bkt.cummax()) / bkt.cummax()).to_numpy()
    hi60 = bkt.rolling(60).max().to_numpy()
    T = len(dd)
    prev_dd = np.r_[0.0, dd[:-1]]
    prev_norm = np.r_[norm[0], norm[:-1]]
    prev_hi = np.r_[np.nan, hi60[:-1]]
    gate = np.ones(T)
    flat = False
    for t in range(T):
        if flat:
            if prev_norm[t] >= prev_hi[t] and not np.isnan(prev_hi[t]):
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
    return np.array(curve)


def main() -> int:
    print(f"Fetching LQ45 ({len(LQ45)} tickers, {PERIOD})...")
    closes = fetch(LQ45)
    if closes.shape[1] < 4:
        print("Not enough data."); return 1
    print(f"Got {closes.shape[1]} stocks over {len(closes)} days: {list(closes.columns)}\n")
    N = closes.shape[1]
    rel = closes.to_numpy()
    gate = gate_new60(rel)
    eq = run(rel, gate)
    sr, sdd = metrics(eq)

    bh_basket = np.cumprod(np.insert(((rel[1:] / rel[:-1]) * (np.ones(N) / N)).sum(1)
                                     + (1 - 1.0), 0, 1.0))  # equal-weight B&H basket (no rebal)
    # equal-weight B&H: constant 1/N weights, rebalanced
    bh_eqw = run(rel, np.ones(len(rel)))
    br, bdd = metrics(bh_eqw)

    print(f"{'strategy':34} {'return':>9} {'maxDD':>7}")
    print(f"{'CRP-bands + crash-gate':34} {sr:+8.0f}% {sdd:6.1f}%")
    print(f"{'B&H equal-weight LQ45 basket':34} {br:+8.0f}% {bdd:6.1f}%")

    # IHSG benchmark
    try:
        ih = yf.Ticker("^JKSE").history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
        ih = ih.reindex(closes.index).dropna()
        if len(ih) > 100:
            ir, idd = metrics(ih.to_numpy())
            print(f"{'IHSG (^JKSE) B&H':34} {ir:+8.0f}% {idd:6.1f}%")
            print(f"\nStrategy vs IHSG: {'BEATS' if sr > ir else 'under'} on return; "
                  f"DD {sdd:.0f}% vs {idd:.0f}%.")
        else:
            print("(IHSG data insufficient)")
    except Exception as e:
        print(f"(IHSG fetch failed: {e!r})")
    print(f"\nStrategy vs equal-weight basket: {'BEATS' if sr > br else 'under'} on return; "
          f"DD {sdd:.0f}% vs {bdd:.0f}%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
