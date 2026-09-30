"""crash brake on market AND basket (goal: cagr >= 3x index AND max dd no worse than index, in 1927-62, 1963-2007, 2008-26).
fixed rule (same 7% / 20-day brake as r4_lever4.py, now also on the held basket's own price): extra leverage (k > 1) only while
BOTH the market and the R4 basket closed within 7% of their 20-day highs at the prior close; otherwise plain R4 weights."""
import numpy as np, pandas as pd
from engine import *
import s2_rates2 as r
from strat_bc import rule_c

w4 = rule_c(r.ens * (1 - 0.5 * r.hike.shift(-1).fillna(0)), r.idx)
bpx = (1 + r.b3).cumprod()
calm = ((r.d.px >= 0.93 * r.d.px.rolling(20).max()) & (bpx >= 0.93 * bpx.rolling(20).max())).astype(float).shift(1).fillna(0)
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
print(f"brake on {1 - calm.mean():.0%} of days")
for spread in [0.01, 0.005]:
    rows = []
    for k in np.arange(1.5, 5.01, 0.25):
        wk = pd.Series(w4.values * np.where(calm == 1, k, 1.0), index=r.idx)
        rk = run(wk, r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
        st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
        okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
        margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
        rows.append((margin, k, st, okk))
    for mg, k, st, okk in sorted(rows, key=lambda t: -t[0])[:3]:
        print(f"spread {spread:.1%} k {k:.2f}: {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} dd {st[e]['maxdd']:.1%}" for e in P))
