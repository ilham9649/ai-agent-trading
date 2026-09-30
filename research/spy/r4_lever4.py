"""crash brake on extra leverage (goal: cagr >= 3x index AND max dd no worse than index, in 1927-62, 1963-2007, 2008-26).
fixed rule (no tuning): extra leverage (k > 1) only while the market's prior close is within 7% of its highest close of the last 20 days;
otherwise plain R4 weights. scan one k for all periods; borrow spread 1.0% and 0.5%."""
import numpy as np, pandas as pd
from engine import *
import s2_rates2 as r
from strat_bc import rule_c

w4 = rule_c(r.ens * (1 - 0.5 * r.hike.shift(-1).fillna(0)), r.idx)
calm = (r.d.px >= 0.93 * r.d.px.rolling(20).max()).astype(float).shift(1).fillna(0)
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
print(f"brake on {1 - calm.mean():.0%} of days; 1987-10-19 calm={calm.loc['1987-10-19']:.0f}, 2020-02-27 {calm.loc['2020-02-27']:.0f}, 2020-03-09 {calm.loc['2020-03-09']:.0f}")
res = {}
for spread in [0.01, 0.005]:
    rows = []
    for k in np.arange(1.5, 5.01, 0.25):
        wk = pd.Series(w4.values * np.where(calm == 1, k, 1.0), index=r.idx)
        rk = run(wk, r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
        st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
        okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
        margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
        rows.append((margin, k, st, okk)); res[(spread, k)] = rk
    for mg, k, st, okk in sorted(rows, key=lambda t: -t[0])[:3]:
        print(f"spread {spread:.1%} k {k:.2f}: {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} dd {st[e]['maxdd']:.1%}" for e in P))
