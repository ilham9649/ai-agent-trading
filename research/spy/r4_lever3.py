"""leverage only in a healthy short-term trend (goal: cagr >= 3x index AND max dd no worse than index, all 3 periods).
fixed rule: exposure = k x R4 weights while the market closed above its 50-day sma at the prior close; otherwise plain R4 weights (k = 1).
scan a single k for all periods; borrow spread 1.0% and 0.5%."""
import numpy as np, pandas as pd
from engine import *
import s2_rates2 as r
from strat_bc import rule_c

w4 = rule_c(r.ens * (1 - 0.5 * r.hike.shift(-1).fillna(0)), r.idx)
fast = (r.d.px > r.d.px.rolling(50).mean()).astype(float).shift(1).fillna(0)
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
print(f"extra leverage allowed on {fast.mean():.0%} of days; on 1987-10-19: {fast.loc['1987-10-19']:.0f}; on 2020-02-24: {fast.loc['2020-02-24']:.0f}")
for spread in [0.01, 0.005]:
    rows = []
    for k in np.arange(1.5, 5.01, 0.25):
        wk = w4 * np.where(fast == 1, k, 1.0)
        rk = run(pd.Series(wk, index=r.idx), r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
        st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
        okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
        margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
        rows.append((margin, k, st, okk))
    for mg, k, st, okk in sorted(rows, key=lambda t: -t[0])[:3]:
        print(f"spread {spread:.1%} k {k:.2f}: {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} dd {st[e]['maxdd']:.1%}" for e in P))
