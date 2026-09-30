"""volatility-targeted diversified combination (goal: cagr >= 3x index AND max dd no worse than index, in 1963-2007 AND 2008-26).
unit portfolio u = R4 (rule C weights, R4 basket) + 0.5 x ML long-short + 0.5 x CARRY (as r4_div.py).
fixed sizing rule: m = target_vol / 60-day realised vol of u (prior close), capped at 4; while the crash brake is on
(market > 7% below its 20-day high) m is capped at 1. R4 weights x m (borrow t-bill + spread), ML and CARRY x m.
scan target_vol; borrow spread 1.0% and 0.5%."""
import numpy as np, pandas as pd
from engine import *
import r4_div as v

idx, r = v.idx, v.r
u = run(v.w4, r.d.assign(r=r.b3), lag=0, borrow_spread=0.01)[0] + 0.5 * v.ml + 0.5 * v.carry
rv = u.rolling(60).std() * TD ** .5
P = {"1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
for spread in [0.01, 0.005]:
    rows = []
    for tv in np.arange(0.15, 0.46, 0.025):
        m = (tv / rv).clip(upper=4).shift(1)
        m = m.where(v.calm == 1, m.clip(upper=1)).fillna(1)
        rr = run(v.w4 * m, r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
        rk = (rr + m * 0.5 * v.ml + m * 0.5 * v.carry).loc["1963":]
        st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
        okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
        margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
        rows.append((margin, tv, st, okk, m.loc["1963":].mean()))
    for mg, tv, st, okk, mm in sorted(rows, key=lambda t: -t[0])[:3]:
        print(f"spread {spread:.1%} target vol {tv:.1%} (avg m {mm:.2f}): {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} sh {st[e]['sharpe']:.2f} dd {st[e]['maxdd']:.1%}" for e in P))
