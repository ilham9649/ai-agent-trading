"""leverage engineering for the user goal (cagr >= 3x index AND max dd no worse than the index, in 1927-62, 1963-2007, 2008-26).
fixed configurations (no new signals):
  spread: borrow at t-bill + 1.0% (as before) or + 0.5% (futures-like financing)
  volcap: none, or exposure x min(1, 12% / 20-day realised vol of the held basket, known at the prior close)
for each config, scan k (single k for all periods) and report whether any k meets the goal in all periods."""
import numpy as np, pandas as pd
from engine import *
import s2_rates2 as r
from strat_bc import rule_c

w4 = rule_c(r.ens * (1 - 0.5 * r.hike.shift(-1).fillna(0)), r.idx)
rv = r.b3.rolling(20).std() * TD ** .5
vc = (0.12 / rv).clip(upper=1).shift(1).fillna(1)
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
for spread in [0.01, 0.005]:
    for vname, m in [("no volcap", 1.0), ("volcap 12%", vc)]:
        best = None
        for k in np.arange(1.5, 6.01, 0.25):
            rk = run(w4 * m * k, r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
            st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
            okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
            margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
            if best is None or margin > best[0]: best = (margin, k, st, okk)
        mg, k, st, okk = best
        print(f"spread {spread:.1%}, {vname:10s}: best k {k:.2f} {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} dd {st[e]['maxdd']:.1%}" for e in P))
