"""user goal: cagr >= 3x the index (market) cagr with max drawdown no worse than the index's, per period.
R4 scaled by k (all rule C weights x k, cap 2k), borrow at t-bill + 1%, 10 bp costs. no new signal."""
import numpy as np, pandas as pd
from engine import *
import s2_rates2 as r
from strat_bc import rule_c

w4 = rule_c(r.ens * (1 - 0.5 * r.hike.shift(-1).fillna(0)), r.idx)
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
print("index: " + "  ".join(f"{e} {m['cagr']:.1%} (x3 = {3 * m['cagr']:.1%}) dd {m['maxdd']:.1%}" for e, m in mk.items()))
for k in [1, 1.5, 2, 2.5, 2.75, 3, 3.5]:
    rk = run(w4 * k, r.d.assign(r=r.b3), lag=0, borrow_spread=0.01)[0]
    cells = []
    for e, (a, b) in P.items():
        s = stats(rk.loc[a:b], r.d.rf); ok = s["cagr"] >= 3 * mk[e]["cagr"] and s["maxdd"] >= mk[e]["maxdd"]
        cells.append(f"{e} {s['cagr']:5.1%} dd {s['maxdd']:6.1%} {'OK' if ok else '--'}")
    print(f"k={k:<4} max {2 * k:.1f}x | " + " | ".join(cells))
