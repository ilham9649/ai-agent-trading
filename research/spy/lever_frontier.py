"""what leverage does to S2 (rule C timing on the low-vol + momentum industry basket): return vs drawdown. no new signal.
exposure = k x rule C weights (cap 2k), borrow at t-bill + 1%, 10 bp costs on the basket."""
import numpy as np, pandas as pd
import cl_next as c
from engine import *

to = c.h_s2.astype(float).diff().abs().sum(axis=1) / 20
bs = (c.R.where(c.h_s2).mean(axis=1) - to.fillna(0) * 10e-4).fillna(0)
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
for k in [1.0, 1.5, 2.0, 3.0, 4.0]:
    r = run(c.wC * k, c.d.assign(r=bs), lag=0, borrow_spread=0.01)[0]
    print(f"k={k:.1f} (max {2 * k:.0f}x): " + "  ".join(f"{e} {stats(r.loc[a:b], c.d.rf)['cagr']:6.1%} dd {stats(r.loc[a:b], c.d.rf)['maxdd']:6.1%}" for e, (a, b) in P.items()))
