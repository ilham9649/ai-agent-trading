"""diversify first, then lever (goal: cagr >= 3x index AND max dd no worse than index, in 1963-2007 AND 2008-26).
streams (all tested before, not retuned): R4 (rule C weights on the R4 basket), ML = monthly ml long-short industry book (ml_ls.py),
CARRY = 0.5 x (synthetic 10y - t-bill) while 10y - 3m > 1 pt (carry_vix.py).
fixed combination: R4 + 0.5 x ML + 0.5 x CARRY excess returns. leverage k scales ALL streams; extra leverage on R4 only while the
crash brake is off (market within 7% of its 20-day high, r4_lever4.py). borrow t-bill + 1.0% and 0.5%. scan one k."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
import s2_rates2 as r
import ml_ls
from strat_bc import rule_c

idx = r.idx
w4 = rule_c(r.ens * (1 - 0.5 * r.hike.shift(-1).fillna(0)), idx)
calm = (r.d.px >= 0.93 * r.d.px.rolling(20).max()).astype(float).shift(1).fillna(0)
y = _fred("DGS10").reindex(idx).ffill() / 100
dmod = (1 - (1 + y / 2) ** -20) / y
br = y.shift(1) / TD - dmod.shift(1) * y.diff()
steep = ((_fred("DGS10") - _fred("DTB3")).reindex(idx).ffill() > 1.0).astype(float).shift(1)
carry = (steep * (br - r.d.rf)).fillna(0)
ml = ml_ls.book.reindex(idx).fillna(0)
P = {"1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
print(f"corr since 1963: R4-ML {run(w4, r.d.assign(r=r.b3), lag=0)[0].loc['1963':].corr(ml.loc['1963':]):+.2f}, "
      f"R4-CARRY {run(w4, r.d.assign(r=r.b3), lag=0)[0].loc['1963':].corr(carry.loc['1963':]):+.2f}, ML-CARRY {ml.loc['1963':].corr(carry.loc['1963':]):+.2f}")
for spread in [0.01, 0.005]:
    rows = []
    for k in np.arange(1.5, 5.01, 0.25):
        wk = pd.Series(w4.values * np.where(calm == 1, k, 1.0), index=idx)
        rr = run(wk, r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
        rk = (rr + k * 0.5 * ml + k * 0.5 * carry).loc["1963":]
        st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
        okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
        margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
        rows.append((margin, k, st, okk))
    for mg, k, st, okk in sorted(rows, key=lambda t: -t[0])[:3]:
        print(f"spread {spread:.1%} k {k:.2f}: {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} sh {st[e]['sharpe']:.2f} dd {st[e]['maxdd']:.1%}" for e in P))
