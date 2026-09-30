"""drawdown brake on the levered diversified combination (goal: cagr >= 3x index AND max dd no worse than index, 1963-2007 AND 2008-26).
portfolio = R4 (rule C weights x k on the R4 basket; crash brake as r4_lever4) + k x 0.5 x ML + k x 0.5 x CARRY (as r4_div.py).
fixed brake: when the portfolio is > 20% below its own peak, k -> 1 until it is back within 10% of the peak (state from prior closes).
scan one k; borrow spread 1.0% and 0.5%. simulated day by day."""
import numpy as np, pandas as pd
from engine import *
import r4_div as v

idx, r = v.idx, v.r
st0 = idx.get_loc(idx[idx >= "1963-01-01"][0])
R4r, rf, w4, calm = r.b3.values, r.d.rf.values, v.w4.values, v.calm.values
ml, cr = v.ml.values, v.carry.values

def sim(k, spread, trig=0.20, rel=0.10):
    eq, peak, braked, prev, out = 1.0, 1.0, False, 0.0, np.zeros(len(idx))
    for t in range(st0, len(idx)):
        kk = 1.0 if (braked or calm[t] == 0) else k
        p = w4[t] * kk
        rt = p * R4r[t] + max(1 - p, 0) * rf[t] + min(1 - p, 0) * (rf[t] + spread / TD) - abs(p - prev) * 10e-4
        kx = 1.0 if braked else k
        rt += kx * 0.5 * (ml[t] + cr[t])
        out[t], prev = rt, p
        eq *= 1 + rt; peak = max(peak, eq)
        if not braked and eq < peak * (1 - trig): braked = True
        elif braked and eq >= peak * (1 - rel): braked = False
    return pd.Series(out[st0:], index=idx[st0:])

P = {"1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
for spread in [0.01, 0.005]:
    rows = []
    for k in np.arange(1.5, 4.01, 0.25):
        rk = sim(k, spread)
        st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
        okk = all(st[e]["cagr"] >= 3 * mk[e]["cagr"] and st[e]["maxdd"] >= mk[e]["maxdd"] for e in P)
        margin = min(min(st[e]["cagr"] / (3 * mk[e]["cagr"]), mk[e]["maxdd"] / st[e]["maxdd"]) for e in P)
        rows.append((margin, k, st, okk))
    for mg, k, st, okk in sorted(rows, key=lambda t: -t[0])[:3]:
        print(f"spread {spread:.1%} k {k:.2f}: {'MEETS GOAL' if okk else 'fails'} (worst ratio {mg:.2f}) | "
              + " | ".join(f"{e} {st[e]['cagr']:.1%} sh {st[e]['sharpe']:.2f} dd {st[e]['maxdd']:.1%}" for e in P))
