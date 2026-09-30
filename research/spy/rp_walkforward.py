"""walk-forward risk-parity weights for the R4 + ML + CARRY + FX mix (replaces the fixed 1 / 0.5 / 0.5 / 0.5 notionals).
frozen before running:
  each month end, weight_i = 1 / (trailing 756-day vol of stream i), rescaled so the four weights keep the same trailing
  portfolio vol as the fixed mix would have had (uses trailing covariance only). weights held for the next month.
  R4 part is implemented through its rule C weights x (weight_R4 x k) with the crash brake on k > 1; other streams x weight x k.
  k = largest k on a 0.25 grid whose 1973-2007 max drawdown stays within the index's; test on 2008-2026 (1% borrow spread).
  compare with the fixed mix calibrated the same way."""
import numpy as np, pandas as pd
from engine import *
import r4_div as v
import fx_tsmom as f

idx, r = v.idx, v.r
R4u = run(v.w4, r.d.assign(r=r.b3), lag=0, borrow_spread=0.01)[0]
X = pd.DataFrame({"R4": R4u - r.d.rf, "ML": v.ml, "CARRY": v.carry, "FX": f.fx}).reindex(idx).fillna(0)
fixed = pd.Series({"R4": 1.0, "ML": 0.5, "CARRY": 0.5, "FX": 0.5})
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
rows = {}
for t in idx[me.values]:
    h = X.loc[:t].iloc[-756:]
    if len(h) < 756 or (h.std() == 0).any(): continue
    iv = 1 / h.std(); C = h.cov()
    w = iv * np.sqrt(fixed @ C @ fixed) / np.sqrt(iv @ C @ iv)
    rows[t] = w
Wm = pd.DataFrame(rows).T
W = Wm.reindex(idx).shift(1).ffill()
a0 = "1973-01-01"
print("average walk-forward weights 1973-2007:", W.loc[a0:"2007"].mean().round(2).to_dict(), "| 2008-26:", W.loc["2008":].mean().round(2).to_dict())

def mix(k, weights):
    get = (lambda c: fixed[c]) if weights is fixed else (lambda c: W[c])
    wr = pd.Series(fixed["R4"], index=idx) if weights is fixed else W["R4"]
    wk = pd.Series(v.w4.values * wr.values * np.where(v.calm == 1, k, 1.0), index=idx)
    rr = run(wk.fillna(0), r.d.assign(r=r.b3), lag=0, borrow_spread=0.01)[0]
    return (rr + k * sum(get(c) * X[c] for c in ["ML", "CARRY", "FX"])).loc[a0:]

P = {"1973-2007 (calibrate)": (a0, "2007-12-31"), "2008-26 (test)": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
for name, weights in [("fixed 1/0.5/0.5/0.5", fixed), ("walk-forward risk parity", "rp")]:
    kc = None
    for k in np.arange(1.0, 5.01, 0.25):
        if stats(mix(k, weights).loc[a0:"2007"], r.d.rf)["maxdd"] >= mk["1973-2007 (calibrate)"]["maxdd"]: kc = k
    rk = mix(kc, weights); st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
    t, m = st["2008-26 (test)"], mk["2008-26 (test)"]
    print(f"{name}: k {kc:.2f} | " + " | ".join(f"{e} {s['cagr']:.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:.1%}" for e, s in st.items())
          + f" | 2008-26 = {t['cagr'] / m['cagr']:.2f}x index, dd {'within' if t['maxdd'] >= m['maxdd'] else 'WORSE than'} index")
