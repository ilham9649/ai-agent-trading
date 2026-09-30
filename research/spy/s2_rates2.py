"""rate-aware S2, part 2 (follows s2_rates.py; same data, so not fully clean). hiking definition unchanged.
2 configs, frozen before running:
  R3 momentum in hikes : while hiking, basket = top 10 of all 49 industries by 12-1 month return (no low-vol filter); else S2 basket
  R4 = R3 with rule C base x 0.5 while hiking
  pass = cagr ABOVE S2 AND max drawdown no worse than S2, in 1927-62, 1963-2007 AND 2008-2026. 10 bp costs."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_c

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var(); lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
m12 = lvl.shift(21) / lvl.shift(252) - 1

def basket(sel):
    h = sel.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
    return R.where(h).mean(axis=1).fillna(0), h

s2, h2 = basket(m12[me].where(var[me].rank(axis=1, method="first") <= 20).rank(axis=1, ascending=False, method="first") <= 10)
mo, hm = basket(m12[me].rank(axis=1, ascending=False, method="first") <= 10)
y = (d.rf * TD * 100).rolling(21).mean()
hike = (y - y.shift(252) > 1.0).astype(float).shift(1).fillna(0)
H = pd.DataFrame(np.where(hike.values[:, None] == 1, hm.values, h2.values), index=idx, columns=R.columns)
cost = lambda h: h.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 10e-4
b3 = s2.where(hike == 0, mo) - cost(H)
s2n = s2 - cost(h2)
wC = rule_c(ens, idx)
rS2 = run(wC, d.assign(r=s2n), lag=0, borrow_spread=0.01)[0]
cfg = {"R3 momentum in hikes": run(wC, d.assign(r=b3), lag=0, borrow_spread=0.01)[0],
       "R4 R3 at half in hikes": run(rule_c(ens * (1 - 0.5 * hike.shift(-1).fillna(0)), idx), d.assign(r=b3), lag=0, borrow_spread=0.01)[0]}
assert (hm.sum(axis=1).loc["1928":] == 10).all() and (H.sum(axis=1).loc["1928":] == 10).all()
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
for k, rX in cfg.items():
    ok, line = True, []
    for e, (a, b) in P.items():
        s0, s1 = stats(rS2.loc[a:b], d.rf), stats(rX.loc[a:b], d.rf)
        line.append(f"{e}: S2 {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
        ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
