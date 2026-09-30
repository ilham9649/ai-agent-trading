"""rate-aware S2. diagnosis (same data, so not fully clean): S2's worst drawdowns since 1950 were rate-rise episodes
(1973, 1978, 1980, 1981, 1994, 1999-2000, 2022-23); low-vol industries act like bonds.
hiking = t-bill yield up > 1 pt over 252 days (same definition as fed_cycle.py, not retuned). 2 configs, frozen before running:
  R1 switch : while hiking, hold the market instead of the S2 basket (same rule C weights)
  R2 cash   : while hiking, rule C base x 0.5 on the S2 basket
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
sel = m12[me].where(var[me].rank(axis=1, method="first") <= 20).rank(axis=1, ascending=False, method="first") <= 10
hold = sel.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
s2 = (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 10e-4).fillna(0)
y = (d.rf * TD * 100).rolling(21).mean()
hike = (y - y.shift(252) > 1.0).astype(float).shift(1).fillna(0)        # known at the prior close
wC = rule_c(ens, idx)
rS2 = run(wC, d.assign(r=s2), lag=0, borrow_spread=0.01)[0]
sw = hike.diff().abs().fillna(0) * 10e-4                                  # cost of swapping the whole basket
b1 = s2.where(hike == 0, d.r) - sw
cfg = {"R1 switch to market": run(wC, d.assign(r=b1), lag=0, borrow_spread=0.01)[0],
       "R2 half while hiking": run(rule_c(ens * (1 - 0.5 * hike.shift(-1).fillna(0)), idx), d.assign(r=s2), lag=0, borrow_spread=0.01)[0]}
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
print("hiking share of days:", {e: f"{hike.loc[a:b].mean():.0%}" for e, (a, b) in P.items()})
for k, rX in cfg.items():
    ok, line = True, []
    for e, (a, b) in P.items():
        s0, s1 = stats(rS2.loc[a:b], d.rf), stats(rX.loc[a:b], d.rf)
        line.append(f"{e}: S2 {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
        ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
