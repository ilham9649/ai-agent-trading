"""defensive multi-factor basket (low-vol + momentum + quality), survivorship-free (french/crsp portfolios incl. delisted stocks), timed by rule C.
parts: S2 = 10 of the 20 lowest-variance industries with the best 12-1 return (as cl_next.py), 10 bp costs
       MOM = "BIG HiPRIOR" (largest size quintile x top prior 12-2 return quintile, 25_Portfolios_ME_Prior_12_2 daily, 1926-11+), 1%/yr cost haircut
       QUAL = "BIG HiOP" (largest size quintile x top operating profitability quintile, 25_Portfolios_ME_OP_5x5 daily, 1963-07+), 0.3%/yr haircut
frozen before running, 2 configs declared together, no retuning after:
  MF2 = equal weight of S2 and MOM (daily rebalanced), 1927+
  MF3 = equal weight of S2, MOM and QUAL, 1963-07+
  pass = cagr ABOVE S2's AND max drawdown no worse than S2's (both under rule C) in every period available:
         MF2: 1927-62, 1963-2007, 2008-26; MF3: 1964-2007, 2008-26."""
import numpy as np, pandas as pd
import strat_f
from engine import *
from strat_bc import rule_c

ind = strat_f.french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = strat_f.french("F-F_Research_Data_Factors")
if not (D / "french_25_Portfolios_ME_Prior_12_2.csv").exists(): strat_f.U = strat_f.U.replace("_daily_CSV", "_Daily_CSV")
pr = strat_f.french("25_Portfolios_ME_Prior_12_2").mask(lambda x: x <= -0.99)
strat_f.U = strat_f.U.replace("_Daily_CSV", "_daily_CSV")
op = strat_f.french("25_Portfolios_ME_OP_5x5").mask(lambda x: x <= -0.99)
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"], "mom": pr["BIG HiPRIOR"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
mom_p = x["mom"] - 0.01 / TD
qual = (op["BIG HiOP"].reindex(idx) - 0.003 / TD)
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var()
lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
m12 = lvl.shift(21) / lvl.shift(252) - 1
sel = m12[me].where(var[me].rank(axis=1, method="first") <= 20).rank(axis=1, ascending=False, method="first") <= 10
hold = sel.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
s2 = (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 10e-4).fillna(0)
assert (hold.sum(axis=1).loc["1928":] == 10).all() and qual.loc["1963-08":].notna().all() and mom_p.loc["1927":].notna().all()
wC = rule_c(ens, idx)
R_ = lambda b: run(wC, d.assign(r=b), lag=0, borrow_spread=0.01)[0]
rS2 = R_(s2)
cfg = {"MF2 lowvol-mom + big mom": (R_((s2 + mom_p) / 2), ["1927-62", "1963-2007", "2008-26"]),
       "MF3 + big quality": (R_(((s2 + mom_p + qual) / 3).where(qual.notna(), s2)), ["1964-2007", "2008-26"])}
P = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "1964-2007": ("1964-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
for k, (rX, pers) in cfg.items():
    ok, line = True, []
    for e in pers:
        a, b = P[e]; s0, s1 = stats(rS2.loc[a:b], d.rf), stats(rX.loc[a:b], d.rf)
        line.append(f"{e}: S2 {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
        ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
for nm, b in {"S2 basket": s2, "big mom": mom_p, "big quality": qual}.items():
    s = stats(b.loc["2008":].dropna(), d.rf); print(f"{nm:11s} buy & hold 2008-26: {s['cagr']:.1%}/{s['sharpe']:.2f}/{s['maxdd']:.1%}")
