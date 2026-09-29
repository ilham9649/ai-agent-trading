"""improve CL (rule C on the low-vol industry basket) toward the user's goal: better return with lower drawdown.
frozen before running, 3 configs declared together, no retuning after:
  V1 low-beta : 10 industries with the lowest 250-day beta to the market (instead of lowest 60d variance)
  V2 ml in lv : among the 20 lowest-variance industries, the 10 with the highest seed-averaged ml probability (f_seeds.py, 5 seeds)
  V3 lever    : CL basket with season 2x nov-apr / 1x may-oct and cap 2.5x (instead of 1.5/1, cap 2)
  pass = cagr ABOVE CL's AND max drawdown no worse than CL's, in 1927-62 (V2: 1950-62), 1963-2007 AND 2008-2026. 2 bp costs as CL."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_c, tom_days, preholiday_days

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var()
beta = R.rolling(250, min_periods=200).cov(d.r).div(d.r.rolling(250, min_periods=200).var(), axis=0)

def basket(sel_me):
    hold = sel_me.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
    return (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 2e-4).fillna(0), hold

lv_rank = var[me].rank(axis=1, method="first")
b_cl, h_cl = basket(lv_rank <= 10)
b_v1, h_v1 = basket(beta[me].rank(axis=1, method="first") <= 10)
pa = pd.read_csv(D / "ml_industry_prob_seedavg.csv", index_col=[0, 1], parse_dates=[0]).iloc[:, 0].unstack()
pa = pa.reindex(lv_rank.index)
sel2 = pa.where(lv_rank <= 20).rank(axis=1, ascending=False, method="first") <= 10
b_v2, h_v2 = basket(sel2)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
w_v3 = (ens.shift(1).fillna(0) * (1 + winter) * (1 + tom_days(idx, 1, 3)) * (1 + preholiday_days(idx))).clip(upper=2.5)
wC = rule_c(ens, idx)
rCL, _ = run(wC, d.assign(r=b_cl), lag=0, borrow_spread=0.01)
cfg = {"V1 low-beta": run(wC, d.assign(r=b_v1), lag=0, borrow_spread=0.01)[0],
       "V2 ml in low-vol": run(wC, d.assign(r=b_v2), lag=0, borrow_spread=0.01)[0],
       "V3 lever 2/1 cap 2.5": run(w_v3, d.assign(r=b_cl), lag=0, borrow_spread=0.01)[0]}
assert (h_v1.sum(axis=1).loc["1928":] == 10).all() and (h_v2.sum(axis=1).loc["1950-03":] == 10).all()
for k, rX in cfg.items():
    ok, line = True, []
    first = ("1950-03-01", "1962-12-31") if k.startswith("V2") else ("1927-07-01", "1962-12-31")
    for e, (a, bb) in {"old": first, "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rCL.loc[a:bb], d.rf), stats(rX.loc[a:bb], d.rf)
        line.append(f"{e} {a[:4]}: CL {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
        ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
