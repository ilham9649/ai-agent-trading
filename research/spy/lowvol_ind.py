"""daily low-volatility basket from the 49 french industries (1926+), run under chosen rule E instead of the market.
frozen before running, no retuning after:
  basket  = at each month end, the 10 industries with the lowest 60-day return variance (>= 40 obs), equal weight, held next month
  rule    = E (market trend ensemble x tom x pre-holiday, cap 1.5): signals from the market, asset = the basket
  pass    = vs E on the market, dSharpe >= +0.03 AND max drawdown no worse than E, in 1927-62, 1963-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
var = R.rolling(60, min_periods=40).var()
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
pick = var[me].rank(axis=1, method="first") <= 10                              # month-end selection
hold = pick.reindex(idx).shift(1).ffill().fillna(False).astype(bool)           # held from the next trading day
assert (hold.sum(axis=1).loc["1928":] == 10).all()
b = R.where(hold).mean(axis=1)
turn = hold.astype(float).diff().abs().sum(axis=1) / 20                          # one-way share of basket replaced
db = d.assign(r=b - turn * 2e-4)                                                # 2 bp per unit of basket turnover
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
w = rule_e(ens, idx)
rE, _ = run(w, d, lag=0, borrow_spread=0.01)
rL, _ = run(w, db, lag=0, borrow_spread=0.01)
print("most-held industries:", hold.loc["1928":].mean().nlargest(8).round(2).to_dict())
print(f"basket one-way turnover per rebalance {turn[turn > 0].mean():.0%}")
ok = True
for e, (a, b_) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    st = {k: stats(v.loc[a:b_], d.rf) for k, v in {"spy": d.r, "low-vol basket": db.r, "E": rE, "E on low-vol": rL}.items()}
    print(f"=== {e} ===")
    for k, s in st.items(): print(f"  {k:14s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%} vol {s['vol']:5.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rL.loc[a:b_], rE.loc[a:b_], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E on low-vol"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
