"""option income: run chosen rule E on the cboe put-write index (PUT: 1-month atm spx puts, cash collateralised) instead of the market.
signals (trend, calendar) still come from the market; only the asset held changes.
frozen before running, no retuning after:
  pass = vs E on the market, dSharpe >= +0.03 AND max drawdown no worse than E, in BOTH 2008-16 and 2017-26.
  (daily PUT data starts 2007; earlier rows are 7 scattered points, so no old holdout exists.)"""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
p = pd.read_csv(D / "cboe_PUT.csv", index_col=0, parse_dates=True).iloc[:, 0]
p = p.loc["2007-01-01":]
pr = p.reindex(idx.union(p.index)).ffill().reindex(idx).pct_change()
dp = d.loc["2008-01-01":].copy(); dp["r"] = pr.loc[dp.index]
assert dp.r.notna().all() and dp.r.abs().max() < 0.2
w = rule_e(ens, idx).loc[dp.index]
rE, _ = run(w, d.loc[dp.index], lag=0, borrow_spread=0.01)
rP, _ = run(w, dp, lag=0, borrow_spread=0.01)
ok = True
for e, (a, b) in {"2008-16": ("2008-01-01", "2016-12-31"), "2017-26": ("2017-01-01", None), "2008-26 (info)": ("2008-01-01", None)}.items():
    print(f"=== {e} ===")
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r.loc[dp.index], "PUT index": dp.r, "E": rE, "E on PUT": rP}.items()}
    for k, s in st.items(): print(f"  {k:10s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rP.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    if "info" not in e: ok &= pt >= 0.03 and st["E on PUT"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
