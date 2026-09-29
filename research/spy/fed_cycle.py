"""monetary policy ("don't fight the fed") on chosen rule E. t-bill yield from the french rf series (daily, 1926+).
frozen before running, no retuning after:
  hiking = annualised t-bill yield today minus 252 trading days ago > +1.0 percentage point (known at the close)
  base_H = ens x (0.5 if hiking else 1), then rule E (tom x pre-holiday, cap 1.5)
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 1927-62, 1963-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
y = (d.rf * TD * 100).rolling(21).mean()                      # annualised yield in %, smoothed one month (daily rf is rounded)
hiking = (y - y.shift(252) > 1.0).astype(float)
assert hiking.loc["2022-12-30"] == 1 and hiking.loc["2021-06-30"] == 0
E_ = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
for e, (a, b) in E_.items(): print(f"{e}: hiking on {hiking.loc[a:b].mean():.0%} of days")
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
rH, _ = run(rule_e(ens * (1 - 0.5 * hiking), idx), d, lag=0, borrow_spread=0.01)
ok = True
for e, (a, b) in E_.items():
    print(f"=== {e} ===")
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "E": rE, "E fed cycle": rH}.items()}
    for k, s in st.items(): print(f"  {k:12s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(rH.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E fed cycle"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
