"""mega-caps: rule E holding the largest size decile (french Portfolios_Formed_on_ME daily, value-weighted "Hi 10", 1926+)
instead of the whole market. signals (trend, calendar) still from the market.
frozen before running, no retuning after:
  pass = vs E on the market, dSharpe >= +0.03 AND max drawdown no worse than E, in 1927-62, 1963-2007 AND 2008-2026.
  (file cached as data/french_Portfolios_Formed_on_ME.csv; site name uses "_Daily_CSV")"""
import numpy as np, pandas as pd
import strat_f
from engine import *
from strat_bc import rule_e

ff = strat_f.french("F-F_Research_Data_Factors")
if not (D / "french_Portfolios_Formed_on_ME.csv").exists():
    strat_f.U = strat_f.U.replace("_daily_CSV", "_Daily_CSV")
me = strat_f.french("Portfolios_Formed_on_ME").mask(lambda x: x <= -0.99)
x = merge_sat(me[["Hi 10"]].join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
d = x[["r", "rf"]].copy(); d["px"] = (1 + d.r).cumprod()
idx = d.index
assert x["Hi 10"].notna().all()
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
w = rule_e(ens, idx)
rE, _ = run(w, d, lag=0, borrow_spread=0.01)
rM, _ = run(w, d.assign(r=x["Hi 10"]), lag=0, borrow_spread=0.01)
ok = True
for e, (a, b) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "top decile b&h": x["Hi 10"], "E": rE, "E on top decile": rM}.items()}
    print(f"=== {e} ===")
    for k, s in st.items(): print(f"  {k:16s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rM.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E on top decile"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
