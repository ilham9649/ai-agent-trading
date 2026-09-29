"""recession alarms as a risk cut on chosen rule E (never adds exposure, unlike growth-trend).
frozen before running, no retuning after:
  sahm   = 3-month avg UNRATE minus its low over the prior 12 months >= 0.5 pt; month m value used from month m+2
  claims = 4-week avg initial claims (ICSA, weekly, week ending saturday) >= 1.2 x its 52-week low; used from the next monday + 5 days
  base   = ens x 0.5 while the alarm is on, then rule E (tom x pre-holiday, cap 1.5)
  pass   = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in the holdout (sahm 1950-2007, claims 1968-2007) AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
u = _fred("UNRATE"); u.index = u.index.to_period("M")
u3 = u.rolling(3).mean()
sahm_m = (u3 - u3.shift(1).rolling(12).min() >= 0.5); sahm_m.index = sahm_m.index + 2
sahm = sahm_m.reindex(idx.to_period("M")).set_axis(idx).fillna(False).astype(float)
c = _fred("ICSA"); c4 = c.rolling(4).mean()
cl = (c4 >= 1.2 * c4.rolling(52).min()).astype(float)
cl.index = cl.index + pd.Timedelta(days=7)                   # released the thursday after the week; used from the following saturday
claims = cl.reindex(idx.union(cl.index)).ffill().reindex(idx).fillna(0)
assert sahm.loc["2008-06-02"] == 1 and claims.loc["2020-04-15"] == 1
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
for name, al, hold in [("sahm", sahm, ("1950-01-01", "2007-12-31")), ("claims", claims, ("1968-01-01", "2007-12-31"))]:
    rA, _ = run(rule_e(ens * (1 - 0.5 * al), idx), d, lag=0, borrow_spread=0.01)
    print(f"\n##### {name}: alarm on {al.loc[hold[0]:].mean():.0%} of days")
    ok = True
    for e, (a, b) in {f"{hold[0][:4]}-2007": hold, "2008-26": ("2008-01-01", None)}.items():
        st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "E": rE, f"E x {name}": rA}.items()}
        print(f"=== {e} ===")
        for k, s in st.items(): print(f"  {k:10s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
        pt, lo, hi, _ = sharpe_diff_ci(rA.loc[a:b], rE.loc[a:b], d.rf, n=1000)
        print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and st[f"E x {name}"]["maxdd"] >= st["E"]["maxdd"]
    print("ADOPT" if ok else "REJECT")
