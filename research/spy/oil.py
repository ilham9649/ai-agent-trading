"""oil shocks (driesprong, jacobsen, maat 2008: oil price rises predict lower stock returns next month). 2008-26 is post-publication.
oil = fred WTISPLC (monthly average wti spot, 1946+).
frozen before running, no retuning after:
  shock = oil rose more than 10% in month m-1 (known at the end of month m-1)
  during month m: base = ens x 0.5 if shock else ens, then rule E (tom x pre-holiday, cap 1.5)
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in the 1948-2007 holdout AND 2008-2026."""
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
o = _fred("WTISPLC"); o.index = o.index.to_period("M")
shock_m = (o.pct_change() > 0.10)
shock_m.index = shock_m.index + 1                                        # month m-1 change applies in month m
shock = shock_m.reindex(idx.to_period("M")).set_axis(idx).fillna(False).astype(float)
assert shock.loc["1990-09-04"] == 1                                       # aug 1990 oil spike (kuwait)
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
rO, _ = run(rule_e(ens * (1 - 0.5 * shock), idx), d, lag=0, borrow_spread=0.01)
ok = True
for e, (a, b) in {"1948-2007": ("1948-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    x = d.r.loc[a:b]; s_ = shock.loc[a:b]
    print(f"=== {e} === shock months {s_.groupby(s_.index.to_period('M')).first().sum():.0f}; "
          f"mean daily mkt return in shock months {x[s_ == 1].mean()*1e4:.1f} bp vs {x[s_ == 0].mean()*1e4:.1f} bp")
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "E": rE, "E x oil": rO}.items()}
    for k, s in st.items(): print(f"  {k:8s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rO.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E x oil"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
