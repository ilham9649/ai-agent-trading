"""growth-trend timing (philosophical economics, 2016) on chosen rule E: follow a trend exit only when the economy is weak.
base_G = max(ens, 1 - weak): in the market if the trend is up OR the economy is not weak.
frozen before running, no retuning after:
  unemployment: weak = UNRATE above its 12-month average (fred, 1948+)
  industrial production: weak = INDPRO below its 12-month average (fred, 1919+)
  month m value is used from the first trading day of month m+2 (release is in month m+1). current vintage (revisions ignored).
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 2008-2026 AND every holdout
  (unrate: 1950-2007; indpro: 1927-62 and 1963-2007)."""
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

def weak_daily(sid, above):
    m = _fred(sid); m.index = m.index.to_period("M")
    w = (m > m.rolling(12).mean()) if above else (m < m.rolling(12).mean())
    w = w.where(m.rolling(12).mean().notna())
    w.index = w.index + 2                                         # usable from month m+2
    return w.reindex(idx.to_period("M")).set_axis(idx).astype(float)

W = {"unemployment": weak_daily("UNRATE", True), "industrial production": weak_daily("INDPRO", False)}
assert W["unemployment"].loc["2008-06-02"] == 1                  # unemployment rose from 2007-08, known by 2008
H = {"unemployment": {"1950-2007": ("1950-01-01", "2007-12-31")},
     "industrial production": {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31")}}
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
for name, weak in W.items():
    baseG = np.maximum(ens, 1 - weak.fillna(1))                   # before data exists: plain E
    rG, _ = run(rule_e(baseG, idx), d, lag=0, borrow_spread=0.01)
    print(f"\n##### {name}: weak on {weak.mean():.0%} of days; macro overrides a trend exit on {((baseG > ens) & (ens < 1)).mean():.0%} of days")
    ok = True
    for e, (a, b) in (H[name] | {"2008-26": ("2008-01-01", None)}).items():
        print(f"=== {e} ===")
        st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "E": rE, "E growth-trend": rG}.items()}
        for k, s in st.items(): print(f"  {k:16s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
        pt, lo, hi, p = sharpe_diff_ci(rG.loc[a:b], rE.loc[a:b], d.rf, n=1000)
        print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and st["E growth-trend"]["maxdd"] >= st["E"]["maxdd"]
    print("ADOPT" if ok else "REJECT")
    # exit episodes since 2008: does macro keep the rule in?
    print("  trend-off stretches since 2008 and macro state (share weak):")
    off = (ens.loc["2008":] == 0).astype(int); st_ = off.index[(off.diff() == 1).values]
    for s in st_:
        seg = off.loc[s:]; e_ = seg[seg == 0].index[0] if (seg == 0).any() else seg.index[-1]
        print(f"    {s.date()} -> {e_.date()}  weak {weak.loc[s:e_].mean():.0%}  mkt {d.px.loc[e_] / d.px.loc[s] - 1:+.1%}")
