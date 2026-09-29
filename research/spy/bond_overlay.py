"""permanent treasury sleeve (no timing) as an overlay on chosen rule E: stock/bond diversification.
bond = synthetic 10y treasury total return from fred dgs10 (carry + duration x yield change), 1962+.
frozen before running, no retuning after:
  overlay = E + 0.5 x (bond return - t-bill), daily; 1 bp cost per unit of daily turnover ignored (static weight, futures-like)
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in the 1963-2007 holdout AND 2008-2026."""
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
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
y = _fred("DGS10").reindex(idx).ffill() / 100
dmod = (1 - (1 + y / 2) ** -20) / y
br = (y.shift(1) / TD - dmod.shift(1) * y.diff()).loc["1962-02-01":]
assert br.notna().mean() > 0.99
br = br.fillna(0)
rB = (rE + 0.5 * (br - d.rf)).dropna()
ok = True
for e, (a, b) in {"1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None), "2022 only (info)": ("2022-01-01", "2022-12-31")}.items():
    print(f"=== {e} ===  corr(E, bond) {rE.loc[a:b].corr(br.loc[a:b]):+.2f}")
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "10y bond": br, "E": rE, "E + 0.5 bond": rB}.items()}
    for k, s in st.items(): print(f"  {k:12s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    if "info" in e: continue
    pt, lo, hi, _ = sharpe_diff_ci(rB.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E + 0.5 bond"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
