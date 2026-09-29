"""rule E run on the nasdaq-100 (^NDX price index, 1985-10+, no dividends added: conservative) with its own trend and calendar.
frozen before running, no retuning after:
  pass = vs E on the market, dSharpe >= +0.03 AND max drawdown no worse than E, in the 1987-2007 holdout AND 2008-2026."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, d.index), d, lag=0, borrow_spread=0.01)
n = yf.download("^NDX", period="max", progress=False, auto_adjust=True)["Close"].squeeze()
n.index = n.index.tz_localize(None)
q = pd.DataFrame({"px": n}).join(d[["rf"]], how="inner")
q["r"] = q.px.pct_change()
assert q.index[0] < pd.Timestamp("1986-01-01") and q.r.iloc[1:].abs().max() < 0.2
ensq = (band(q.px, 150, .03) + band(q.px, 200, .03) + band(q.px, 250, .03)) / 3
rQ, _ = run(rule_e(ensq, q.index), q, lag=0, borrow_spread=0.01)
ok = True
for e, (a, b) in {"1987-2007": ("1987-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    print(f"=== {e} ===")
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "ndx b&h (price)": q.r, "E": rE, "E on ndx": rQ}.items()}
    for k, s in st.items(): print(f"  {k:16s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rQ.loc[a:b].reindex(rE.loc[a:b].index).fillna(0), rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E on ndx"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
