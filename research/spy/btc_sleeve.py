"""bitcoin trend sleeve as an overlay on chosen rule E. btc from yfinance BTC-USD (2014-09+), weekends compounded into the next us trading day.
frozen before running, no retuning after:
  on = btc close above its 200-day sma (btc calendar days), known at the prior close
  overlay = E + 0.1 x on x (btc return - t-bill), 10 bp per unit of sleeve turnover
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in BOTH 2015-2020 and 2021-2026 (no older data exists)."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
btc = yf.download("BTC-USD", period="max", progress=False, auto_adjust=True)["Close"].squeeze()
btc.index = btc.index.tz_localize(None)
on_cal = (btc > btc.rolling(200).mean()).astype(float).where(btc.rolling(200).mean().notna())
px = btc.reindex(idx.union(btc.index)).ffill().reindex(idx).loc["2015-01-01":]
on = on_cal.reindex(idx.union(btc.index)).ffill().reindex(idx).loc[px.index]
r_btc = px.pct_change()
w = 0.1 * on.shift(1)
over = (w * (r_btc - d.rf.loc[px.index]) - w.diff().abs() * 1e-3).fillna(0)
assert r_btc.iloc[1:].notna().all() and (px.index[0] >= pd.Timestamp("2015-01-01"))
rO = rE.loc[px.index] + over
print(f"sleeve on {on.mean():.0%} of days; corr(btc, market) {r_btc.corr(d.r.loc[px.index]):+.2f}")
ok = True
for e, (a, b) in {"2015-20": ("2015-01-01", "2020-12-31"), "2021-26": ("2021-01-01", None), "2015-26 (info)": ("2015-01-01", None)}.items():
    print(f"=== {e} ===")
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r.loc[px.index], "E": rE.loc[px.index], "E + 0.1 btc trend": rO}.items()}
    for k, s in st.items(): print(f"  {k:18s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rO.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    if "info" not in e: ok &= pt >= 0.03 and st["E + 0.1 btc trend"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
