"""short-volatility sleeve: cboe ^SHORTVOL (short vix short-term futures, daily, 2005-12+), as an overlay on the current best rule.
frozen before running, no retuning after:
  gate = trend ensemble fully on (all 3 bands) AND vix < vix3m (contango), both at the prior close
  overlay = 0.2 x gate x shortvol daily return (treated as excess return, no cash credit), 5 bp per unit of sleeve turnover
adopt only if dSharpe vs current >= +0.03 in BOTH 2008-2016 and 2017-2026 (no pre-2004 data exists, so no old holdout)."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred
from strat_f import french
from strat_bc import tom_days

sv = yf.download("^SHORTVOL", period="max", progress=False, auto_adjust=True)["Close"].squeeze()
ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
nxt = idx[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(idx[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=idx).astype(float)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
cur = (ens.shift(1).fillna(0) * (1 + winter) * (1 + tom_days(idx, 1, 3)) * (1 + ph)).clip(upper=3)
base, _ = run(cur, d, lag=0, borrow_spread=0.01)

d = d.loc["2008-01-01":].copy(); base = base.loc[d.index]
vix, vix3m = _fred("VIXCLS").reindex(d.index).ffill(), _fred("VXVCLS").reindex(d.index).ffill()
rs = sv.pct_change().reindex(d.index)

assert rs.isna().sum() < 30, rs.isna().sum()   # ~18 single-day gaps; the next day's return spans both days, so the total is kept
rs = rs.fillna(0)
gate = ((ens.reindex(d.index) == 1) & (vix < vix3m)).astype(float).shift(1).fillna(0)
w = 0.2 * gate
over = w * rs - w.diff().abs().fillna(0) * 5e-4
R = {"market b&h": d.r, "current best": base, "current + short vol 0.2": base + over, "short vol sleeve alone (1x, always)": rs + d.rf}
print(f"gate on {gate.mean():.0%} of days; sleeve alone cagr {(1 + rs).prod() ** (TD / len(rs)) - 1:.1%}")
E = {"2008-16": ("2008-01-01", "2016-12-31"), "2017-26": ("2017-01-01", None), "2008-26": ("2008-01-01", None)}
dS = {}
for e, (a, b) in E.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:36s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%} worst day {r.loc[a:b].min():6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["current + short vol 0.2"].loc[a:b], base.loc[a:b], d.rf, n=1000)
    dS[e] = pt; print(f"  overlay vs current: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
for day in ["2018-02-05", "2020-02-24", "2020-03-16", "2024-08-05"]:
    if pd.Timestamp(day) in d.index: print(f"{day}: gate {gate.loc[day]:.0f}, shortvol {rs.loc[day]:+.1%}, overlay {over.loc[day]:+.2%}")
print("ADOPT" if dS["2008-16"] >= 0.03 and dS["2017-26"] >= 0.03 else "REJECT")

# after the frozen verdict: sensitivity (not used to choose the rule)
print("\nsensitivity, 2008-26 sharpe / maxdd / cagr (current best: 0.76):")
trend_on = (ens.reindex(d.index) == 1).astype(float).shift(1).fillna(0)
for lab, g, wt, bp in [("gate, 0.1", gate, .1, 5), ("gate, 0.3", gate, .3, 5), ("gate, 0.2, 20bp", gate, .2, 20),
                       ("trend only (no contango), 0.2", trend_on, .2, 5), ("no gate, 0.2", pd.Series(1.0, index=d.index), .2, 5)]:
    ww = wt * g; rr = base + ww * rs - ww.diff().abs().fillna(0) * bp * 1e-4
    st = stats(rr, d.rf); print(f"  {lab:32s} {st['sharpe']:.2f} {st['maxdd']:6.1%} {st['cagr']:6.1%}")
rr = base + over
dd = (1 + rr).cumprod() / (1 + rr).cumprod().cummax() - 1
print("worst drawdown trough:", dd.idxmin().date(), f"{dd.min():.1%}; overlay sum over that year {over.loc[str(dd.idxmin().year)].sum():+.1%}")
