"""month-end pension rebalancing flows (harvey et al. 2025). when stocks beat bonds month-to-date, balanced funds sell stocks near month end.
frozen before running, no retuning after:
  s = stock mtd return minus 10y bond mtd return, at the close of the 6th-last trading day of the month (known then)
  on the last 5 trading days: position x0.5 if s > 0, else x1.5. on top of current best (ens x season x tom x pre-holiday), cap 3x.
adopt only if sharpe beats current best in BOTH 2008-2026 and the 1963-2007 holdout.
stocks = french market, bond = synthetic 10y treasury from dgs10 (carry + duration x yield change)."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
from strat_f import french
from strat_bc import tom_days

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
y = _fred("DGS10").reindex(d.index).ffill() / 100
dmod = (1 - (1 + y / 2) ** -20) / y
d["br"] = y.shift(1) / TD - dmod.shift(1) * y.diff()
d = d.loc["1962-02-01":].copy()                        # dgs10 starts 1962-01-02

ym = d.index.to_period("M")
k_last = pd.Series(d.index, index=d.index).groupby(ym).cumcount(ascending=False)
mtd = lambda r: (1 + r).groupby(ym).cumprod() - 1
spread = (mtd(d.r) - mtd(d.br)).where(k_last == 5)     # value at close of 6th-last day
s = spread.groupby(ym).transform("max")                # one value per month
window = (k_last < 5).astype(float)
assert window.groupby(ym).sum().max() == 5 and s.loc["2008-10"].iloc[0] == spread.loc["2008-10"].dropna().iloc[0]
m = pd.Series(1.0, index=d.index).mask((window == 1) & (s > 0), 0.5).mask((window == 1) & (s <= 0), 1.5)

E = {"1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
print("raw: mean daily market return (bp) on last 5 days, by sign of s; other days")
for e, (a, b) in E.items():
    r, w, ss = d.r.loc[a:b], window.loc[a:b], s.loc[a:b]
    print(f"  {e}: s>0 {r[(w == 1) & (ss > 0)].mean()*1e4:5.1f}  s<=0 {r[(w == 1) & (ss <= 0)].mean()*1e4:5.1f}  other {r[w == 0].mean()*1e4:4.1f}")

nxt = d.index[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(d.index[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=d.index).astype(float)
px_full = (1 + merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"]})).r).cumprod()   # full history for sma warmup
ens = ((band(px_full, 150, .03) + band(px_full, 200, .03) + band(px_full, 250, .03)) / 3).reindex(d.index)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
tom = tom_days(d.index, 1, 3)
cur = ens.shift(1).fillna(0) * (1 + winter.shift(1).fillna(0)) * (1 + tom) * (1 + ph)
cfg = {"market b&h": pd.Series(1.0, index=d.index), "current best": cur.clip(upper=3), "current x rebalancing": (cur * m).clip(upper=3)}
R = {k: run(w, d, lag=0, borrow_spread=0.01)[0] for k, w in cfg.items()}
for e, (a, b) in E.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:24s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["current x rebalancing"].loc[a:b], R["current best"].loc[a:b], d.rf, n=1000)
    print(f"  layer vs current: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
ok = all(stats(R["current x rebalancing"].loc[a:b], d.rf)["sharpe"] > stats(R["current best"].loc[a:b], d.rf)["sharpe"] for a, b in E.values())
print("ADOPT" if ok else "REJECT")
