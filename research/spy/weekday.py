"""weekend effect (french 1980): returns from friday close to the first close of the week are low.
frozen before running, no retuning after:
  position x0.5 on the first trading day of each week, on top of the current best (ens x season x tom x pre-holiday), cap 3x
adopt only if dSharpe vs current is >= +0.03 in BOTH 2008-2026 and the 1927-1962 holdout (smaller = tie)."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
wk = idx.to_period("W")
first = pd.Series(pd.Series(idx, index=idx).groupby(wk).cumcount().values == 0, index=idx).astype(float)
assert first.loc["2025-09-02"] == 1 and first.loc["2025-09-03"] == 0      # tuesday after labor day opens the week

E = {"1927-62 holdout": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
print("raw mean daily return (bp): first day of week vs other days")
for e, (a, b) in E.items():
    r, f = d.r.loc[a:b], first.loc[a:b]
    print(f"  {e}: first {r[f == 1].mean()*1e4:5.1f}  other {r[f == 0].mean()*1e4:5.1f}")

ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
nxt = idx[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(idx[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=idx).astype(float)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
cur = ens.shift(1).fillna(0) * (1 + winter) * (1 + tom_days(idx, 1, 3)) * (1 + ph)
cfg = {"market b&h": pd.Series(1.0, index=idx).shift(1).fillna(0), "current best": cur.clip(upper=3), "current x monday 0.5": (cur * (1 - 0.5 * first)).clip(upper=3)}
R = {k: run(w, d, lag=0, borrow_spread=0.01)[0] for k, w in cfg.items()}
dS = {}
for e, (a, b) in E.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:22s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["current x monday 0.5"].loc[a:b], R["current best"].loc[a:b], d.rf, n=1000)
    dS[e] = pt; print(f"  layer vs current: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
print("ADOPT" if dS["1927-62 holdout"] >= 0.03 and dS["2008-26"] >= 0.03 else "REJECT")
