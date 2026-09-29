"""market breadth as a 4th trend vote. breadth = share of the 49 french industries above their own 200-day sma.
frozen before running, no retuning after:
  vote = on when breadth > 0.6, off when breadth < 0.4, else unchanged
  ens4 = (band150 + band200 + band250 + vote) / 4, then the current best (season x tom x pre-holiday, cap 3x)
adopt only if sharpe beats current best in BOTH 2008-2026 and the 1927-1962 holdout."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
above = (lvl > lvl.rolling(200).mean()).astype(float).where(lvl.rolling(200).mean().notna())
breadth = above.mean(axis=1)
vote, on = [], 0.0
for b in breadth.values:
    if b > 0.6: on = 1.0
    elif b < 0.4: on = 0.0
    vote.append(on)
vote = pd.Series(vote, index=d.index)
print(f"breadth vote on: {vote.mean():.0%} of days; agrees with band200 {(vote == band(d.px, 200, .03)).mean():.0%}")

bands = band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)
nxt = d.index[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(d.index[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=d.index).astype(float)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
cal = (1 + tom_days(d.index, 1, 3)) * (1 + ph)
sh = lambda w: w.shift(1).fillna(0)
cfg = {"market b&h": sh(pd.Series(1.0, index=d.index)),
       "current best": (sh(bands / 3 * (1 + winter)) * cal).clip(upper=3),
       "current, breadth 4th vote": (sh((bands + vote) / 4 * (1 + winter)) * cal).clip(upper=3)}
R = {k: run(w, d, lag=0, borrow_spread=0.01)[0] for k, w in cfg.items()}
E = {"1927-62 holdout": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
for e, (a, b) in E.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:28s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["current, breadth 4th vote"].loc[a:b], R["current best"].loc[a:b], d.rf, n=1000)
    print(f"  breadth vs current: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
sh_ = lambda k, e: stats(R[k].loc[E[e][0]:E[e][1]], d.rf)["sharpe"]
print("ADOPT" if all(sh_("current, breadth 4th vote", e) > sh_("current best", e) for e in ["1927-62 holdout", "2008-26"]) else "REJECT")
