"""monthly options-expiration week (stivers & sun 2013): the mon-fri week that holds the 3rd friday of the month.
frozen before running, no retuning after:
  position x1.5 on opex-week days, on top of the current best (ens x season x tom x pre-holiday), cap 3x
adopt only if sharpe beats current best in BOTH 2008-2026 and the 1983-2007 holdout (index options listed 1983).
1927-72 has no listed options: a placebo, the effect should be absent there."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
m0 = idx.to_period("M").to_timestamp()
third_fri = m0 + pd.to_timedelta((4 - m0.dayofweek) % 7 + 14, unit="D")
opex = pd.Series(((third_fri - idx).days >= 0) & ((third_fri - idx).days <= 4), index=idx).astype(float)
assert opex.loc["2025-09-15":"2025-09-19"].sum() == 5 and opex.loc["2025-09-22":"2025-09-26"].sum() == 0   # 3rd fri = 2025-09-19
assert opex.loc["2025-08-11":"2025-08-15"].sum() == 5                                                        # month starting on friday

E = {"1927-72 placebo": ("1927-07-01", "1972-12-31"), "1983-2007 holdout": ("1983-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
print("raw mean daily return (bp): opex week vs other days")
for e, (a, b) in E.items():
    r, f = d.r.loc[a:b], opex.loc[a:b]
    print(f"  {e}: opex {r[f == 1].mean()*1e4:5.1f}  other {r[f == 0].mean()*1e4:5.1f}")

ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
nxt = idx[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(idx[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=idx).astype(float)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
cur = ens.shift(1).fillna(0) * (1 + winter) * (1 + tom_days(idx, 1, 3)) * (1 + ph)
cfg = {"market b&h": pd.Series(1.0, index=idx).shift(1).fillna(0), "current best": cur.clip(upper=3), "current x opex 1.5": (cur * (1 + 0.5 * opex)).clip(upper=3)}
R = {k: run(w, d, lag=0, borrow_spread=0.01)[0] for k, w in cfg.items()}
for e, (a, b) in E.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:22s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["current x opex 1.5"].loc[a:b], R["current best"].loc[a:b], d.rf, n=1000)
    print(f"  opex vs current: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
sh_ = lambda k, e: stats(R[k].loc[E[e][0]:E[e][1]], d.rf)["sharpe"]
print("ADOPT" if all(sh_("current x opex 1.5", e) > sh_("current best", e) for e in ["1983-2007 holdout", "2008-26"]) else "REJECT")
# result note: the frozen check printed ADOPT on a tie (2008-26 sharpe 0.77 vs 0.76, dSharpe +0.00 [-0.07,+0.07]).
# raw opex-week return is below other days in 2008-26 (2.5 vs 5.9 bp), so the gain is leverage, not edge. not adopted.
