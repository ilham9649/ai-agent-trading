"""valuation (shiller cape) as a leverage scale. run: uv run --with xlrd python valuation.py (data/ie_data.xls from shiller's site, ends 2023-09).
frozen before running, no retuning after:
  p = expanding percentile of cape since 1881, known with a 1-month lag
  position x0.75 when p > 0.8, x1.25 when p < 0.2, else x1; on top of the current best (ens x season x tom x pre-holiday), cap 3x
adopt only if sharpe beats current best in BOTH 2008-2023 and the 1927-1962 holdout."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

x = pd.read_excel(D / "ie_data.xls", sheet_name="Data", header=7)[["Date", "CAPE"]].dropna()
x = x[pd.to_numeric(x.CAPE, errors="coerce").notna()]
x.index = pd.PeriodIndex([f"{int(v)}-{round(v % 1 * 100):02d}" for v in x.Date], freq="M")
cape = x.CAPE.astype(float)
pct = cape.expanding(120).apply(lambda s: (s[:-1] < s[-1]).mean(), raw=True)
mult_m = pd.Series(1.0, index=pct.index).mask(pct > 0.8, 0.75).mask(pct < 0.2, 1.25).shift(1)   # 1-month publication lag
assert x.index[0] == pd.Period("1881-01", "M") and cape.loc["2000-03"] > 40

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]})).loc[:"2023-09-30"]
d["px"] = (1 + d.r).cumprod()
idx = d.index
mult = mult_m.reindex(idx.to_period("M")).set_axis(idx)
mult = mult.shift(1).fillna(1.0)                      # month m value first used on the first day of month m (+ publication lag above)
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
nxt = idx[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(idx[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=idx).astype(float)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
cur = ens.shift(1).fillna(0) * (1 + winter) * (1 + tom_days(idx, 1, 3)) * (1 + ph)
for e, (a, b) in {"1927-62": ("1927", "1962"), "1963-2007": ("1963", "2007"), "2008-23": ("2008", None)}.items():
    m = mult.loc[a:b]; print(f"{e}: share of days x0.75 {(m == .75).mean():.0%}, x1.25 {(m == 1.25).mean():.0%}")
cfg = {"market b&h": pd.Series(1.0, index=idx).shift(1).fillna(0), "current best": cur.clip(upper=3), "current x valuation": (cur * mult).clip(upper=3)}
R = {k: run(w, d, lag=0, borrow_spread=0.01)[0] for k, w in cfg.items()}
E = {"1927-62 holdout": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-23": ("2008-01-01", None)}
for e, (a, b) in E.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:22s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["current x valuation"].loc[a:b], R["current best"].loc[a:b], d.rf, n=1000)
    print(f"  valuation vs current: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
sh_ = lambda k, e: stats(R[k].loc[E[e][0]:E[e][1]], d.rf)["sharpe"]
print("ADOPT" if all(sh_("current x valuation", e) > sh_("current best", e) for e in ["1927-62 holdout", "2008-23"]) else "REJECT")
# result note: the frozen check printed ADOPT on a near tie (dSharpe +0.01 holdout, +0.02 [-0.05,+0.09] 2008-23).
# after 2008 the layer is x0.75 on 84% of days, so it acts as lower leverage (cagr 15.9% -> 14.3%), not a new edge. not adopted.
