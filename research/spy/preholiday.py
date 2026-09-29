"""pre-holiday effect (ariel 1990): the last trading day before a weekday market closure.
frozen before running: layer = position x2 on pre-holiday days, on top of the current best rule, cap 3x.
adopt only if sharpe beats current best in BOTH 2008-2026 and 1927-55. french market 1927-2026, saturdays merged.
flag = next trading day is more than 1 business day away. this also flags a few unscheduled closures (1933, 1968, 2001, 2012): small lookahead."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days
from industry import merge_sat

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
nxt = d.index[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(d.index[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=d.index).astype(float)
assert ph.loc["2025-12-24"] == 1 and ph.loc["2025-12-23"] == 0   # christmas eve 2025 is pre-holiday

E = {"1927-55": ("1927-07-01", "1955-12-31"), "1956-93": ("1956-01-01", "1993-12-31"), "1994-07": ("1994-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
print("raw effect, mean daily return (bp): pre-holiday vs other days, and days/yr")
for e, (a, b) in E.items():
    r, f = d.r.loc[a:b], ph.loc[a:b]
    print(f"  {e}: pre-hol {r[f == 1].mean()*1e4:6.1f}  other {r[f == 0].mean()*1e4:5.1f}  n/yr {f.sum()/(len(f)/TD):.1f}")

ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
tom = tom_days(d.index, 1, 3)
sh = lambda w: w.shift(1).fillna(0)
cfg = {
    "market b&h": sh(pd.Series(1.0, index=d.index)),
    "current best (max 3x)": (sh(ens * (1 + winter)) * (1 + tom)).clip(upper=3),
    "current + pre-holiday x2": (sh(ens * (1 + winter)) * (1 + tom) * (1 + ph)).clip(upper=3),
    "max 2x + pre-holiday x2": (sh(ens * (1 + winter)) * (1 + ph)).clip(upper=3),
}
rows = []
for name, w in cfg.items():
    r, _ = run(w, d, lag=0, borrow_spread=0.01)
    row = {"name": name}
    for e, (a, b) in E.items():
        s = stats(r.loc[a:b], d.rf); row |= {f"{e} cagr": s["cagr"], f"{e} sh": s["sharpe"], f"{e} dd": s["maxdd"]}
    rows.append(row)
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250)
print(df.to_string(formatters={c: ("{:.2f}" if c.endswith("sh") else "{:.1%}").format for c in df}))
c, n = df.loc["current best (max 3x)"], df.loc["current + pre-holiday x2"]
print("ADOPT" if n["2008-26 sh"] > c["2008-26 sh"] and n["1927-55 sh"] > c["1927-55 sh"] else "REJECT")

# after the frozen verdict: significance of the added layer, and cost stress
rc, _ = run(cfg["current best (max 3x)"], d, lag=0, borrow_spread=0.01)
for e, (a, b) in {"1927-55": E["1927-55"], "2008-26": E["2008-26"], "all": ("1927-07-01", None)}.items():
    rn, _ = run(cfg["current + pre-holiday x2"], d, lag=0, borrow_spread=0.01)
    pt, lo, hi, p = sharpe_diff_ci(rn.loc[a:b], rc.loc[a:b], d.rf, n=1000)
    print(f"pre-holiday layer vs current, {e}: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
for bp, bs in [(5, 0.02), (10, 0.03)]:
    s = stats(run(cfg["current + pre-holiday x2"], d, lag=0, cost_bp=bp, borrow_spread=bs)[0].loc["2008":], d.rf)
    print(f"stress {bp}bp borrow+{bs:.0%}, 2008-26: sharpe {s['sharpe']:.2f}")
