"""diagnosis only (no retuning): why the current best rule trails spy in 2017-2026. french market data, same rule as preholiday.py."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
nxt = idx[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(idx[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=idx).astype(float)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
w = (ens.shift(1).fillna(0) * (1 + winter) * (1 + tom_days(idx, 1, 3)) * (1 + ph)).clip(upper=3)
r, pos = run(w, d, lag=0, borrow_spread=0.01)
x = pd.DataFrame({"rule": r, "mkt": d.r, "pos": w, "on": ens.shift(1)}).loc["2008":]
y = x.groupby(x.index.year).agg(rule=("rule", lambda s: (1 + s).prod() - 1), mkt=("mkt", lambda s: (1 + s).prod() - 1), avg_pos=("pos", "mean"), trend_on=("on", "mean"))
y["gap"] = y.rule - y.mkt
print(y.to_string(formatters={c: "{:+.1%}".format for c in ["rule", "mkt", "gap"]} | {c: "{:.2f}".format for c in ["avg_pos", "trend_on"]}))

# split the 2017-26 gap: days trend fully off (in cash) vs days in market
z = x.loc["2017":]
off = z.on == 0
print(f"\n2017-26 days with trend fully off: {off.mean():.0%}. market return on those days, compounded: {(1 + z.mkt[off]).prod() - 1:+.1%}")
print(f"market return on days trend on, compounded: {(1 + z.mkt[~off]).prod() - 1:+.1%}; rule on those days: {(1 + z.rule[~off]).prod() - 1:+.1%}")
# each exit episode: market move from exit to re-entry (positive = whipsaw, missed a rise)
st = ens.loc["2008":].round(3); ep, cur = [], None
for t, v in st.items():
    if v == 0 and cur is None: cur = t
    elif v > 0 and cur is not None: ep.append((cur, t, d.px.loc[t] / d.px.loc[cur] - 1)); cur = None
print("\nfull-exit episodes since 2008 (exit date, re-entry date, market move while out):")
for a, b, m in ep: print(f"  {a.date()} -> {b.date()}  {m:+.1%}")
