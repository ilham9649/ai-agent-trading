"""daily equity and drawdown chart: spy vs rule E vs S2 (rule C timing on the low-vol + momentum industry basket), 10 bp.
run: uv run --with matplotlib python plot_daily.py  -> data/daily_chart.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from engine import *
from strat_f import french
from strat_bc import rule_c, rule_e

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var(); lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
m12 = lvl.shift(21) / lvl.shift(252) - 1
sel = m12[me].where(var[me].rank(axis=1, method="first") <= 20).rank(axis=1, ascending=False, method="first") <= 10
hold = sel.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
s2 = (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 10e-4).fillna(0)
S = {"spy (market)": d.r, "rule E": run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)[0],
     "S2": run(rule_c(ens, idx), d.assign(r=s2), lag=0, borrow_spread=0.01)[0]}
a = "2008-01-01"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for k, r in S.items():
    eq = (1 + r.loc[a:]).cumprod(); s = stats(r.loc[a:], d.rf)
    ax1.plot(eq.index, eq, label=f"{k}: {s['cagr']:.1%}/yr, sharpe {s['sharpe']:.2f}, max dd {s['maxdd']:.0%}")
    ax2.plot(eq.index, eq / eq.cummax() - 1)
ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)"); ax1.legend(loc="upper left"); ax1.grid(alpha=.3)
ax2.set_ylabel("drawdown"); ax2.grid(alpha=.3)
ax1.set_title("daily backtest 2008-2026 (french/crsp data incl. delisted stocks; trades at the close, costs included)")
fig.tight_layout(); fig.savefig(D / "daily_chart.png", dpi=110)
print("saved", D / "daily_chart.png")
