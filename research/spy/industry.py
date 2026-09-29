"""new return source: industry momentum on ken french 49 value-weighted industries (no survivorship).
frozen design: month-end rank by 12-1 month return, hold top 10 equal weight, 2bp per unit turnover.
timing overlay = the same rule as holdout_1926 layer 5, signalled from the market, applied to the basket.
design period 1963+, holdout 1927-1962 (run once, no retuning)."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days


ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
raw = ind.join(pd.DataFrame({"mkt": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner")
x = merge_sat(raw)
R, mkt, rf = x[ind.columns], x.mkt, x.rf

# industry momentum basket
lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
me = R.index.to_series().groupby(R.index.to_period("M")).transform("max") == R.index.to_series()
mom = (lvl.shift(21) / lvl.shift(252) - 1)
sel = (mom.rank(axis=1, ascending=False) <= 10).astype(float).div(10)
w = sel.where(me).ffill().fillna(0).shift(1)                         # formed at month-end close, held from next day
basket = (w * R.fillna(0)).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * 2 / 1e4

# market-signalled timing weight (layer 5)
px = (1 + mkt).cumprod()
ens = (band(px, 150, .03) + band(px, 200, .03) + band(px, 250, .03)) / 3
winter = pd.Series(px.index.month.isin([11, 12, 1, 2, 3, 4]), index=px.index).astype(float)
L5 = (ens * (1 + winter)).shift(1).fillna(0) * (1 + tom_days(px.index, 1, 3))
L5 = L5.clip(upper=3.0)
d_m = pd.DataFrame({"r": mkt, "rf": rf}); d_b = pd.DataFrame({"r": basket, "rf": rf})

ser = {"market b&h": mkt, "industry mom basket b&h": basket,
       "market + timing rule": run(L5, d_m, lag=0, borrow_spread=0.01)[0],
       "basket + timing rule": run(L5, d_b, lag=0, borrow_spread=0.01)[0]}
for per, (a, b) in {"design 1963-2026": ("1963-01-01", None), "holdout 1927-1962": ("1927-07-01", "1962-12-31")}.items():
    print(f"\n=== {per} ===")
    for name, r in ser.items():
        r = r.loc[a:b]; s = stats(r, rf)
        print(f"{name:26s} cagr {s['cagr']:6.1%} vol {s['vol']:5.1%} sh {s['sharpe']:.2f} sortino {s['sortino']:.2f} dd {s['maxdd']:6.1%}")
    for base_name, new_name in [("market b&h", "industry mom basket b&h"), ("market + timing rule", "basket + timing rule")]:
        pt, lo, hi, p = sharpe_diff_ci(ser[new_name].loc[a:b], ser[base_name].loc[a:b], rf, n=1000)
        print(f"   {new_name} vs {base_name}: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
r = ser["industry mom basket b&h"] - mkt
print("\nbasket minus market, annualised, by decade:", {k: round(v * TD * 100, 1) for k, v in r.groupby(r.index.year // 10 * 10).mean().items()})

print("\n=== recent 2008-2026 (inside design period, reported for decay) ===")
for name, r in ser.items():
    s = stats(r.loc["2008-01-01":], rf); print(f"{name:26s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
pt, lo, hi, p = sharpe_diff_ci(ser["basket + timing rule"].loc["2008":], ser["market + timing rule"].loc["2008":], rf, n=1000)
print(f"   basket+timing vs market+timing: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
print("avg monthly one-way turnover of basket: %.0f%%" % (w.diff().abs().sum(axis=1).sum() / 2 / (len(w) / 21) * 100))
