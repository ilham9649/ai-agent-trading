"""robustness of trend-filtered 2x spy: sma length, execution lag, costs, borrow spread."""
import pandas as pd
from engine import *
d = load("1993-02-01"); S = "1994-03-01"
spy = stats(d.r.loc[S:], d.rf)
print(f"spy cagr {spy['cagr']:.1%} sharpe {spy['sharpe']:.2f} maxdd {spy['maxdd']:.1%}")
for n in [100, 150, 200, 250]:
    for lag, c, bs in [(1, 2, 0.01), (2, 2, 0.01), (1, 5, 0.01), (1, 2, 0.02), (2, 5, 0.02)]:
        w = 2.0 * (d.px > d.px.rolling(n).mean())
        r, _ = run(w, d, lag=lag, cost_bp=c, borrow_spread=bs); s = stats(r.loc[S:], d.rf)
        print(f"sma{n} lag{lag} cost{c}bp borrow+{bs:.0%}: cagr {s['cagr']:.1%} sharpe {s['sharpe']:.2f} maxdd {s['maxdd']:.1%}")
w = 2.0 * (d.px > d.px.rolling(200).mean()); r, pos = run(w, d)
print("trades/yr", (pos.diff().abs() > 0).sum() / (len(pos) / TD), " time invested", pos.gt(0).mean())
print(report("trend200 2x", r.loc[S:], d.loc[S:], d.r.loc[S:]))
