"""robustness of hysteresis-band trend: grid of sma x band x leverage, halves, stress on lag/cost/borrow."""
import itertools
import numpy as np, pandas as pd
from engine import *

d = load("1993-02-01"); S = "1994-03-01"


def band(n, w):
    m = d.px.rolling(n).mean().values; on, out = 0.0, []
    for p, mm in zip(d.px.values, m):
        if np.isnan(mm): out.append(0.0); continue
        if p > mm * (1 + w): on = 1.0
        elif p < mm * (1 - w): on = 0.0
        out.append(on)
    return pd.Series(out, index=d.index)


spy = d.r.loc[S:]
sb = stats(spy, d.rf)
print(f"spy cagr {sb['cagr']:.1%} sharpe {sb['sharpe']:.2f} maxdd {sb['maxdd']:.1%}\n")
rows = []
for n, w, L in itertools.product([150, 200, 250], [0.0, 0.01, 0.02, 0.03, 0.04], [1.5, 2.0]):
    r, _ = run(L * band(n, w), d, borrow_spread=0.01); r = r.loc[S:]
    a, h1, h2 = stats(r, d.rf), stats(r.loc[:"2009-12-31"], d.rf), stats(r.loc["2010-01-01":], d.rf)
    rows.append(dict(sma=n, band=w, L=L, cagr=a["cagr"], sharpe=a["sharpe"], maxdd=a["maxdd"], cagr_h1=h1["cagr"], cagr_h2=h2["cagr"]))
g = pd.DataFrame(rows)
f = {c: "{:.1%}".format for c in ["band", "cagr", "maxdd", "cagr_h1", "cagr_h2"]} | {"sharpe": "{:.2f}".format}
print(g[g.L == 2.0].to_string(index=False, formatters=f))
print("\nL=2 grid: cagr>spy in both halves: %.0f%%, sharpe>=spy: %.0f%%, maxdd better than spy: %.0f%%" % (
    ((g.L == 2) & (g.cagr_h1 > 0.075) & (g.cagr_h2 > 0.142)).sum() / (g.L == 2).sum() * 100,
    (g[g.L == 2].sharpe >= sb["sharpe"]).mean() * 100, (g[g.L == 2].maxdd > sb["maxdd"]).mean() * 100))
print("median by band (L=2):"); print(g[g.L == 2].groupby("band")[["cagr", "sharpe", "maxdd"]].median().round(3))

print("\nstress: sma200 band2% L2")
for lag, c, bs in [(1, 2, .01), (2, 2, .01), (1, 5, .01), (1, 2, .02), (2, 5, .02), (1, 10, .03)]:
    r, _ = run(2 * band(200, .02), d, lag=lag, cost_bp=c, borrow_spread=bs); s = stats(r.loc[S:], d.rf)
    print(f"lag{lag} cost{c}bp borrow+{bs:.0%}: cagr {s['cagr']:.1%} sharpe {s['sharpe']:.2f} maxdd {s['maxdd']:.1%}")
# walk-forward: choose (sma, band) by sharpe on 1994-2009, evaluate 2010+
tr = {(n, w): stats(run(2 * band(n, w), d, borrow_spread=.01)[0].loc[S:"2009-12-31"], d.rf)["sharpe"] for n in [150, 200, 250] for w in [0, .01, .02, .03, .04]}
n, w = max(tr, key=tr.get)
r = run(2 * band(n, w), d, borrow_spread=.01)[0].loc["2010-01-01":]
s, b = stats(r, d.rf), stats(d.r.loc["2010-01-01":], d.rf)
print(f"\nwalk-forward pick on 1994-2009: sma{n} band{w:.0%} -> 2010+ cagr {s['cagr']:.1%} sharpe {s['sharpe']:.2f} maxdd {s['maxdd']:.1%} | spy {b['cagr']:.1%} {b['sharpe']:.2f} {b['maxdd']:.1%}")
