"""vol-managed spy with regime filters (a extended): grid over sigma target, leverage cap, filters, vol window."""
import itertools
import numpy as np, pandas as pd
from engine import *


def vm_weights(d, sig, cap, filt, vol_win, floor=0.05):
    rv = d.r.pow(2).ewm(halflife=vol_win / 2.0).mean().mul(TD).pow(0.5) if vol_win < 0 else d.r.pow(2).rolling(vol_win).mean().mul(TD).pow(0.5)
    on = pd.Series(True, index=d.index)
    if "trend" in filt:
        on &= d.px > d.px.rolling(200).mean()
    if "vrp" in filt:
        on &= (d.vix / 100 - rv) > 0
    w = on * (sig / rv.clip(lower=floor)).clip(upper=cap)
    return w.where(rv.notna() & d.px.rolling(200).mean().notna(), 0.0)


GRID = list(itertools.product([0.10, 0.12, 0.15, 0.18], [1.0, 1.5, 2.0], ["none", "trend", "vrp", "trend+vrp"], [21, 63]))

if __name__ == "__main__":
    d = load("1993-02-01").loc["1993-02-01":]
    S = "1994-03-01"
    res = {}
    for g in GRID:
        r, pos = run(vm_weights(d, *g), d)
        res[g] = r.loc[S:]
    spy = d.r.loc[S:]
    rows = []
    for g, r in res.items():
        a = stats(r, d.rf); h1 = stats(r.loc[:"2009-12-31"], d.rf); h2 = stats(r.loc["2010-01-01":], d.rf)
        rows.append(dict(g=str(g), cagr=a["cagr"], sharpe=a["sharpe"], maxdd=a["maxdd"], sh_h1=h1["sharpe"], sh_h2=h2["sharpe"]))
    df = pd.DataFrame(rows).set_index("g")
    b = stats(spy, d.rf); b1 = stats(spy.loc[:"2009-12-31"], d.rf); b2 = stats(spy.loc["2010-01-01":], d.rf)
    print(f"spy: cagr {b['cagr']:.1%} sharpe {b['sharpe']:.2f} maxdd {b['maxdd']:.1%} h1 {b1['sharpe']:.2f} h2 {b2['sharpe']:.2f}")
    print(df.sort_values("sharpe", ascending=False).head(12).round(3).to_string())
    print("median sharpe", df.sharpe.median().round(3), " share of grid beating spy sharpe in both halves:",
          ((df.sh_h1 > b1["sharpe"]) & (df.sh_h2 > b2["sharpe"])).mean().round(2))
    print(df.groupby(df.index.str.contains("trend")).sharpe.median())
