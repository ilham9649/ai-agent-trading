"""upgrades to trend200 2x: vol-scaled leverage, hysteresis band, graded leverage, vrp filter. borrow rf+1%, 2bp."""
import numpy as np, pandas as pd
from engine import *

d = load("1993-02-01"); S = "1994-03-01"
sma = lambda n: d.px.rolling(n).mean()
rv = d.r.pow(2).rolling(63).mean().mul(TD).pow(0.5)
rv21 = d.r.pow(2).rolling(21).mean().mul(TD).pow(0.5)


def band(n, up, dn):
    """stateful: on when px > sma*(1+up), off when px < sma*(1-dn)."""
    s = sma(n); on, out = 0.0, []
    for p, m in zip(d.px.values, s.values):
        if np.isnan(m): out.append(0.0); continue
        if p > m * (1 + up): on = 1.0
        elif p < m * (1 - dn): on = 0.0
        out.append(on)
    return pd.Series(out, index=d.index)


t200 = (d.px > sma(200)).astype(float)
cfg = {
    "spy b&h": pd.Series(1.0, index=d.index),
    "base trend200 2x": 2 * t200,
    "trend200 x vol-scaled (0.20/rv, cap2)": t200 * (0.20 / rv.clip(lower=0.05)).clip(upper=2),
    "trend200 x vol-scaled (0.25/rv, cap2)": t200 * (0.25 / rv.clip(lower=0.05)).clip(upper=2),
    "trend200 band +-1% 2x": 2 * band(200, .01, .01),
    "trend200 band +-2% 2x": 2 * band(200, .02, .02),
    "graded: 2x if trend & rv21<16%, 1x if trend, else 0": t200 * np.where(rv21 < 0.16, 2.0, 1.0),
    "graded: 2x if trend & vix<25, 1x if trend": t200 * np.where(d.vix < 25, 2.0, 1.0),
    "trend200 2x + vrp>0 filter": 2 * t200 * ((d.vix / 100 - rv21) > 0),
    "trend200&250 both 2x": 2 * t200 * (d.px > sma(250)),
    "trend 2x, 3x if px>sma*1.05? no: 1.5x above 200 and 250 avg": t200 * (1.0 + 0.5 * (d.px > sma(250))) * 1.333,
}
rows = []
for name, w in cfg.items():
    r, pos = run(w, d, borrow_spread=0.01)
    r = r.loc[S:]
    a, h1, h2 = stats(r, d.rf), stats(r.loc[:"2009-12-31"], d.rf), stats(r.loc["2010-01-01":], d.rf)
    rows.append(dict(name=name[:52], cagr=a["cagr"], sharpe=a["sharpe"], sortino=a["sortino"], maxdd=a["maxdd"], cagr_h1=h1["cagr"], cagr_h2=h2["cagr"], sh_h1=h1["sharpe"], sh_h2=h2["sharpe"], tr_yr=(pos.diff().abs() > 0.01).sum() / (len(pos) / TD)))
df = pd.DataFrame(rows).set_index("name")
f = {c: "{:.1%}".format for c in ["cagr", "maxdd", "cagr_h1", "cagr_h2"]} | {c: "{:.2f}".format for c in ["sharpe", "sortino", "sh_h1", "sh_h2", "tr_yr"]}
print(df.to_string(formatters=f))
