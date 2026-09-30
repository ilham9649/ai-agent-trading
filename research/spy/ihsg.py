"""rules designed on us data, applied unchanged to IHSG (jakarta composite, yfinance ^JKSE, price index in IDR, 1990+).
IHSG was not in the earlier 10-country test (intl.py), so this is out-of-sample by market.
cash = indonesian call-money rate (fred IRSTCI01IDM156N, monthly, prior month's value); borrow = that + 1%; 10 bp costs.
turn-of-month and pre-holiday flags come from IHSG's own trading days. dividends (~2-3%/yr) are missing from index and strategies alike.
rules (frozen, no retuning): trend only (band ensemble 150/200/250 +-3%, max 1x); E (ens x tom x pre-holiday, cap 1.5);
C (E + 1.5x nov-apr / 1x may-oct, cap 2)."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred
from strat_bc import rule_c, rule_e

px = yf.download("^JKSE", period="max", progress=False, auto_adjust=True)["Close"].squeeze()
px.index = px.index.tz_localize(None); px = px[px > 0].dropna()
rate = _fred("IRSTCI01IDM156N"); rate.index = rate.index.to_period("M"); rate = rate.shift(1) / 100
d = pd.DataFrame({"px": px})
d["r"] = d.px.pct_change()
d["rf"] = (1 + rate.reindex(d.index.to_period("M")).values) ** (1 / TD) - 1
d = d.dropna()
assert d.r.abs().max() < 0.25 and len(d) > 8000
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
cfg = {"IHSG buy & hold": pd.Series(1.0, index=idx).shift(1).fillna(0),
       "trend only (1x)": ens.shift(1).fillna(0),
       "rule E (cap 1.5)": rule_e(ens, idx),
       "rule C (cap 2)": rule_c(ens, idx)}
R = {k: run(w, d, lag=0, cost_bp=10, borrow_spread=0.01)[0] for k, w in cfg.items()}
P = {"1991-2007": ("1991-04-01", "2007-12-31"), "2008-26": ("2008-01-01", None), "all 1991-2026": ("1991-04-01", None)}
for e, (a, b) in P.items():
    print(f"=== {e} ===  (avg cash rate {d.rf.loc[a:b].mean() * TD:.1%})")
    for k, rr in R.items():
        s = stats(rr.loc[a:b], d.rf); print(f"  {k:18s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
# calendar effects on IHSG's own data (raw mean daily return, bp)
from strat_bc import tom_days, preholiday_days
tom, ph = tom_days(idx, 1, 3), preholiday_days(idx)
win = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx)
for e, (a, b) in list(P.items())[:2]:
    x = d.r.loc[a:b]
    print(f"{e}: tom {x[tom.loc[a:b] == 1].mean() * 1e4:.1f} vs other {x[tom.loc[a:b] == 0].mean() * 1e4:.1f} bp | pre-holiday {x[ph.loc[a:b] == 1].mean() * 1e4:.1f} bp "
          f"| winter {x[win.loc[a:b]].mean() * 1e4:.1f} vs summer {x[~win.loc[a:b]].mean() * 1e4:.1f} bp")
