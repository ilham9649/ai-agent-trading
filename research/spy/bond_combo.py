"""second return source: banded trend on synthetic 10y treasury total return (from dgs10, 1962+), combined with the equity rule."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
from pooled import pooled, signals

d = pooled()
d = d.loc["1962-01-03":].copy()
y = (_fred("DGS10").reindex(d.index).ffill() / 100)
dmod = (1 - (1 + y / 2) ** -20) / y                       # modified duration of a 10y par bond
d["br"] = y.shift(1) / TD - dmod.shift(1) * y.diff()      # carry + price change (convexity ignored)
d = d.dropna()
d["bpx"] = (1 + d.br).cumprod()
bd = d.assign(r=d.br)                                     # frame whose "r" is the bond return, for run()
S = signals(d.px)
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
bens = (band(d.bpx, 150, .03) + band(d.bpx, 200, .03) + band(d.bpx, 250, .03)) / 3

eq, _ = run(ens * (1 + winter), d, borrow_spread=0.01)
sp = d.r.iloc[1:]
bh_bond = d.br.iloc[1:]
rows = {}
for L in [1.0, 1.5, 2.0]:
    rows[f"bond trend {L}x"], _ = run(L * bens, bd, borrow_spread=0.01)
ERAS = {"1963-93": ("1963-01-01", "1993-12-31"), "1994-26": ("1994-02-01", None), "all": ("1963-01-01", None)}


def line(name, r):
    out = f"{name:34s}"
    for e, (a, b) in ERAS.items():
        s = stats(r.loc[a:b], d.rf)
        out += f" | {e}: {s['cagr']:5.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}"
    print(out)


line("spy b&h", sp); line("10y bond b&h (synthetic)", bh_bond); line("equity rule (ens, 2x/1x season)", eq)
for k, v in rows.items(): line(k, v)
print("corr eq-rule vs bond-trend(1x) daily: %.2f" % eq.corr(rows["bond trend 1.0x"]))
b1 = rows["bond trend 1.5x"]
for a in [0.8, 0.7, 0.6, 0.5]:
    for bk, br in [("1x", rows["bond trend 1.0x"]), ("1.5x", b1)]:
        line(f"{a:.0%} eq rule + {1 - a:.0%} bond trend {bk}", a * eq + (1 - a) * br)
# portfolio leverage on the best-looking mix
mix = 0.7 * eq + 0.3 * rows["bond trend 1.5x"]
for k in [1.25, 1.5]:
    line(f"70/30 mix (bond 1.5x) x{k} levered", k * mix - (k - 1) * (d.rf.loc[mix.index] + 0.01 / TD))
