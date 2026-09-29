"""corrected timing: trend/season weight uses lag 1, calendar boosts (tom, fomc) are known in advance so apply on the day itself."""
import numpy as np, pandas as pd
from engine import *
from pooled import pooled, signals, ERAS
from strat_bc import tom_days, fomc_days

d = pooled(); S = signals(d.px)
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
base = ens * (1 + winter)
fomc = fomc_days(d.index)


def build(tb=2.0, cap=3.0, nl=1, nf=3, fb=1.0):
    pos = base.shift(1).fillna(0.0)                       # decided at prior close
    cal = 1 + (tb - 1) * tom_days(d.index, nl, nf) + (fb - 1) * fomc
    return (pos * cal).clip(upper=cap)


def line(name, w):
    r, pos = run(w, d, lag=0, borrow_spread=0.01)
    o = f"{name:40s}"
    for e, (a, b) in {**ERAS, "all": ("1956-01-01", None)}.items():
        s = stats(r.loc[a:b], d.rf); o += f" | {e}: {s['cagr']:5.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}"
    print(o); return r, pos


line("spy b&h", pd.Series(1.0, index=d.index).shift(1).fillna(0))
line("base (no calendar boost)", build(tb=1.0))
for tb, cap, nl, nf in [(1.5, 3, 1, 3), (2, 3, 1, 3), (2, 3, 1, 2), (2, 3, 2, 3), (2, 3, 1, 4), (2, 2.5, 1, 3), (2.5, 3, 1, 3)]:
    line(f"tom x{tb} cap{cap} last{nl}+first{nf}", build(tb, cap, nl, nf))
print("\nfomc-day boost added (tom x2 cap3), fomc data 1994+:")
for fb in [1.0, 1.5, 2.0]:
    line(f"tom x2 + fomc x{fb}", build(fb=fb))
