"""short-term mean reversion sleeve (rsi2 dip-buy inside uptrend), alone and as an overlay on the equity rule. pooled 1956-2026."""
import numpy as np, pandas as pd
from engine import *
from pooled import pooled, signals, ERAS

d = pooled(); S = signals(d.px); px = d.px
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
eq_w = ens * (1 + winter)


def rsi(p, n=2):
    ch = p.diff(); up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-ch.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def sleeve(enter=10, exit_=65, trend_n=200):
    r2 = rsi(px).values; ok = (px > px.rolling(trend_n).mean()).values; on, out = 0.0, []
    for x, t in zip(r2, ok):
        if on == 0.0 and x < enter and t: on = 1.0
        elif on == 1.0 and x > exit_: on = 0.0
        out.append(on)
    return pd.Series(out, index=px.index)


def line(name, r):
    o = f"{name:44s}"
    for e, (a, b) in {**ERAS, "all": ("1956-01-01", None)}.items():
        s = stats(r.loc[a:b], d.rf)
        o += f" | {e}: {s['cagr']:5.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}"
    print(o)


for lag in [0, 1]:
    print(f"\n=== execution lag {lag} (0 = decide and trade at same close via MOC; 1 = next close) ===")
    eq, _ = run(eq_w, d, lag=lag, borrow_spread=0.01)
    line("spy b&h", d.r.iloc[1:]); line("equity rule", eq)
    for en, ex in [(10, 65), (5, 65), (15, 70), (10, 50)]:
        w = sleeve(en, ex)
        sl, pos = run(w, d, lag=lag, borrow_spread=0.01)
        line(f"rsi2 sleeve alone <{en} exit>{ex} [{pos.mean():.0%} inv]", sl)
        exc = sl - d.rf.loc[sl.index]                       # return over cash, funded by borrowing at rf
        for m in [1.0, 2.0]:
            line(f"   equity rule + {m:.0f}x sleeve overlay", eq + m * exc - m * pos.diff().abs().fillna(0).loc[eq.index] * 0)
    print("corr(eq rule, rsi2 sleeve excess) %.2f" % eq.corr(exc))
