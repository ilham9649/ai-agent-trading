"""does trend200 on real 2x etfs (sso, upro) match the modelled 2x spy? signal from spy, trade the etf."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
d = load("2006-07-17")
for tk, start in [("SSO", "2006-08-01"), ("UPRO", "2009-07-01")]:
    e = yf.download(tk, start="2006-01-01", auto_adjust=True, progress=False)["Close"].squeeze().reindex(d.index).ffill()
    dd = d.loc[start:]
    er = e.pct_change().loc[start:]
    on = (d.px > d.px.rolling(200).mean()).astype(float)
    pos = on.shift(1).loc[start:].fillna(0)
    r = (pos * er + (1 - pos) * dd.rf - pos.diff().abs().fillna(0) * 2 / 1e4).iloc[1:]
    m, _ = run(2.0 * on, d.loc[start:].assign(), cost_bp=2.0, borrow_spread=0.01) if False else run((2.0 * on).loc[start:], dd)
    b = dd.r.iloc[1:]
    for name, x in [(f"{tk} trend200 (real)", r), ("model 2x trend200", m), ("spy b&h", b)]:
        s = stats(x, dd.rf)
        print(f"{start[:4]}+ {name:24s} cagr {s['cagr']:6.1%} vol {s['vol']:5.1%} sharpe {s['sharpe']:.2f} maxdd {s['maxdd']:6.1%}")
    eq = (1 + er.dropna()).cumprod(); print(f"   {tk} buy&hold cagr {eq.iloc[-1] ** (TD / len(er.dropna())) - 1:.1%} maxdd {(eq / eq.cummax() - 1).min():.1%}")
