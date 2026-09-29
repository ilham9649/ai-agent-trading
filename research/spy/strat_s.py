"""sector-etf cross-sectional momentum (monthly), with optional absolute-momentum cash filter."""
import numpy as np, pandas as pd, yfinance as yf, pathlib
from engine import *

SEC = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
p = D / "sectors.csv"
if not p.exists():
    x = yf.download(SEC, start="1998-12-01", auto_adjust=True, progress=False)["Close"]
    x.to_csv(p)
px_all = pd.read_csv(p, index_col=0, parse_dates=True)


def sector_weights(d, px, top=3, lb=252, skip=21, abs_filter=True, vol_scale=False):
    mom = px.shift(skip) / px.shift(lb) - 1
    rank = mom.rank(axis=1, ascending=False)
    sel = (rank <= top).astype(float)
    if abs_filter:
        sel = sel * (mom > 0)
    w = sel.div(top)
    me = w.index.to_series().groupby(w.index.to_period("M")).transform("max") == w.index.to_series()  # month-end rebalance days
    w = w.where(me).ffill().fillna(0.0)
    return w


def run_multi(w, px, d, lag=1, cost_bp=2.0):
    pos = w.shift(lag).fillna(0.0)
    r = px.pct_change()
    tot = pos.sum(axis=1)
    ret = (pos * r).sum(axis=1).fillna(0) + (1 - tot) * d.rf - pos.diff().abs().sum(axis=1).fillna(0) * cost_bp / 1e4
    return ret.iloc[1:]


if __name__ == "__main__":
    d = load("1999-01-01")
    px = px_all.reindex(d.index).ffill()
    S = "2000-06-01"
    d = d.loc[S:] if False else d
    rows = [report("spy b&h", d.r.loc[S:], d.loc[S:], d.r.loc[S:])]
    for name, kw in {"top3 12-1 abs": {}, "top3 12-1 no abs": dict(abs_filter=False), "top3 6-1 abs": dict(lb=126), "top2 12-1 abs": dict(top=2),
                     "top4 12-1 abs": dict(top=4), "top3 3-1 abs": dict(lb=63)}.items():
        r = run_multi(sector_weights(d, px, **kw), px, d).loc[S:]
        rows.append(report(name, r, d.loc[S:], d.r.loc[S:]))
    print(table(rows))
