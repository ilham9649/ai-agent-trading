"""beat banded 2x trend: (1) ensemble over sma lengths, (2) hold ief/gld/tlt instead of cash when trend is off."""
import numpy as np, pandas as pd
from engine import *
from engine import _yf


def run_off(w, off, d, lag=1, cost_bp=2.0, borrow=0.01):
    """spy weight w (0..L); when trend off, hold off-asset return `off` (daily pct) instead of cash. off-asset costs the same."""
    pos = w.shift(lag).fillna(0.0)
    offw = (pos == 0).astype(float)
    borrow_cost = (pos - 1).clip(lower=0) * (d.rf + borrow / TD) + (1 - pos).clip(0, 1) * (1 - offw) * d.rf
    r = pos * d.r + offw * off.fillna(0.0) + (1 - pos).clip(0, 1) * (1 - offw) * d.rf - (pos - 1).clip(lower=0) * (d.rf + borrow / TD)
    turn = pos.diff().abs().fillna(0) + offw.diff().abs().fillna(0)
    return (r - turn * cost_bp / 1e4).iloc[1:]


def show(rows, d):
    df = pd.DataFrame(rows).set_index("name")
    f = {c: "{:.1%}".format for c in ["cagr", "maxdd", "h1", "h2"]} | {c: "{:.2f}".format for c in ["sharpe", "sortino"]}
    print(df.to_string(formatters=f))


def row(name, r, d, S):
    r = r.loc[S:]; a = stats(r, d.rf)
    mid = r.index[len(r) // 2]
    return dict(name=name, cagr=a["cagr"], sharpe=a["sharpe"], sortino=a["sortino"], maxdd=a["maxdd"], h1=stats(r.loc[:mid], d.rf)["cagr"], h2=stats(r.loc[mid:], d.rf)["cagr"])


if __name__ == "__main__":
    d = load("1993-02-01"); S = "1994-03-01"
    px = d.px
    b200 = band(px, 200, .03)
    ens = (band(px, 150, .03) + band(px, 200, .03) + band(px, 250, .03)) / 3
    rows = [row("spy b&h", d.r, d, S)]
    for name, w in [("banded 200 3% 2x", 2 * b200), ("ensemble 150/200/250 3% 2x", 2 * ens), ("ensemble 2.5x", 2.5 * ens), ("ensemble 1.5x", 1.5 * ens)]:
        rows.append(row(name, run(w, d, borrow_spread=0.01)[0], d, S))
    print("1994+"); show(rows, d)

    d2 = load("2004-12-01"); S2 = "2005-12-01"
    px2 = d2.px
    b = band(px2, 200, .03)
    ens2 = (band(px2, 150, .03) + band(px2, 200, .03) + band(px2, 250, .03)) / 3
    ret = lambda k: _yf(k).reindex(d2.index).ffill().pct_change()
    rows = [row("spy b&h", d2.r, d2, S2), row("banded 200 3% 2x, cash off", run(2 * b, d2, borrow_spread=.01)[0], d2, S2)]
    for k in ["ief", "tlt", "gld"]:
        rows.append(row(f"banded 2x, {k} when off", run_off(2 * b, ret(k), d2), d2, S2))
        rows.append(row(f"ensemble 2x, {k} when off", run_off(2 * ens2, ret(k), d2), d2, S2))
    print("\n2005+"); show(rows, d2)
