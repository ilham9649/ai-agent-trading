"""multi-asset trend/vol-parity (spy, ief, tlt, gld) and overnight-only spy."""
import numpy as np, pandas as pd
from engine import *
from engine import _yf


def assets(start="2004-12-01"):
    d = load(start)
    px = pd.DataFrame({k: _yf(k.lower()).reindex(d.index).ffill() for k in ["SPY", "IEF", "TLT", "GLD"]})
    return d, px


def multi_weights(px, d, sma=200, vol_win=63, sig=0.12, cap=1.5, names=("SPY", "IEF", "TLT", "GLD")):
    r = px[list(names)].pct_change()
    trend = (px[list(names)] > px[list(names)].rolling(sma).mean()).astype(float)
    iv = 1 / r.rolling(vol_win).std().mul(np.sqrt(TD))
    raw = trend * iv                                    # inverse-vol among assets in uptrend
    raw = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)  # weights sum to 1 when any asset on
    # portfolio vol estimate assuming avg pairwise corr 0.3 is overkill: use realized vol of the constructed book
    book = (raw.shift(1) * r).sum(axis=1)
    bv = book.rolling(vol_win).std().mul(np.sqrt(TD))
    scale = (sig / bv).clip(upper=cap).fillna(0)
    return raw.mul(scale, axis=0).where(trend.notna().all(axis=1), 0.0)


def run_multi(w, px, d, lag=1, cost_bp=2.0, borrow_spread=0.005):
    pos = w.shift(lag).fillna(0.0)
    r = px.pct_change()
    tot = pos.sum(axis=1)
    ret = (pos * r).sum(axis=1) + (1 - tot).clip(lower=0) * d.rf + (1 - tot).clip(upper=0) * (d.rf + borrow_spread / TD)
    ret -= pos.diff().abs().sum(axis=1).fillna(0) * cost_bp / 1e4
    return ret.iloc[1:]


if __name__ == "__main__":
    d, px = assets()
    b = d.r.iloc[1:]
    rows = [report("spy b&h", b, d)]
    for name, kw in {
        "multi trend ivol vt10 cap1.5": dict(sig=0.10),
        "multi trend ivol vt12 cap1.5": dict(sig=0.12),
        "multi trend ivol vt15 cap2": dict(sig=0.15, cap=2.0),
        "multi trend ivol vt12 cap1.5 sma150": dict(sma=150),
        "multi spy+ief+gld vt12": dict(names=("SPY", "IEF", "GLD")),
    }.items():
        rows.append(report(name, run_multi(multi_weights(px, d, **kw), px, d), d))
    print(table(rows))
    d2 = load("1993-02-01")
    on = d2.op / d2.px.shift(1) - 1
    idn = d2.px / d2.op - 1
    print("\novernight vs intraday (1993+): overnight cagr %.1f%% intraday cagr %.1f%%" % (((1 + on.dropna()).prod() ** (TD / len(on.dropna())) - 1) * 100, ((1 + idn).prod() ** (TD / len(idn)) - 1) * 100))
    rows = [report("spy b&h", d2.r.iloc[1:], d2)]
    for c in [0.0, 0.5, 1.0, 2.0]:
        r = on.fillna(0) * 1.0 - c / 1e4 * 2  # round trip each night
        rows.append(report(f"overnight only, {c}bp/side", r.iloc[1:], d2))
    print(table(rows))
