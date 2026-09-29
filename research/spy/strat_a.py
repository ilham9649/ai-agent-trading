"""hypothesis a: vrp + vix term structure regime filter with vol targeting."""
import numpy as np, pandas as pd
from engine import *


def weights(d, vol_win=21, sig=0.10, ts_thr=1.0, floor=0.06, use_vrp=True, use_ts=True, cap=1.0):
    rv = d.r.pow(2).rolling(vol_win).mean().mul(TD).pow(0.5)
    vrp = d.vix / 100 - rv
    on = pd.Series(True, index=d.index)
    if use_ts:
        on &= (d.vix / d.vix3m) < ts_thr
    if use_vrp:
        on &= vrp > 0
    w = on * (sig / rv.clip(lower=floor)).clip(upper=cap)
    return w.where(rv.notna(), 0.0)


if __name__ == "__main__":
    d = load()
    rows = [report("spy b&h", d.r.iloc[1:], d)]
    for name, kw in {
        "a full (ts+vrp+vt10)": {},
        "a ts only": dict(use_vrp=False, sig=9),
        "a vrp only": dict(use_ts=False, sig=9),
        "a full, no vt": dict(sig=9),
        "a vt only": dict(use_ts=False, use_vrp=False),
    }.items():
        r, _ = run(weights(d, **kw), d)
        rows.append(report(name, r, d))
    print(table(rows))
