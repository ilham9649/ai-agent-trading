"""new information on the 71-year pool: halloween seasonality overlay and yield-curve regime, on top of the band ensemble."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
from pooled import pooled, signals, evaluate

d = pooled(); S = signals(d.px)
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
curve = (_fred("DGS10").reindex(d.index).ffill() - _fred("DTB3").reindex(d.index).ffill())
inv = (curve < 0).astype(float)
print("share of days: winter %.0f%%, curve inverted %.0f%%; data starts %s" % (winter.mean() * 100, inv[curve.notna()].mean() * 100, curve.dropna().index.min().date()))
# raw seasonal edge: mean daily excess return winter vs summer per era
ex = d.r - d.rf
for e, (a, b) in {"1956-93": ("1956-01-01", "1993-12-31"), "1994-26": ("1994-02-01", None)}.items():
    x, w = ex.loc[a:b], winter.loc[a:b]
    print(e, "annualised excess: winter %.1f%% summer %.1f%%" % (x[w == 1].mean() * TD * 100, x[w == 0].mean() * TD * 100))
cfg = {
    "spy b&h": pd.Series(1.0, index=d.index),
    "band ensemble 2x (ref)": 2 * ens,
    "ens 2x winter, 1x summer": ens * (1 + winter),
    "ens 2x winter, 0.5x summer": ens * (0.5 + 1.5 * winter),
    "ens 2x winter, 0 summer": 2 * ens * winter,
    "ens 2x, 1x when curve inverted": ens * (2 - inv),
    "ens 2x, 0 when curve inverted": 2 * ens * (1 - inv),
}
evaluate(cfg, d)
