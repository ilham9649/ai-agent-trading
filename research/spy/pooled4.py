"""two more designs on the band-ensemble x season base, judged by both eras (lag 1 only): (a) vol-scaled leverage, (b) turn-of-month boost."""
import numpy as np, pandas as pd
from engine import *
from pooled import pooled, signals, evaluate
from strat_bc import tom_days

d = pooled(); S = signals(d.px)
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
base = ens * (1 + winter)
rv = d.r.pow(2).rolling(63).mean().mul(TD).pow(0.5)
tom = tom_days(d.index, 1, 3)
print("avg leverage of base when on: %.2f; rv median %.1f%%" % (base[base > 0].mean(), rv.median() * 100))
cfg = {"spy b&h": pd.Series(1.0, index=d.index), "base: ens x (2 winter/1 summer)": base}
for tgt in [0.15, 0.20, 0.25]:
    for cap in [2.5, 3.0]:
        lev = (tgt / rv.clip(lower=0.05)).clip(upper=cap)
        cfg[f"vol-scaled: ens x season-tilt x lev(tgt {tgt:.0%}, cap {cap})/1.5"] = ens * (1 + winter) * lev / 1.5
for tb in [1.5, 2.0]:
    cfg[f"tom boost x{tb} on base"] = base * (1 + (tb - 1) * tom)
cfg["tom boost x1.5 only in summer"] = base * (1 + 0.5 * tom * (1 - winter))
evaluate(cfg, d)
