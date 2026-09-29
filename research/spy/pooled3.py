"""plateau + significance for the winter/summer leverage tilt on the band ensemble, pooled 1956-2026."""
import numpy as np, pandas as pd
from engine import *
from pooled import pooled, signals

d = pooled(); S = signals(d.px)
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
P0 = "1956-01-01"
bench = d.r.loc[P0:]
print("winter x summer leverage on ens (pooled 1956-2026; spy cagr %.1f%% sharpe %.2f dd %.1f%%)" % tuple(np.array([stats(bench, d.rf)[k] for k in ["cagr", "sharpe", "maxdd"]]) * [100, 1, 100]))
for wn in [2.0, 2.5]:
    for sm in [0.5, 1.0, 1.5, 2.0]:
        r, _ = run(ens * (sm + (wn - sm) * winter), d, borrow_spread=0.01); r = r.loc[P0:]
        s = stats(r, d.rf)
        print(f"winter {wn}x summer {sm}x: cagr {s['cagr']:.1%} sharpe {s['sharpe']:.2f} sortino {s['sortino']:.2f} maxdd {s['maxdd']:.1%}")
ref, _ = run(2 * ens, d, borrow_spread=0.01); ref = ref.loc[P0:]
for name, w in [("ens 2x (ref)", 2 * ens), ("ens 2x winter/1x summer", ens * (1 + winter)), ("ens 2x winter/1.5x summer", ens * (1.5 + 0.5 * winter))]:
    r, _ = run(w, d, borrow_spread=0.01); r = r.loc[P0:]
    pt, lo, hi, p = sharpe_diff_ci(r, bench, d.rf); a, t, beta = alpha(r, bench, d.rf)
    pt2, lo2, hi2, p2 = sharpe_diff_ci(r, ref, d.rf)
    print(f"{name:26s} vs spy: dSharpe {pt:+.2f} ci[{lo:+.2f},{hi:+.2f}] p={p:.3f} alpha {a:.1%} t={t:.1f} | vs ens2x: {pt2:+.2f} ci[{lo2:+.2f},{hi2:+.2f}]")
# stress
w = ens * (1 + winter)
for lag, c, bs in [(1, 2, .01), (2, 5, .02), (1, 10, .03)]:
    r, _ = run(w, d, lag=lag, cost_bp=c, borrow_spread=bs); s = stats(r.loc[P0:], d.rf)
    print(f"stress lag{lag} {c}bp borrow+{bs:.0%}: cagr {s['cagr']:.1%} sharpe {s['sharpe']:.2f} maxdd {s['maxdd']:.1%}")
