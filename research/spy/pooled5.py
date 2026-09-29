"""turn-of-month leverage boost on the base rule: cap, window plateau, stress, significance, tail risk."""
import numpy as np, pandas as pd
from engine import *
from pooled import pooled, signals, evaluate
from strat_bc import tom_days

d = pooled(); S = signals(d.px)
ens = S[["band150", "band200", "band250"]].mean(axis=1)
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
base = ens * (1 + winter)
P0 = "1956-01-01"
mk = lambda tb, cap, nl=1, nf=3: (base * (1 + (tb - 1) * tom_days(d.index, nl, nf))).clip(upper=cap)
cfg = {"spy b&h": pd.Series(1.0, index=d.index), "base": base}
for tb, cap in [(1.5, 3.0), (2.0, 3.0), (2.0, 2.5)]:
    cfg[f"tom x{tb} cap {cap}"] = mk(tb, cap)
for nl, nf in [(1, 2), (2, 3), (1, 4)]:
    cfg[f"tom x2 cap3, window last{nl}+first{nf}"] = mk(2.0, 3.0, nl, nf)
evaluate(cfg, d)

bench = d.r.loc[P0:]; ref, _ = run(base, d, borrow_spread=0.01); ref = ref.loc[P0:]
w = mk(2.0, 3.0)
print("\nchosen: tom x2 cap 3 (window last1+first3)")
for lag, c, bs in [(1, 2, .01), (1, 5, .02), (2, 5, .02), (1, 10, .03)]:
    r, pos = run(w, d, lag=lag, cost_bp=c, borrow_spread=bs); s = stats(r.loc[P0:], d.rf)
    print(f"  lag{lag} {c}bp borrow+{bs:.0%}: cagr {s['cagr']:.1%} sharpe {s['sharpe']:.2f} sortino {s['sortino']:.2f} maxdd {s['maxdd']:.1%}")
r, pos = run(w, d, borrow_spread=0.01); r = r.loc[P0:]
pt, lo, hi, p = sharpe_diff_ci(r, bench, d.rf); a, t, beta = alpha(r, bench, d.rf)
pt2, lo2, hi2, p2 = sharpe_diff_ci(r, ref, d.rf)
print(f"  vs spy: dSharpe {pt:+.2f} ci[{lo:+.2f},{hi:+.2f}] p={p:.3f} alpha {a:.1%} t={t:.1f}; vs base: {pt2:+.2f} ci[{lo2:+.2f},{hi2:+.2f}] p={p2:.3f}")
print("  worst day %.1f%% (spy %.1f%%), worst 5-day %.1f%%, max leverage %.1fx, days >2.5x: %.0f%%, trades/yr %.0f" % (
    r.min() * 100, bench.min() * 100, r.rolling(5).apply(lambda x: (1 + x).prod() - 1).min() * 100, pos.max(), (pos > 2.5).mean() * 100, (pos.diff().abs() > 0.01).sum() / (len(pos) / TD)))
