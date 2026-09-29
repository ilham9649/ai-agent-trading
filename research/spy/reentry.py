"""fast re-entry, slow exit, on chosen rule E. base_F = max(ens, 1[px > sma50]): the slow band ensemble decides exits,
but a close above the standard 50-day sma puts the base back to full. (earlier tests averaged fast votes; this is an OR.)
frozen before running, one config, no grid:
  pass = vs E, dSharpe >= +0.03 AND maxdd no worse than E, in BOTH 1927-62 and 1963-2007.
  2008-26 and 2017-26 are in-sample (the 2018/2020/2025 whipsaws motivated the idea): reported only."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
fast = (d.px > d.px.rolling(50).mean()).astype(float)
baseF = np.maximum(ens, fast)
cfg = {"spy": pd.Series(1.0, index=idx).shift(1).fillna(0), "E": rule_e(ens, idx), "F fast re-entry": rule_e(baseF, idx)}
R = {k: run(w, d, lag=0, borrow_spread=0.01)[0] for k, w in cfg.items()}
E_ = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26 (in-sample)": ("2008-01-01", None), "2017-26 (in-sample)": ("2017-01-01", None)}
ok = True
for e, (a, b) in E_.items():
    print(f"\n=== {e} ===")
    for k, r in R.items():
        st = stats(r.loc[a:b], d.rf); print(f"{k:16s} cagr {st['cagr']:6.1%} sh {st['sharpe']:.2f} dd {st['maxdd']:6.1%}")
    pt, lo, hi, p = sharpe_diff_ci(R["F fast re-entry"].loc[a:b], R["E"].loc[a:b], d.rf, n=1000)
    print(f"  F vs E: dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")
    if "in-sample" not in e:
        ok &= pt >= 0.03 and stats(R["F fast re-entry"].loc[a:b], d.rf)["maxdd"] >= stats(R["E"].loc[a:b], d.rf)["maxdd"]
print("\nADOPT" if ok else "\nREJECT")

# bear-market rallies: each episode where the fast trigger adds exposure (fast on, ens < 1), market move over the episode
extra = (baseF > ens).astype(int)
starts = idx[(extra.diff() == 1).values]
print("\nfast-trigger episodes inside bear markets (start, end, days, market move, extra exposure):")
for a, b in [("1929-09", "1932-07"), ("1973-01", "1974-12"), ("2000-09", "2002-10"), ("2007-10", "2009-03"), ("2022-01", "2022-10")]:
    tot = []
    for s in starts[(starts >= a) & (starts <= b + "-28")]:
        seg = extra.loc[s:]; end = seg[seg == 0].index[0] if (seg == 0).any() else idx[-1]
        mv = d.px.loc[end] / d.px.loc[s] - 1
        dx = (baseF - ens).loc[s:end].mean(); tot.append(mv * dx)
        print(f"  {s.date()} -> {end.date()} {len(d.loc[s:end]):4d}d  mkt {mv:+6.1%}  extra {dx:.2f}")
    print(f"  {a}..{b}: sum of (move x extra) {sum(tot):+.1%}")
