"""two calendar layers on chosen rule E, declared together, frozen before running:
  P  presidential cycle: base = ens x 1.5 in pre-election years (1927, 1931, ..., 2023), else ens
  S  summer half: base = ens x (1 if nov-apr else 0.5)   (season tested before in the 3x rule and in 10 countries, not inside E)
  both then rule E (tom x pre-holiday, cap 1.5)
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 1927-62, 1963-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
pre = pd.Series(((idx.year - 1927) % 4 == 0), index=idx).astype(float)
assert pre.loc["2023-06-01"] == 1 and pre.loc["2024-06-03"] == 0
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
cfg = {"P presidential": ens * (1 + 0.5 * pre), "S summer half": ens * (0.5 + 0.5 * winter)}
for name, base in cfg.items():
    rX, _ = run(rule_e(base, idx), d, lag=0, borrow_spread=0.01)
    print(f"\n##### {name}")
    ok = True
    for e, (a, b) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rE.loc[a:b], d.rf), stats(rX.loc[a:b], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(rX.loc[a:b], rE.loc[a:b], d.rf, n=1000)
        print(f"  {e}: E {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}  dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print("  ADOPT" if ok else "  REJECT")
