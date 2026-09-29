"""feature variants for chosen rule F, all at 10 bp. 2 configs declared together, frozen:
  M4 season: + each industry's mean return in the same calendar month over the past 20 years (>= 5 obs)  [heston-sadka 2008]
  M5 macro : + t-bill yield level and its 252-day change
  pass = vs F, dSharpe >= +0.03 AND max drawdown no worse than F, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from ml_industry import industry_probs
from ml_models import f_returns, rF, d

for k, ex in {"M4 season": ("season",), "M5 macro": ("macro",)}.items():
    rX = f_returns(industry_probs(extra=ex)[6])
    ok, line = True, []
    for e, (a, bb) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rF.loc[a:bb], d.rf), stats(rX.loc[a:bb], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(rX.loc[a:bb], rF.loc[a:bb], d.rf, n=1000)
        line.append(f"{e}: F {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%} dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line), flush=True)
