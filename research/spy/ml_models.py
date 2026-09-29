"""model variants for chosen rule F (E timing on the ml top-10 industry basket), all at 10 bp. 3 configs declared together, frozen:
  M1 riskadj: target = next-month return / 60d vol above the cross-sectional median
  M2 gb+lr  : average of gradient boosting and logistic regression probabilities (1-month target)
  M3 3m     : target = next 3-month return above median (label ends 3 months later; training cut moved back to match)
  pass = vs F, dSharpe >= +0.03 AND max drawdown no worse than F, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_bc import rule_e
from ml_industry import industry_probs

base = industry_probs()
R, d, idx, ens = base[:4]
w = rule_e(ens, idx).loc["1950":]

def f_returns(prob, cost_bp=10):
    top = prob.dropna().unstack().rank(axis=1, ascending=False, method="first") <= 10
    W = top.astype(float).div(10).reindex(idx).shift(1).ffill().fillna(0).loc["1950":]
    b = (W * R.loc["1950":].fillna(0)).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) / 2 * cost_bp * 1e-4
    return run(w, d.loc["1950":].assign(r=b), lag=0, borrow_spread=0.01)[0]

rF = f_returns(base[6])
cfg = {"M1 riskadj": dict(target="riskadj"), "M2 gb+lr": dict(model="gb+lr"), "M3 3m": dict(target="3m")}
for k, kw in cfg.items():
    rX = f_returns(industry_probs(**kw)[6])
    ok, line = True, []
    for e, (a, bb) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rF.loc[a:bb], d.rf), stats(rX.loc[a:bb], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(rX.loc[a:bb], rF.loc[a:bb], d.rf, n=1000)
        line.append(f"{e}: F {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%} dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line), flush=True)
