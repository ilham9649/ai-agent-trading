"""robustness of chosen rule F to the model's random seed (early stopping uses a random validation split). 10 bp costs."""
import numpy as np, pandas as pd
from engine import *
from ml_industry import industry_probs
from ml_models import f_returns, rF, d, w, idx
from strat_bc import rule_e

rE = run(w, d.loc["1950":], lag=0, borrow_spread=0.01)[0]
rows = {"E (market)": rE, "F seed 0": rF}
for s in [1, 2, 3, 4]:
    rows[f"F seed {s}"] = f_returns(industry_probs(seed=s)[6])
for k, r in rows.items():
    out = []
    for e, (a, bb) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s_ = stats(r.loc[a:bb], d.rf); out.append(f"{e} {s_['cagr']:.1%}/{s_['sharpe']:.2f}/{s_['maxdd']:.1%}")
    print(f"{k:11s} " + "   ".join(out), flush=True)

# seed-averaged model (bagging over seeds 0-4): the stable version of F
probs = [industry_probs(seed=s)[6] for s in range(5)]
pa = sum(probs) / 5
ra = f_returns(pa)
for bp in [2, 10]:
    r_ = f_returns(pa, cost_bp=bp)
    for e, (a, bb) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rE.loc[a:bb], d.rf), stats(r_.loc[a:bb], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(r_.loc[a:bb], rE.loc[a:bb], d.rf, n=1000)
        print(f"F seed-avg, {bp} bp, {e}: E {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}  dSh vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]", flush=True)
pd.concat(probs, axis=1).mean(axis=1).to_csv(D / "ml_industry_prob_seedavg.csv")
