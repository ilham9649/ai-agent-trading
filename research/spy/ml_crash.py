"""machine-learning crash-risk model on top of E (risk forecast, not return forecast). same walk-forward and features as ml_forecast.py.
frozen before running, no retuning after:
  target = market excess return over the next 21 trading days < -5% (a large fall)
  model  = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
  base   = ens x (1 - clip((pc - 0.10) / 0.20, 0, 1)): full trend position below 10% crash probability, zero above 30%
  pass   = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import ml_forecast as mf
from engine import *
from strat_bc import rule_e

d, idx, X, ens, fwd, ok_row = mf.d, mf.idx, mf.X, mf.ens, mf.fwd, mf.ok_row
y = (fwd < -0.05).astype(float).where(fwd.notna())
pc = pd.Series(np.nan, index=idx)
for yr in range(1950, idx[-1].year + 1):
    test = (idx.year == yr) & ok_row
    cut = idx[idx < f"{yr}-01-01"][-22]
    tr = ok_row & y.notna() & (idx <= cut)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
    pc[test] = m.predict_proba(X[test])[:, 1]
v = pc.notna() & y.notna()
print(f"crash base rate {y[v].mean():.1%}; out-of-sample auc {roc_auc_score(y[v], pc[v]):.3f} "
      f"(auc of -ens as a crash score: {roc_auc_score(y[v], -ens[v]):.3f})")
base = ens * (1 - ((pc - 0.10) / 0.20).clip(0, 1).fillna(0))
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
rC, _ = run(rule_e(base, idx), d, lag=0, borrow_spread=0.01)
ok = True
for e, (a, b) in {"1950-2007": ("1950-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    st = {n: stats(v_.loc[a:b], d.rf) for n, v_ in {"spy": d.r, "E": rE, "E x crash model": rC}.items()}
    print(f"=== {e} ===")
    for n, s in st.items(): print(f"  {n:16s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rC.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["E x crash model"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
