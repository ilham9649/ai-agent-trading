"""machine-learning forecast replacing E's trend signal. walk-forward: refit every january on all data up to the prior
december, minus 21 trading days (so every training label was known). predictions 1950-2026 are all out-of-sample.
frozen before running, no retuning after:
  target  = market excess return over the next 21 trading days > 0
  features (known at close t): log(px/sma n) n=50,100,150,200,250; returns 21/63/126/252d; realised vol 21/63d; vol 21/252 ratio;
            drawdown from 252d high; t-bill yield and its 252d change; winter flag; ens value
  models  = GB: sklearn HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
            LR: StandardScaler + LogisticRegression(C=1.0)
  base    = clip((p - 0.45) / 0.2, 0, 1), then rule E layers (tom x pre-holiday, cap 1.5), position lagged 1 day as in E
  pass    = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from engine import *
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx, px = d.index, d.px
ens = (band(px, 150, .03) + band(px, 200, .03) + band(px, 250, .03)) / 3
X = pd.DataFrame(index=idx)
for n in [50, 100, 150, 200, 250]: X[f"sma{n}"] = np.log(px / px.rolling(n).mean())
for n in [21, 63, 126, 252]: X[f"ret{n}"] = px.pct_change(n)
X["vol21"], X["vol63"] = d.r.rolling(21).std() * TD ** .5, d.r.rolling(63).std() * TD ** .5
X["volratio"] = d.r.rolling(21).std() / d.r.rolling(252).std()
X["dd252"] = px / px.rolling(252).max() - 1
yld = (d.rf * TD * 100).rolling(21).mean()
X["yield"], X["dyield"] = yld, yld - yld.shift(252)
X["winter"] = idx.month.isin([11, 12, 1, 2, 3, 4]).astype(float)
X["ens"] = ens
ex = d.r - d.rf
fwd = ex[::-1].rolling(21).sum()[::-1].shift(-1)                 # excess return over t+1..t+21
y = (fwd > 0).astype(float).where(fwd.notna())
ok_row = X.notna().all(axis=1)

if __name__ == "__main__":
    models = {"GB": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0),
              "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))}
    P = {k: pd.Series(np.nan, index=idx) for k in models}
    for yr in range(1950, idx[-1].year + 1):
        test = (idx.year == yr) & ok_row
        cut = idx[idx < f"{yr}-01-01"][-22]                           # last row whose 21d label is complete before the year starts
        tr = ok_row & y.notna() & (idx <= cut)
        assert fwd.index[tr].max() <= cut
        for k, mk in models.items():
            m = mk().fit(X[tr], y[tr]); P[k][test] = m.predict_proba(X[test])[:, 1]

    rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
    E_ = {"1950-2007": ("1950-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
    for k, p in P.items():
        base = ((p - 0.45) / 0.2).clip(0, 1).fillna(0)
        rM, _ = run(rule_e(base, idx), d, lag=0, borrow_spread=0.01)
        hit = ((p > 0.5) == (y == 1))[p.notna() & y.notna()].mean()
        print(f"\n##### {k}: out-of-sample hit rate {hit:.1%} (base rate up {y[p.notna()].mean():.1%}); mean base {base.loc['1950':].mean():.2f} vs ens {ens.loc['1950':].mean():.2f}")
        ok = True
        for e, (a, b) in E_.items():
            st = {n: stats(v.loc[a:b], d.rf) for n, v in {"spy": d.r, "E": rE, f"ML {k}": rM}.items()}
            print(f"=== {e} ===")
            for n, s in st.items(): print(f"  {n:6s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
            pt, lo, hi, _ = sharpe_diff_ci(rM.loc[a:b], rE.loc[a:b], d.rf, n=1000)
            print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
            ok &= pt >= 0.03 and st[f"ML {k}"]["maxdd"] >= st["E"]["maxdd"]
        print("ADOPT" if ok else "REJECT")
    pd.DataFrame(P).to_csv(D / "ml_forecast_probs.csv")
