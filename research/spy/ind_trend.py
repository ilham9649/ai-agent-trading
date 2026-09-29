"""trend following per industry: each of the 49 french industries (1926+) gets its own band ensemble (150/200/250, +-3%)
and exits alone. equal weight across available industries, each held at weight ens_i / N. then rule E's calendar layers
(tom x pre-holiday) scale the whole book, total exposure capped at 1.5x.
frozen before running, no retuning after:
  cost 2 bp per unit of turnover, cash earns t-bill, exposure above 1 pays t-bill + 1%
  pass = vs E on the market, dSharpe >= +0.03 AND max drawdown no worse than E, in 1927-62, 1963-2007 AND 2008-2026.
  note: the industry file was used before (momentum, breadth, low-vol), so 1927-62 is only partly clean."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e, tom_days, preholiday_days

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
ENS = pd.DataFrame({c: (band(lvl[c].dropna(), 150, .03) + band(lvl[c].dropna(), 200, .03) + band(lvl[c].dropna(), 250, .03)) / 3
                    for c in R.columns}).reindex(idx)
avail = R.notna() & lvl.rolling(250).count().ge(250)
W = (ENS.where(avail).fillna(0)).div(avail.sum(axis=1), axis=0)             # decided at close t
cal = (1 + tom_days(idx, 1, 3)) * (1 + preholiday_days(idx))

def book(W, cal, cap=1.5, cost_bp=2.0, spread=0.01):
    Wl = W.shift(1).fillna(0)
    expo = Wl.sum(axis=1)
    scale = np.minimum(cal, (cap / expo).where(expo > 0, 0)).fillna(0)
    P = Wl.mul(scale, axis=0)
    e = P.sum(axis=1)
    r = (P * R.fillna(0)).sum(axis=1) + (1 - e).clip(lower=0) * d.rf + (1 - e).clip(upper=0) * (d.rf + spread / TD)
    return (r - P.diff().abs().sum(axis=1).fillna(0) * cost_bp / 1e4).iloc[1:], e

ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
rI, eI = book(W, cal)
ew = R.where(avail).mean(axis=1).fillna(0)
assert eI.max() <= 1.5 + 1e-9 and avail.loc["1928"].sum(axis=1).min() >= 30
print(f"avg exposure: industry book {eI.loc['1928':].mean():.2f}, E {rule_e(ens, idx).loc['1928':].mean():.2f}")
ok = True
for e, (a, b) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r, "eq-wt industries": ew, "E": rE, "industry trend": rI}.items()}
    print(f"=== {e} ===")
    for k, s in st.items(): print(f"  {k:16s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%} vol {s['vol']:5.1%}")
    pt, lo, hi, _ = sharpe_diff_ci(rI.loc[a:b], rE.loc[a:b], d.rf, n=1000)
    print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= pt >= 0.03 and st["industry trend"]["maxdd"] >= st["E"]["maxdd"]
print("ADOPT" if ok else "REJECT")
