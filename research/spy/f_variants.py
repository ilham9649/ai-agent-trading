"""improvements to chosen rule F (E timing on the ml top-10 industry basket). all at 10 bp per unit of turnover.
frozen before running, 3 configs declared together, no retuning after:
  V1 buffer : keep a held industry while its rank <= 20; fill empty slots with the best-ranked new ones (always 10 held)
  V2 inv-vol: top 10, weights proportional to 1 / 60-day vol at the month end
  V3 top 20 : top 20, equal weight
  pass = vs F, dSharpe >= +0.03 AND max drawdown no worse than F, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_bc import rule_e
from ml_industry import industry_probs, top_hold

R, d, idx, ens, P, nxt, prob = industry_probs()
pw = prob.dropna().unstack()
rank = pw.rank(axis=1, ascending=False, method="first")
vol60 = R.rolling(60, min_periods=40).std()
w = rule_e(ens, idx).loc["1950":]

def book(Wm):
    """Wm: month-end target weights (rows sum to 1), held from the next trading day. returns basket return net of 10 bp."""
    W = Wm.reindex(idx).shift(1).ffill().fillna(0).loc["1950":]
    r = (W * R.loc["1950":].fillna(0)).sum(axis=1)
    return r - W.diff().abs().sum(axis=1).fillna(0) / 2 * 10e-4, W.diff().abs().sum(axis=1)[W.diff().abs().sum(axis=1) > 0].mean() / 2

eq = lambda sel: sel.astype(float).div(sel.sum(axis=1), axis=0)
cur, held = [], set()
for t, rk in rank.iterrows():
    keep = {c for c in held if rk.get(c, np.inf) <= 20}
    new = [c for c in rk.sort_values().index if c not in keep][: 10 - len(keep)]
    held = keep | set(new); cur.append(pd.Series(1.0, index=list(held)).reindex(rank.columns).fillna(0))
buf = pd.DataFrame(cur, index=rank.index)
assert (buf.sum(axis=1) == 10).all()
top10 = rank <= 10
iv = (1 / vol60.reindex(rank.index)).where(top10)
cfg = {"F (top 10)": eq(top10), "V1 buffer": eq(buf > 0), "V2 inv-vol": iv.div(iv.sum(axis=1), axis=0).fillna(0), "V3 top 20": eq(rank <= 20)}
res = {}
for k, Wm in cfg.items():
    b, to = book(Wm)
    res[k] = run(w, d.loc["1950":].assign(r=b), lag=0, borrow_spread=0.01)[0]
    print(f"{k:12s} turnover per rebalance {to:.0%}")
E_ = {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
for k in list(cfg)[1:]:
    ok = True; line = []
    for e, (a, bb) in E_.items():
        s0, s1 = stats(res["F (top 10)"].loc[a:bb], d.rf), stats(res[k].loc[a:bb], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(res[k].loc[a:bb], res["F (top 10)"].loc[a:bb], d.rf, n=1000)
        line.append(f"{e}: F {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%} dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print(f"\n##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
