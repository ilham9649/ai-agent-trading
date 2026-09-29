"""cross-sectional ml: rank the 49 french industries each month, hold the top 10, time the basket with rule E.
this is where ml usually helps: many assets x months (~58k rows) instead of one index. walk-forward, refit every january.
frozen before running, no retuning after:
  features (industry i, month end): ret 1m, 3m, 6m, 12-1m; 60d vol; 250d beta to market; log(px/sma200); drawdown from 252d high;
            + market ens and market 12m return (same for all industries that month)
  target  = next-month industry return above that month's cross-sectional median
  model   = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0); train on months whose
            label ended before the refit date; predictions 1950+
  book    = top 10 by probability, equal weight, 2 bp per unit turnover; timing = rule E (market signals), cap 1.5
  pass    = vs E on the market, dSharpe >= +0.03 AND max drawdown no worse than E, in 1950-2007 AND 2008-2026.
  caveat: the industry file was mined 3 times before (momentum, breadth, low-vol), so it is only partly clean."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from engine import *
from strat_f import french
from strat_bc import rule_e



def industry_probs():
    """walk-forward ml probabilities (month-end x industry) and the inputs they came from (~40 s)."""
    ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
    ff = french("F-F_Research_Data_Factors")
    x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
    R, d = x[ind.columns], x[["r", "rf"]].copy()
    d["px"] = (1 + d.r).cumprod()
    idx = d.index
    lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
    ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
    me = idx[pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max").values == idx]
    mvar = d.r.rolling(250, min_periods=200).var()
    F = {
        "r1": lvl / lvl.shift(21) - 1, "r3": lvl / lvl.shift(63) - 1, "r6": lvl / lvl.shift(126) - 1, "r12_1": lvl.shift(21) / lvl.shift(252) - 1,
        "vol60": R.rolling(60, min_periods=40).std(), "beta": R.rolling(250, min_periods=200).cov(d.r).div(mvar, axis=0),
        "sma200": np.log(lvl / lvl.rolling(200).mean()), "dd252": lvl / lvl.rolling(252).max() - 1,
    }
    P = pd.concat({k: v.loc[me] for k, v in F.items()}, axis=1).stack(future_stack=True)
    P["mkt_ens"] = ens.loc[me].reindex(P.index.get_level_values(0)).values
    P["mkt_r12"] = (d.px / d.px.shift(252) - 1).loc[me].reindex(P.index.get_level_values(0)).values
    mret = (1 + R.fillna(0)).groupby(idx.to_period("M")).prod() - 1                    # month m return (NaN-safe)
    mret = mret.where(R.notna().groupby(idx.to_period("M")).all())
    nxt = mret.shift(-1)
    nxt.index = me
    y = nxt.sub(nxt.median(axis=1), axis=0).gt(0).astype(float).where(nxt.notna()).stack(future_stack=True)
    P["y"] = y.reindex(P.index)
    P = P.dropna(subset=list(F) + ["mkt_ens", "mkt_r12"])
    feats = list(F) + ["mkt_ens", "mkt_r12"]
    prob = pd.Series(np.nan, index=P.index)
    dates = P.index.get_level_values(0)
    for yr in range(1950, idx[-1].year + 1):
        test = dates.year == yr
        cut = pd.Timestamp(f"{yr - 1}-11-30")                                          # month-end labels through dec (yr-1) need jan data: stop at nov
        tr = (dates <= cut) & P.y.notna()
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0).fit(P.loc[tr, feats], P.loc[tr, "y"])
        if test.any(): prob[test] = m.predict_proba(P.loc[test, feats])[:, 1]
    return R, d, idx, ens, P, nxt, prob


def top_hold(prob, idx, n=10):
    """industries chosen at each month end, held from the next trading day."""
    pw = prob.dropna().unstack()
    top = pw.rank(axis=1, ascending=False, method="first") <= n
    return top.reindex(idx).shift(1).ffill().fillna(False).astype(bool)


if __name__ == "__main__":
    R, d, idx, ens, P, nxt, prob = industry_probs()
    pw = prob.dropna().unstack()
    top = pw.rank(axis=1, ascending=False, method="first") <= 10
    hold = top.reindex(idx).shift(1).ffill().fillna(False).astype(bool).loc["1950":]   # chosen at month end, held from next day
    assert (hold.loc["1950-02":].sum(axis=1) == 10).all()
    b = R.loc["1950":].where(hold).mean(axis=1)
    turn = hold.astype(float).diff().abs().sum(axis=1) / 20
    db = d.loc["1950":].assign(r=b - turn.fillna(0) * 2e-4)
    w = rule_e(ens, idx).loc["1950":]
    rE, _ = run(w, d.loc["1950":], lag=0, borrow_spread=0.01)
    rB, _ = run(w, db, lag=0, borrow_spread=0.01)
    hit = ((prob > 0.5) == (P.y == 1))[prob.notna() & P.y.notna()].mean()
    spread = (nxt.stack(future_stack=True).reindex(prob.index)[prob.notna()].groupby(level=0).apply(lambda s: s[prob.loc[s.index].rank(ascending=False) <= 10].mean() - s.mean()))
    print(f"oos hit rate {hit:.1%}; top-10 minus all-industry next-month return: {spread.mean() * 12:+.1%}/yr (t {spread.mean() / spread.std() * len(spread) ** .5:.1f}); "
          f"since 2008 {spread.loc['2008':].mean() * 12:+.1%}/yr")
    ok = True
    for e, (a, bb) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        st = {k: stats(v.loc[a:bb], d.rf) for k, v in {"spy": d.r.loc["1950":], "ml top-10 b&h": db.r, "E": rE, "E on ml top-10": rB}.items()}
        print(f"=== {e} ===")
        for k, s in st.items(): print(f"  {k:15s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
        pt, lo, hi, _ = sharpe_diff_ci(rB.loc[a:bb], rE.loc[a:bb], d.rf, n=1000)
        print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and st["E on ml top-10"]["maxdd"] >= st["E"]["maxdd"]
    print("ADOPT" if ok else "REJECT")

    # after the frozen verdict (does not change it): turnover and cost stress
    mt = turn[turn > 0]
    print(f"\nturnover per monthly rebalance {mt.mean():.0%} one-way (~{mt.mean() * 12:.1f}x per year)")
    for bp in [10, 25]:
        dbs = d.loc["1950":].assign(r=b - turn.fillna(0) * bp * 1e-4)
        rs, _ = run(w, dbs, lag=0, borrow_spread=0.01)
        for e, (a, bb) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
            s = stats(rs.loc[a:bb], d.rf); pt, *_ = sharpe_diff_ci(rs.loc[a:bb], rE.loc[a:bb], d.rf, n=300)
            print(f"  cost {bp} bp, {e}: cagr {s['cagr']:.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:.1%}  dSharpe vs E {pt:+.2f}")
