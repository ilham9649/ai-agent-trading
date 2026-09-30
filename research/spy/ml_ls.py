"""market-neutral monthly ml industry book added to S2. uses the seed-averaged walk-forward probabilities from f_seeds.py
(data/ml_industry_prob_seedavg.csv, 5 seeds, predictions 1950+). survivorship-free (crsp-based industries).
frozen before running, no retuning after:
  book  = long top 10, short bottom 10 by probability, equal weight, chosen at month end, held next month; 10 bp per unit turnover (both sides)
  combo = S2 (10 bp, rule C) + 0.5 x book
  pass  = cagr ABOVE S2 AND max drawdown no worse than S2, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_c

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
pa = pd.read_csv(D / "ml_industry_prob_seedavg.csv", index_col=[0, 1], parse_dates=[0]).iloc[:, 0].unstack()
pa = pa[pa.notna().sum(axis=1) >= 20]                                  # months with forecasts (1950+)
hi = pa.rank(axis=1, ascending=False, method="first") <= 10
lo = pa.rank(axis=1, ascending=True, method="first") <= 10
Wm = hi.astype(float) / 10 - lo.astype(float) / 10
W = Wm.reindex(idx).shift(1).ffill().fillna(0)
book = (W * R.fillna(0)).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * 10e-4
assert np.isclose(Wm.abs().sum(axis=1), 2).all() and np.abs(Wm.sum(axis=1)).max() < 1e-9
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var(); lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
m12 = lvl.shift(21) / lvl.shift(252) - 1
sel = m12[me].where(var[me].rank(axis=1, method="first") <= 20).rank(axis=1, ascending=False, method="first") <= 10
hold = sel.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
s2 = (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 10e-4).fillna(0)
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rS2 = run(rule_c(ens, idx), d.assign(r=s2), lag=0, borrow_spread=0.01)[0]
if __name__ == "__main__":
    combo = rS2 + 0.5 * book.reindex(rS2.index).fillna(0)
    tw = W.diff().abs().sum(axis=1)
    print(f"book turnover per rebalance {tw[tw > 0].mean():.2f} (of 2.0 gross); cost drag {(tw * 10e-4).loc['1950':].mean() * TD:.1%}/yr; "
          f"corr with S2 {book.loc['1950':].corr(rS2.loc['1950':]):+.2f}")
    ok = True
    for e, (a, b) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1, sb = stats(rS2.loc[a:b], d.rf), stats(combo.loc[a:b], d.rf), stats(book.loc[a:b] + d.rf.loc[a:b], d.rf)
        print(f"{e}: book alone {sb['cagr']:.1%}/{sb['sharpe']:.2f}/{sb['maxdd']:.1%}   S2 {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> S2 + 0.5 book {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
        ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
    print("ADOPT" if ok else "REJECT")
