"""market-neutral weekly ml on the 49 industries (lead-lag; hong, torous, valkanov 2007), added to S2. survivorship-free (crsp-based).
frozen before running, no retuning after:
  week = friday close to friday close. features per (week, industry): own 1w/4w/13w return, own 13w vol, market 1w/4w return,
         and the 1w return of all 49 industries (lead-lag). target = next-week return above the cross-sectional median.
  model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05), 3 seeds averaged; refit every january on
          weeks whose label ended before the year; predictions 1950+.
  book  = long top 5, short bottom 5, equal weight, dollar neutral; 10 bp per unit of turnover (both sides).
  combo = S2 (10 bp) + 0.5 x book.  pass = cagr ABOVE S2 AND max drawdown no worse than S2, in 1950-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from engine import *
from strat_f import french
from strat_bc import rule_c

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
# weekly returns (week ending friday); a week counts for an industry only if it traded every day of it
wk = idx.to_period("W-FRI")
Rw = (1 + R.fillna(0)).groupby(wk).prod() - 1
Rw = Rw.where(R.notna().groupby(wk).all())
mw = (1 + d.r).groupby(wk).prod() - 1
lv = (1 + Rw.fillna(0)).cumprod().where(Rw.notna())
F = {"r1": Rw, "r4": lv / lv.shift(4) - 1, "r13": lv / lv.shift(13) - 1, "vol13": Rw.rolling(13, min_periods=10).std()}
P = pd.concat(F, axis=1).stack(future_stack=True)
wi = P.index.get_level_values(0)
P["m1"] = mw.reindex(wi).values
P["m4"] = ((1 + mw).rolling(4).apply(np.prod, raw=True) - 1).reindex(wi).values
L = Rw.add_prefix("L_").fillna(0)                                                 # lead-lag block: every industry's last-week return
P = pd.concat([P.reset_index(drop=False), L.reindex(wi).reset_index(drop=True)], axis=1).set_index(["level_0", "level_1"])
nxt = Rw.shift(-1)
y = nxt.sub(nxt.median(axis=1), axis=0).gt(0).astype(float).where(nxt.notna()).stack(future_stack=True)
P["y"] = y.reindex(P.index).values
P = P.dropna(subset=["r1", "r4", "r13", "vol13", "m1", "m4"])
feats = [c for c in P.columns if c != "y"]
wks = P.index.get_level_values(0)
cache = D / "leadlag_prob.pkl"
if cache.exists():
    prob, has = pd.read_pickle(cache)
else:
    prob = pd.Series(0.0, index=P.index); has = pd.Series(False, index=P.index)
    for yr in range(1950, wks.max().end_time.year + 1):
        test = (wks.end_time.year == yr)
        if not test.any(): continue
        last_train = pd.Period(f"{yr - 1}-12-31", "W-FRI") - 1                  # its label (next week) ends by the first days of yr at most
        tr = (wks <= last_train) & P.y.notna() & (wks.end_time < pd.Timestamp(f"{yr}-01-01"))
        for s in range(3):
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=s).fit(P.loc[tr, feats], P.loc[tr, "y"])
            prob[test] += m.predict_proba(P.loc[test, feats])[:, 1] / 3
        has[test] = True
        print(yr, end=" ", flush=True)
    print()
    pd.to_pickle((prob, has), cache)
pw = prob[has].unstack()
rk_hi = pw.rank(axis=1, ascending=False, method="first"); rk_lo = pw.rank(axis=1, ascending=True, method="first")
W = (rk_hi <= 5).astype(float) / 5 - (rk_lo <= 5).astype(float) / 5          # decided at friday close
bookw = (W.shift(1) * Rw.reindex(W.index).fillna(0)).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * 10e-4
gross = W.abs().sum(axis=1).iloc[1:]
print(f"weeks with a full 5/5 book: {np.isclose(gross, 2).mean():.1%}")
assert np.isclose(gross, 2).mean() > 0.99 and abs(W.sum(axis=1)).max() < 1e-9
# spread the weekly book return evenly over the week's trading days (approximation for combining with daily S2)
ndays = pd.Series(1, index=idx).groupby(wk).transform("sum")
book = ((1 + bookw.reindex(wk).set_axis(idx)) ** (1 / ndays) - 1).fillna(0)
# S2
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var(); lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
m12 = lvl.shift(21) / lvl.shift(252) - 1
sel = m12[me].where(var[me].rank(axis=1, method="first") <= 20).rank(axis=1, ascending=False, method="first") <= 10
hold = sel.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
s2 = (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 10e-4).fillna(0)
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rS2 = run(rule_c(ens, idx), d.assign(r=s2), lag=0, borrow_spread=0.01)[0]
combo = rS2 + 0.5 * book.reindex(rS2.index).fillna(0)
st = lambda r, a, b: stats(r.loc[a:b], d.rf)
ok = True
print(f"long-short book: weekly turnover {W.diff().abs().sum(axis=1).iloc[1:].mean():.2f}; corr with S2 (weekly) "
      f"{(1 + rS2).groupby(rS2.index.to_period('W-FRI')).prod().sub(1).corr(bookw):+.2f}")
for e, (a, b) in {"1950-2007": ("1950-02-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    s0, s1, sb = st(rS2, a, b), st(combo, a, b), st(book + d.rf, a, b)
    print(f"{e}: book alone {sb['cagr']:.1%}/{sb['sharpe']:.2f}/{sb['maxdd']:.1%}   S2 {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> S2 + 0.5 book {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
    ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
print("ADOPT" if ok else "REJECT")
bookw.to_csv(D / "leadlag_book_weekly.csv")

# after the frozen verdict: is there any signal before costs?
tw = W.diff().abs().sum(axis=1).fillna(0)
gross = bookw + tw * 10e-4
for e, (a, b) in {"1950-2007": ("1950", "2007"), "2008-26": ("2008", None)}.items():
    g = gross.loc[a:b]; print(f"gross (no costs) {e}: {g.mean() * 52:+.1%}/yr, t {g.mean() / g.std() * len(g) ** .5:.1f}; cost drag {(tw.loc[a:b] * 10e-4).mean() * 52:.1%}/yr")
