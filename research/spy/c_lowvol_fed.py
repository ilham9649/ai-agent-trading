"""two tests on user-chosen rule C, declared together, frozen before running, no retuning after:
  L  C on the daily low-vol industry basket (10 lowest 60d variance of the 49 industries, equal weight, monthly; as lowvol_ind.py),
     signals from the market
  H  C x fed cut: base = ens x 0.5 while the t-bill yield is up > 1 pt over 252 days (as fed_cycle.py). RETEST of an E near-miss.
  pass = vs C, dSharpe >= +0.03 AND max drawdown no worse than C, in 1927-62, 1963-2007 AND 2008-2026."""
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
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rC, _ = run(rule_c(ens, idx), d, lag=0, borrow_spread=0.01)
var = R.rolling(60, min_periods=40).var()
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
hold = (var[me].rank(axis=1, method="first") <= 10).reindex(idx).shift(1).ffill().fillna(False).astype(bool)
b = R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 2e-4
rL, _ = run(rule_c(ens, idx), d.assign(r=b.fillna(0)), lag=0, borrow_spread=0.01)
y = (d.rf * TD * 100).rolling(21).mean()
hike = (y - y.shift(252) > 1.0).astype(float)
rH, _ = run(rule_c(ens * (1 - 0.5 * hike), idx), d, lag=0, borrow_spread=0.01)
assert (hold.sum(axis=1).loc["1928":] == 10).all()
for name, rX in [("L C on low-vol basket", rL), ("H C x fed cut (retest)", rH)]:
    ok, line = True, []
    for e, (a, bb) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rC.loc[a:bb], d.rf), stats(rX.loc[a:bb], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(rX.loc[a:bb], rC.loc[a:bb], d.rf, n=1000)
        line.append(f"{e}: C {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%} dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {name}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))

# after the frozen verdict: (1) note L is a retest of the E near-miss (lowvol_ind.py), holdouts seen before.
# (2) cost stress on the basket, (3) reality check with real low-vol etfs (after fees), signals from the market.
import yfinance as yf
for bp in [10, 25]:
    bs = R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * bp * 1e-4
    rs, _ = run(rule_c(ens, idx), d.assign(r=bs.fillna(0)), lag=0, borrow_spread=0.01)
    s = stats(rs.loc["2008":], d.rf); print(f"basket cost {bp} bp, 2008-26: {s['cagr']:.1%}/{s['sharpe']:.2f}/{s['maxdd']:.1%}")
for tk in ["USMV", "SPLV"]:
    p = yf.download(tk, period="max", progress=False, auto_adjust=True)["Close"].squeeze(); p.index = p.index.tz_localize(None)
    rt = p.reindex(idx.union(p.index)).ffill().reindex(idx).pct_change()
    a = rt.first_valid_index() + pd.Timedelta(days=1)
    rX, _ = run(rule_c(ens, idx), d.assign(r=rt.fillna(0)), lag=0, borrow_spread=0.01)
    s0, s1, sm = stats(rC.loc[a:], d.rf), stats(rX.loc[a:], d.rf), stats(d.r.loc[a:], d.rf)
    pt, lo, hi, _ = sharpe_diff_ci(rX.loc[a:], rC.loc[a:], d.rf, n=1000)
    print(f"{tk} {a.date()}..: spy {sm['cagr']:.1%}/{sm['sharpe']:.2f}/{sm['maxdd']:.1%}  C {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%}"
          f" -> C on {tk} {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}  dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
