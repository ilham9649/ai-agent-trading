"""permanent gold sleeve as an overlay on chosen rule E. gold = datahub core/gold-prices monthly (1833+; free float from 1972).
frozen before running, no retuning after:
  overlay = E monthly return + 0.2 x (gold return - t-bill)   (futures-like financing)
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in the 1972-2007 holdout AND 2008-2026. monthly stats.
  caveat: the monthly series may be monthly averages, which smooth gold's volatility; checked against daily GC=F after 2001."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from strat_f import french
from strat_bc import rule_e
from macro_sleeve import mstats, boot

g = pd.read_csv(D / "gold_monthly.csv", index_col=0)["Price"]; g.index = pd.PeriodIndex(g.index, freq="M")
gr = g.pct_change()
ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, d.index), d, lag=0, borrow_spread=0.01)
M = lambda r: (1 + r).groupby(r.index.to_period("M")).prod() - 1
X = pd.DataFrame({"E": M(rE), "rf": M(d.rf), "mkt": M(d.r), "gold": gr}).dropna()
X["E + 0.2 gold"] = X.E + 0.2 * (X.gold - X.rf)
gc = yf.download("GC=F", period="max", progress=False, auto_adjust=True)["Close"].squeeze()
gcm = gc.groupby(gc.index.to_period("M")).last().pct_change()
j = pd.concat([X.gold, gcm], axis=1).dropna()
print(f"check vs daily GC=F month-end returns ({j.index[0]}..): corr {j.corr().iloc[0, 1]:.2f}, vol {j.iloc[:, 0].std() * 12 ** .5:.1%} vs {j.iloc[:, 1].std() * 12 ** .5:.1%}")
print(f"corr(E, gold) {X.loc['1972':].E.corr(X.loc['1972':].gold):+.2f}")
ok = True
for e, (a, b) in {"1972-2007": ("1972-01", "2007-12"), "2008-26": ("2008-01", None)}.items():
    x = X.loc[a:b]; print(f"=== {e} ===")
    st = {k: mstats(x[k], x.rf) for k in ["mkt", "gold", "E", "E + 0.2 gold"]}
    for k, s in st.items(): print(f"  {k:14s} cagr {s['cagr']:6.1%} sh {s['sh']:.2f} dd {s['dd']:6.1%}")
    lo, hi = boot(x["E + 0.2 gold"], x.E, x.rf); dS = st["E + 0.2 gold"]["sh"] - st["E"]["sh"]
    print(f"  dSharpe vs E {dS:+.2f} [{lo:+.2f},{hi:+.2f}]")
    ok &= dS >= 0.03 and st["E + 0.2 gold"]["dd"] >= st["E"]["dd"]
print("ADOPT" if ok else "REJECT")

# after the frozen verdict: the monthly series is a monthly AVERAGE (world bank pink sheet via datahub), which smooths volatility.
# repeat the 2008-26 test with real daily prices: GC=F front gold futures (a futures return already nets financing, so no - t-bill).
if __name__ == "__main__":
    gd = gc.copy(); gd.index = gd.index.tz_localize(None)
    grd = gd.reindex(d.index.union(gd.index)).ffill().reindex(d.index).pct_change().loc["2008-01-01":]
    rG = rE.loc[grd.index] + 0.2 * grd.fillna(0)
    for e, (a, b) in {"2008-16": ("2008-01-01", "2016-12-31"), "2017-26": ("2017-01-01", None), "2008-26 daily": ("2008-01-01", None)}.items():
        st = {k: stats(v.loc[a:b], d.rf) for k, v in {"spy": d.r.loc[grd.index], "E": rE.loc[grd.index], "E + 0.2 gold (daily)": rG}.items()}
        print(f"=== {e} ===")
        for k, s in st.items(): print(f"  {k:20s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
        pt, lo, hi, _ = sharpe_diff_ci(rG.loc[a:b], rE.loc[a:b], d.rf, n=1000)
        print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
    print("worst gold futures day:", f"{grd.min():.1%}", grd.idxmin().date())

# tradable version: GLD etf (after its fee), financed at the t-bill rate: overlay 0.2 x (GLD - t-bill). 2005+.
if __name__ == "__main__":
    gl = yf.download("GLD", period="max", progress=False, auto_adjust=True)["Close"].squeeze(); gl.index = gl.index.tz_localize(None)
    glr = gl.reindex(d.index.union(gl.index)).ffill().reindex(d.index).pct_change().loc["2005-01-01":]
    rGL = rE.loc[glr.index] + 0.2 * (glr.fillna(0) - d.rf.loc[glr.index])
    print("\n##### GLD, 0.2 x (GLD - t-bill)")
    for e, (a, b) in {"2005-07": ("2005-01-01", "2007-12-31"), "2008-16": ("2008-01-01", "2016-12-31"), "2017-26": ("2017-01-01", None), "2008-26": ("2008-01-01", None)}.items():
        st = {k: stats(v.loc[a:b], d.rf) for k, v in {"E": rE.loc[glr.index], "E + 0.2 GLD": rGL}.items()}
        pt, lo, hi, _ = sharpe_diff_ci(rGL.loc[a:b], rE.loc[a:b], d.rf, n=1000)
        print(f"  {e}: E {st['E']['cagr']:.1%}/{st['E']['sharpe']:.2f}/{st['E']['maxdd']:.1%}  ->  {st['E + 0.2 GLD']['cagr']:.1%}/{st['E + 0.2 GLD']['sharpe']:.2f}/{st['E + 0.2 GLD']['maxdd']:.1%}   dSharpe {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")

# robustness for the 1972-2007 holdout: undo the smoothing of the averaged series by scaling gold's deviations from its mean
# by the measured vol ratio (daily-futures month-end vol / averaged-series vol, 2000-26), then retest.
if __name__ == "__main__":
    k = j.iloc[:, 1].std() / j.iloc[:, 0].std()
    x = X.loc["1972":"2007"].copy()
    gs = x.gold.mean() + k * (x.gold - x.gold.mean())
    ov = x.E + 0.2 * (gs - x.rf)
    s0, s1 = mstats(x.E, x.rf), mstats(ov, x.rf); lo, hi = boot(ov, x.E, x.rf)
    print(f"\nholdout 1972-2007 with gold vol scaled x{k:.2f}: E {s0['cagr']:.1%}/{s0['sh']:.2f}/{s0['dd']:.1%} -> {s1['cagr']:.1%}/{s1['sh']:.2f}/{s1['dd']:.1%}"
          f"  dSharpe {s1['sh'] - s0['sh']:+.2f} [{lo:+.2f},{hi:+.2f}]")
    for k2, name in [(1.5, "x1.5"), (2.0, "x2.0")]:
        ov2 = x.E + 0.2 * (x.gold.mean() + k2 * (x.gold - x.gold.mean()) - x.rf); s2 = mstats(ov2, x.rf)
        print(f"  stress {name}: {s2['cagr']:.1%}/{s2['sh']:.2f}/{s2['dd']:.1%}  dSharpe {s2['sh'] - s0['sh']:+.2f}")
