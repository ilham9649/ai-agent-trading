"""fx time-series momentum stream (moskowitz, ooi, pedersen 2012) from free fred data, then the calibrate-then-test goal check.
currencies vs usd (fred daily noon rates): GBP (DEXUSUK), CAD (DEXCAUS), AUD (DEXUSAL), CHF (DEXSZUS), JPY (DEXJPUS), EUR (DEXUSEU).
carry: foreign short rate (oecd monthly, prior month's value) minus the us t-bill; a currency is used only while fx and rate both exist.
frozen before running:
  signal = sign of the currency's 252-day total return (spot + carry), lagged 2 days (noon fx vs us close), weight +-1/N; 3 bp costs
  combo  = R4 + 0.5 x ML long-short + 0.5 x bond CARRY + 0.5 x FX, R4 levered by k with the crash brake, other streams x k (as r4_div.py)
  k      = largest k on a 0.25 grid whose 1972-2007 max drawdown stays within the index's (calibration)
  pass   = with that k, 2008-2026 cagr >= 3x the index AND max drawdown no worse than the index. 1% borrow spread."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
import r4_div as v

idx, r = v.idx, v.r
spot = {"GBP": _fred("DEXUSUK"), "CAD": 1 / _fred("DEXCAUS"), "AUD": _fred("DEXUSAL"), "CHF": 1 / _fred("DEXSZUS"),
        "JPY": 1 / _fred("DEXJPUS"), "EUR": _fred("DEXUSEU")}
def mrate(*sids):
    s = pd.concat([_fred(x) for x in sids], axis=1).bfill(axis=1).iloc[:, 0]
    s.index = s.index.to_period("M"); s = s.shift(1)                                      # prior month's value
    return s
rates = {"GBP": mrate("IR3TIB01GBM156N"), "CAD": mrate("IR3TIB01CAM156N"), "AUD": mrate("IR3TIB01AUM156N"),
         "CHF": mrate("IR3TIB01CHM156N", "IRSTCI01CHM156N"), "JPY": mrate("IR3TIB01JPM156N", "IRSTCI01JPM156N"), "EUR": mrate("IR3TIB01EZM156N")}
S = pd.DataFrame({c: s.reindex(idx).ffill(limit=5) for c, s in spot.items()})
RT = pd.DataFrame({c: s.reindex(idx.to_period("M")).set_axis(idx) for c, s in rates.items()}) / 100
car = RT.sub(r.d.rf * TD, axis=0) / TD                                                   # daily carry of long foreign
tot = S.pct_change() + car
ok = S.notna() & S.shift(252).notna() & RT.notna() & RT.shift(252).notna()
tr12 = (1 + tot.fillna(0)).rolling(252).apply(np.prod, raw=True) - 1
sig = np.sign(tr12).where(ok)
W = sig.div(ok.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).shift(2)
fx = ((W * tot.fillna(0)).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * 3e-4).fillna(0)
print("currencies in use:", {y: int(ok.loc[str(y)].any().sum()) for y in [1973, 1986, 2000, 2010, 2026]})
R4u = run(v.w4, r.d.assign(r=r.b3), lag=0, borrow_spread=0.01)[0]
a0 = "1973-01-01"
print(f"fx stream alone: 1973-2007 {stats(fx.loc[a0:'2007'] + r.d.rf.loc[a0:'2007'], r.d.rf)['sharpe']:.2f} sharpe, 2008-26 "
      f"{stats(fx.loc['2008':] + r.d.rf.loc['2008':], r.d.rf)['sharpe']:.2f}; corr with R4 {fx.loc[a0:].corr(R4u.loc[a0:]):+.2f}, "
      f"with ML {fx.loc[a0:].corr(v.ml.loc[a0:]):+.2f}, with CARRY {fx.loc[a0:].corr(v.carry.loc[a0:]):+.2f}")

def combo(k, spread=0.01, with_fx=True):
    wk = pd.Series(v.w4.values * np.where(v.calm == 1, k, 1.0), index=idx)
    rr = run(wk, r.d.assign(r=r.b3), lag=0, borrow_spread=spread)[0]
    return (rr + k * 0.5 * (v.ml + v.carry + (fx if with_fx else 0))).loc[a0:]

P = {"1973-2007 (calibrate)": (a0, "2007-12-31"), "2008-26 (test)": ("2008-01-01", None)}
mk = {e: stats(r.d.r.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
for with_fx in [False, True]:
    ks = np.arange(1.0, 5.01, 0.25); kc = None
    for k in ks:
        if stats(combo(k, with_fx=with_fx).loc[a0:"2007"], r.d.rf)["maxdd"] >= mk["1973-2007 (calibrate)"]["maxdd"]: kc = k
    rk = combo(kc, with_fx=with_fx)
    st = {e: stats(rk.loc[a:b], r.d.rf) for e, (a, b) in P.items()}
    t = st["2008-26 (test)"]; m = mk["2008-26 (test)"]
    verdict = "MEETS GOAL" if t["cagr"] >= 3 * m["cagr"] and t["maxdd"] >= m["maxdd"] else "fails"
    print(f"{'with FX' if with_fx else 'without FX'}: calibrated k {kc:.2f} | " + " | ".join(f"{e} {s['cagr']:.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:.1%}" for e, s in st.items())
          + f" | 2008-26 = {t['cagr'] / m['cagr']:.2f}x index -> {verdict}")
