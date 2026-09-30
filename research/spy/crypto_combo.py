"""S2 (rule C timing on the low-vol + momentum industry basket) + a crypto trend sleeve.
coins fixed in advance = the 6 largest by market cap in jan 2018 (not today's winners): BTC ETH XRP BCH ADA LTC (yfinance *-USD).
sleeve rule = the repo's RobustTrend entry/exit, vectorised, without its atr stop and take-profit:
  enter when close > prior 20-day high AND close > sma100; exit when close < prior 10-day low OR close < sma100. decided at the close,
  held from the next day. equal weight across coins in a trend (cash otherwise), 20 bp per unit of turnover.
combo = S2 + 0.25 x (sleeve - t-bill). frozen before running. no data before 2017-11, so no old holdout: report only."""
import numpy as np, pandas as pd, yfinance as yf
import cl_next as c
from engine import *

coins = ["BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "ADA-USD", "LTC-USD"]
px = yf.download(coins, period="max", progress=False, auto_adjust=True)
C, H, L = px["Close"], px["High"], px["Low"]
for f in (C, H, L): f.index = f.index.tz_localize(None)
on = pd.DataFrame(0.0, index=C.index, columns=coins)
sma = C.rolling(100).mean(); hi20 = H.shift(1).rolling(20).max(); lo10 = L.shift(1).rolling(10).min()
for k in coins:
    st, out = 0.0, []
    for cl, s, h, l in zip(C[k].values, sma[k].values, hi20[k].values, lo10[k].values):
        if np.isnan(s) or np.isnan(h) or np.isnan(cl): st = 0.0
        elif st == 0 and cl > h and cl > s: st = 1.0
        elif st == 1 and (cl < l or cl < s): st = 0.0
        out.append(st)
    on[k] = out
avail = C.notna() & sma.notna()
W = on.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)            # equal weight across available coins
rc = C.pct_change()
Wl = W.shift(1).fillna(0)
sleeve_cal = (Wl * rc.fillna(0)).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * 20e-4
# to us trading days: compound calendar-day sleeve returns into the next us trading day
idx = c.idx
grp = pd.Series(idx.searchsorted(sleeve_cal.index), index=sleeve_cal.index)
ok_ = grp < len(idx)
sleeve = (1 + sleeve_cal[ok_]).groupby(idx[grp[ok_]]).prod().reindex(idx).fillna(1) - 1
start = pd.Timestamp("2018-06-01")
to = c.h_s2.astype(float).diff().abs().sum(axis=1) / 20
bs = (c.R.where(c.h_s2).mean(axis=1) - to.fillna(0) * 10e-4).fillna(0)
rS2 = run(c.wC, c.d.assign(r=bs), lag=0, borrow_spread=0.01)[0]
rf = c.d.rf
combo = rS2 + 0.25 * (sleeve - rf)
assert sleeve.loc[start:].notna().all() and avail.loc[start].sum() == 6
print(f"coins in trend on average: {on.loc[start:].sum(axis=1).mean():.1f} of 6; corr(sleeve, S2) {sleeve.loc[start:].corr(rS2.loc[start:]):+.2f}")
R = {"spy": c.d.r, "S2 (10 bp)": rS2, "crypto sleeve alone": sleeve + rf, "S2 + 0.25 crypto": combo}
for e, (a, b) in {"2018-06..2021": (start, "2021-12-31"), "2022-26": ("2022-01-01", None), "all 2018-06..2026-08": (start, None)}.items():
    print(f"=== {e} ===")
    for k, r in R.items():
        s = stats(r.loc[a:b], rf); print(f"  {k:20s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
# result note: 2018-06..2026-08 combo 20.2%/0.89/-26.8% vs S2 12.5%/0.63/-21.0%: more return, larger drawdown.
# nearly all the gain is 2018-21 (sleeve 71%/yr); 2022-26 sleeve 8.2%/yr, combo 6.6%. survivorship: coins = 2018 top 6, all still listed;
# coins that died later (not in the 2018 top 6) are not covered.
