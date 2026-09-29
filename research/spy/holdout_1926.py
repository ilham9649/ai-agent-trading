"""one-shot holdout on us market 1926-1955 (ken french mkt-rf + rf, no survivorship, real dividends).
frozen before running, no retuning after:
  1 market b&h
  2 plain 200-day trend, 2x
  3 band ensemble (150/200/250, +-3%), 2x
  4 band ensemble x season (2x nov-apr, 1x may-oct)
  5 layer 4 x turn-of-month boost (x2 on last1+first3 days, cap 3x), calendar applied on the day (pooled6 timing)
cash = rf, borrow = rf + 1% (and rf + 3% stress). stats start after 250 trading days of warmup.
saturday sessions (pre-1953) are compounded into the next weekday."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

f = french("F-F_Research_Data_Factors").loc[:"1955-12-31"]
key = pd.Series(f.index, index=f.index).where(f.index.dayofweek != 5).bfill()   # saturday -> next weekday
g = pd.DataFrame({"r": f["Mkt-RF"] + f["RF"], "rf": f["RF"]}).groupby(key.values).apply(lambda x: (1 + x).prod() - 1)
d = g.copy(); d.index = pd.DatetimeIndex(d.index)
d["px"] = (1 + d.r).cumprod()
assert (d.index.dayofweek == 5).sum() == 0
print("rows/yr 1927-51 after merge:", round(d.loc["1927":"1951"].groupby(d.loc["1927":"1951"].index.year).size().mean()), "| saturdays left: 0")

ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
base = ens * (1 + winter)
L5 = (base.shift(1).fillna(0) * (1 + tom_days(d.index, 1, 3))).clip(upper=3.0)
ladder = [("1 market b&h", pd.Series(1.0, index=d.index), 1), ("2 plain sma200 2x", 2.0 * (d.px > d.px.rolling(200).mean()), 1),
          ("3 band ensemble 2x", 2 * ens, 1), ("4 ens x season", base, 1), ("5 + tom boost cap3", L5, 0)]
S0 = d.index[250]
bench = d.r.loc[S0:]
for bs in [0.01, 0.03]:
    print(f"\n=== borrow rf+{bs:.0%}, {S0.date()} to {d.index[-1].date()} ===")
    for name, w, lag in ladder:
        r, pos = run(w, d, lag=lag, borrow_spread=bs); r, pos = r.loc[S0:], pos.loc[S0:]
        s = stats(r, d.rf); mid = r.index[len(r) // 2]
        h1, h2 = stats(r.loc[:mid], d.rf), stats(r.loc[mid:], d.rf)
        extra = ""
        if name[0] != "1":
            pt, lo, hi, p = sharpe_diff_ci(r, bench, d.rf, n=1000)
            extra = f" | dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]"
        print(f"{name:20s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%} | h1 cagr {h1['cagr']:6.1%} sh {h1['sharpe']:.2f} | h2 cagr {h2['cagr']:6.1%} sh {h2['sharpe']:.2f}{extra}")
        if bs == 0.01 and name[0] in "15":
            wk = (1 + r).rolling(5).apply(np.prod, raw=True) - 1; mo = (1 + r).rolling(21).apply(np.prod, raw=True) - 1
            print(f"   tail: worst day {r.min():.1%}, week {wk.min():.1%}, month {mo.min():.1%}; max pos {pos.max():.1f}x; worst pos*mkt day {(pos * d.r.loc[S0:]).min():.1%}")
