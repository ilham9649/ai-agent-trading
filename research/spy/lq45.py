"""trend rule on individual LQ45 stocks (yfinance *.JK, dividend-adjusted, IDR).
WARNING: uses a recent LQ45 member list, so stocks that left the index or were delisted are missing (survivorship bias):
absolute returns are too high. trend vs buy-and-hold on the SAME list is the fairer comparison.
rule (frozen, as designed on us data): per stock, band ensemble 150/200/250 +-3%, max 1x, decided at the close, held next day.
portfolio: equal weight across stocks with >= 250 days of history; a stock whose trend is off sits in cash (indonesian call rate).
costs 25 bp per unit of turnover. benchmarks: equal-weight buy & hold of the same stocks, and IHSG."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred

T = ["ACES", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARTO", "ASII", "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRIS",
     "BRPT", "CPIN", "CTRA", "ESSA", "EXCL", "GOTO", "ICBP", "INCO", "INDF", "INKP", "ISAT", "ITMG", "JPFA", "JSMR", "KLBF",
     "MAPA", "MAPI", "MBMA", "MDKA", "MEDC", "PGAS", "PGEO", "PTBA", "SIDO", "SMGR", "SMRA", "TLKM", "TOWR", "UNTR", "UNVR"]
raw = yf.download([t + ".JK" for t in T], period="max", progress=False, auto_adjust=True)["Close"]
raw.index = raw.index.tz_localize(None)
P = raw.where(raw > 0)
ok_cols = [c for c in P if P[c].notna().sum() > 300]
P = P[ok_cols]
Rr = P.pct_change(fill_method=None).where(lambda x: x.abs() < 0.5)                       # drop obvious bad ticks (> 50% in a day)
rate = _fred("IRSTCI01IDM156N"); rate.index = rate.index.to_period("M"); rate = rate.shift(1) / 100
rf = pd.Series((1 + rate.reindex(P.index.to_period("M")).values) ** (1 / TD) - 1, index=P.index).ffill()   # rate data ends a month early
ENS = pd.DataFrame({c: (band(P[c].dropna(), 150, .03) + band(P[c].dropna(), 200, .03) + band(P[c].dropna(), 250, .03)) / 3 for c in P}).reindex(P.index)
avail = P.notna() & (P.notna().cumsum() >= 250)
n = avail.sum(axis=1).replace(0, np.nan)
Wbh = avail.astype(float).div(n, axis=0).shift(1).fillna(0)
Wtr = ENS.where(avail).fillna(0).div(n, axis=0).shift(1).fillna(0)
def port(W, bp=25):
    inv = W.sum(axis=1)
    return (W * Rr.fillna(0)).sum(axis=1) + (1 - inv) * rf - W.diff().abs().sum(axis=1).fillna(0) * bp * 1e-4
rbh, rtr = port(Wbh), port(Wtr)
ih = yf.download("^JKSE", period="max", progress=False, auto_adjust=True)["Close"].squeeze(); ih.index = ih.index.tz_localize(None)
rih = ih.reindex(P.index).pct_change(fill_method=None).fillna(0)
print(f"{len(ok_cols)} stocks with data; stocks available per year:", {y: int(avail.loc[str(y)].any().sum()) for y in [2000, 2005, 2008, 2015, 2020, 2026]})
PER = {"2001-2007": ("2001-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None), "2015-26": ("2015-01-01", None)}
for e, (a, b) in PER.items():
    print(f"=== {e} ===")
    for k, r in {"IHSG (price)": rih, "LQ45 list eq-wt buy & hold": rbh, "LQ45 list trend rule": rtr}.items():
        s = stats(r.loc[a:b], rf); print(f"  {k:28s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
# per stock since 2008: does trend beat holding the same stock?
win_sh = win_dd = cnt = 0
for c in ok_cols:
    x = Rr[c].loc["2008":].dropna()
    if len(x) < 750: continue
    w = ENS[c].shift(1).reindex(x.index).fillna(0)
    rt = w * x + (1 - w) * rf.reindex(x.index) - w.diff().abs().fillna(0) * 25e-4
    s0, s1 = stats(x, rf), stats(rt, rf); cnt += 1
    win_sh += s1["sharpe"] > s0["sharpe"]; win_dd += s1["maxdd"] > s0["maxdd"]
print(f"per stock since 2008 ({cnt} stocks with >= 3 years): trend has higher sharpe in {win_sh}, smaller drawdown in {win_dd}")
