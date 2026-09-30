"""one rule for all markets: standard trend rules applied unchanged to 16 stock indexes (yfinance, local currency; DAX is total
return, the others price only). cash = local short rate from fred where available (prior month), else 0 (HK, SG, TW). 10 bp costs.
rules (declared together, same settings everywhere, no per-market tuning):
  T1 sma200: in (1x) when close > 200-day average
  T2 band ensemble: mean of 150/200/250-day bands +-3% (our trend core), max 1x
  T3 12m momentum: in when the 252-day return > 0
  E: T2 x turn-of-month x pre-holiday (own trading calendar), cap 1.5, borrow local rate + 1%
score = number of markets where the rule beats buy & hold on sharpe / on max drawdown (full history after warmup, and 2008-26).
global = equal weight of each market's excess return over its own cash, + us t-bill (a currency-hedged approximation)."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred
from strat_bc import rule_e

M = {"us": ("^GSPC", ["DTB3"]), "japan": ("^N225", ["IRSTCI01JPM156N", "IR3TIB01JPM156N"]), "uk": ("^FTSE", ["IR3TIB01GBM156N"]),
     "germany": ("^GDAXI", ["IR3TIB01DEM156N"]), "france": ("^FCHI", ["IR3TIB01FRM156N"]), "australia": ("^AXJO", ["IR3TIB01AUM156N"]),
     "canada": ("^GSPTSE", ["IR3TIB01CAM156N"]), "korea": ("^KS11", ["IR3TIB01KRM156N"]), "india": ("^BSESN", ["IRSTCI01INM156N"]),
     "brazil": ("^BVSP", ["IRSTCI01BRM156N"]), "mexico": ("^MXX", ["IRSTCI01MXM156N"]), "hong kong": ("^HSI", []),
     "singapore": ("^STI", []), "taiwan": ("^TWII", []), "indonesia": ("^JKSE", ["IRSTCI01IDM156N"]), "switzerland": ("^SSMI", [])}
px_all = yf.download([t for t, _ in M.values()], period="max", progress=False, auto_adjust=True)["Close"]
px_all.index = px_all.index.tz_localize(None)

def cash(sids, idx):
    if not sids: return pd.Series(0.0, index=idx)
    if sids == ["DTB3"]: return (_fred("DTB3").reindex(idx).ffill() / 100 / TD).fillna(0)
    s = pd.concat([_fred(x) for x in sids], axis=1, sort=True).bfill(axis=1).iloc[:, 0]
    s.index = s.index.to_period("M"); s = s.shift(1) / 100
    return pd.Series((1 + s.reindex(idx.to_period("M")).values) ** (1 / TD) - 1, index=idx).ffill()

rows, EX = [], {}
for name, (tk, sids) in M.items():
    p = px_all[tk].dropna(); p = p[p > 0]
    d = pd.DataFrame({"px": p}); d["r"] = d.px.pct_change(); d["rf"] = cash(sids, d.index); d = d.dropna()
    d = d[d.r.abs() < 0.3]                                                   # drop bad ticks
    idx = d.index
    ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
    W = {"buy & hold": pd.Series(1.0, index=idx), "T1 sma200": (d.px > d.px.rolling(200).mean()).astype(float),
         "T2 band ens": ens, "T3 12m mom": (d.px / d.px.shift(252) > 1).astype(float)}
    R = {k: run(w, d, lag=1, cost_bp=10, borrow_spread=0.01)[0] for k, w in W.items()}
    R["E"] = run(rule_e(ens, idx), d, lag=0, cost_bp=10, borrow_spread=0.01)[0]
    start = idx[260]
    for per, a in [("full", start), ("2008-26", max(start, pd.Timestamp("2008-01-01")))]:
        for k, r in R.items():
            s = stats(r.loc[a:], d.rf); rows.append({"market": name, "period": per, "rule": k, "start": a.year, **{m: s[m] for m in ["cagr", "sharpe", "maxdd"]}})
    for k, r in R.items(): EX.setdefault(k, {})[name] = (r - d.rf.reindex(r.index)).loc[start:]
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for per in ["full", "2008-26"]:
    t = T[T.period == per].pivot(index="market", columns="rule", values=["sharpe", "maxdd", "cagr"])
    bh = t.xs("buy & hold", axis=1, level=1)
    print(f"\n=== {per}: markets where the rule beats buy & hold (of {len(bh)}) ===")
    for k in ["T1 sma200", "T2 band ens", "T3 12m mom", "E"]:
        s_ = (t[("sharpe", k)] > bh["sharpe"]).sum(); d_ = (t[("maxdd", k)] > bh["maxdd"]).sum(); c_ = (t[("cagr", k)] > bh["cagr"]).sum()
        print(f"  {k:12s} sharpe {s_:2d}   drawdown {d_:2d}   cagr {c_:2d}   | median sharpe {t[('sharpe', k)].median():.2f} vs b&h {bh['sharpe'].median():.2f}, median dd {t[('maxdd', k)].median():.1%} vs {bh['maxdd'].median():.1%}")
print("\nper market, 2008-26, sharpe  (b&h / T2 / E):")
t = T[T.period == "2008-26"].pivot(index="market", columns="rule", values="sharpe")
print(t[["buy & hold", "T1 sma200", "T2 band ens", "T3 12m mom", "E"]].round(2).to_string())
usrf = _fred("DTB3") / 100 / TD
print("\nglobal equal-weight, currency-hedged approximation:")
for k in ["buy & hold", "T2 band ens", "E"]:
    g = pd.DataFrame(EX[k]).loc["1995":]
    gr = g.mean(axis=1) + usrf.reindex(g.index).ffill().fillna(0)
    for per, a in [("1995-2007", "1995"), ("2008-26", "2008")]:
        s = stats(gr.loc[a:] if per == "2008-26" else gr.loc[a:"2007"], usrf.reindex(gr.index).ffill().fillna(0))
        print(f"  {k:12s} {per}: cagr {s['cagr']:.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:.1%}")
