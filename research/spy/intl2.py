"""(corrected timing) frozen turn-of-month boost on other countries (no retuning): base rule vs base x tom boost (x2, cap 3)."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred
from strat_bc import tom_days
IDX = {"^GSPC": "us", "^N225": "japan", "^FTSE": "uk", "^GDAXI": "germany", "^FCHI": "france", "^HSI": "hong kong", "^AXJO": "australia", "^GSPTSE": "canada", "^STI": "singapore", "^KS11": "korea"}
rf_all = _fred("DTB3") / 100 / TD

def base_rule(px):
    ens = (band(px, 150, .03) + band(px, 200, .03) + band(px, 250, .03)) / 3
    winter = pd.Series(px.index.month.isin([11, 12, 1, 2, 3, 4]), index=px.index).astype(float)
    return ens * (1 + winter)

rows = []
for tk, name in IDX.items():
    px = yf.download(tk, start="1960-01-01", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    px = px[px > 0]
    d = pd.DataFrame({"px": px}); d["r"] = d.px.pct_change(); d["rf"] = rf_all.reindex(d.index).ffill().fillna(0); d = d.iloc[1:]
    b = base_rule(d.px)
    w = (b.shift(1).fillna(0) * (1 + tom_days(d.index, 1, 3))).clip(upper=3.0)  # calendar known in advance: no extra lag
    r0, _ = run(b, d, borrow_spread=.01); r1, _ = run(w, d, lag=0, borrow_spread=.01)
    r0, r1 = r0.iloc[260:], r1.iloc[260:]
    s0, s1, sb = stats(r0, d.rf), stats(r1, d.rf), stats(d.r.loc[r0.index], d.rf)
    tm = tom_days(d.index, 1, 3).loc[r0.index] == 1
    ex = d.r.loc[r0.index] - d.rf.loc[r0.index]
    rows.append(dict(market=name, bh_sh=sb["sharpe"], base_sh=s0["sharpe"], tom_sh=s1["sharpe"], base_cagr=s0["cagr"], tom_cagr=s1["cagr"], tom_dd=s1["maxdd"], base_dd=s0["maxdd"], tom_days=ex[tm].mean() * TD, other=ex[~tm].mean() * TD))
df = pd.DataFrame(rows).set_index("market")
f = {c: "{:.1%}".format for c in ["base_cagr", "tom_cagr", "tom_dd", "base_dd", "tom_days", "other"]} | {c: "{:.2f}".format for c in ["bh_sh", "base_sh", "tom_sh"]}
print(df.to_string(formatters=f))
print("\ntom boost improves sharpe in %d/%d markets, cagr in %d/%d; tom-day excess > other-day excess in %d/%d" % ((df.tom_sh > df.base_sh).sum(), len(df), (df.tom_cagr > df.base_cagr).sum(), len(df), (df.tom_days > df.other).sum(), len(df)))
ex_us = df.drop("us"); print("ex-us: sharpe better %d/%d, cagr better %d/%d; mean sharpe base %.2f -> tom %.2f (b&h %.2f)" % ((ex_us.tom_sh > ex_us.base_sh).sum(), len(ex_us), (ex_us.tom_cagr > ex_us.base_cagr).sum(), len(ex_us), ex_us.base_sh.mean(), ex_us.tom_sh.mean(), ex_us.bh_sh.mean()))
