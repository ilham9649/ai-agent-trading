"""frozen rule (band ensemble 150/200/250 +-3%, 2x nov-apr / 1x may-oct) on other countries' price indexes. no retuning.
price-only, local currency, us t-bill as cash. compares each to its own buy-and-hold."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred

IDX = {"^GSPC": "us", "^N225": "japan", "^FTSE": "uk", "^GDAXI": "germany", "^FCHI": "france", "^HSI": "hong kong", "^AXJO": "australia", "^GSPTSE": "canada", "^STI": "singapore", "^KS11": "korea"}
rf_all = _fred("DTB3") / 100 / TD


def rule(px):
    ens = (band(px, 150, .03) + band(px, 200, .03) + band(px, 250, .03)) / 3
    winter = pd.Series(px.index.month.isin([11, 12, 1, 2, 3, 4]), index=px.index).astype(float)
    return ens * (1 + winter)


rows, rets, bhs = [], {}, {}
for tk, name in IDX.items():
    px = yf.download(tk, start="1960-01-01", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    px = px[px > 0]
    d = pd.DataFrame({"px": px}); d["r"] = d.px.pct_change(); d["rf"] = rf_all.reindex(d.index).ffill().fillna(0)
    d = d.iloc[1:]
    r, _ = run(rule(d.px), d, borrow_spread=0.01)
    r = r.iloc[260:]; b = d.r.loc[r.index]
    s, sb = stats(r, d.rf), stats(b, d.rf)
    pt, lo, hi, p = sharpe_diff_ci(r, b, d.rf, n=500)
    w = pd.Series(b.index.month.isin([11, 12, 1, 2, 3, 4]), index=b.index)
    ex = b - d.rf.loc[b.index]
    rows.append(dict(market=name, start=r.index[0].year, bh_cagr=sb["cagr"], bh_sh=sb["sharpe"], rule_cagr=s["cagr"], rule_sh=s["sharpe"], rule_dd=s["maxdd"], bh_dd=sb["maxdd"], dSh=pt, p=p, win=ex[w].mean() * TD, sum=ex[~w].mean() * TD))
    rets[name], bhs[name] = r, b
df = pd.DataFrame(rows).set_index("market")
f = {c: "{:.1%}".format for c in ["bh_cagr", "rule_cagr", "rule_dd", "bh_dd", "win", "sum"]} | {c: "{:.2f}".format for c in ["bh_sh", "rule_sh", "dSh", "p"]}
print(df.to_string(formatters=f))
print("\nmarkets where rule sharpe > b&h sharpe: %d/%d; cagr > b&h: %d/%d; winter>summer excess: %d/%d" % ((df.rule_sh > df.bh_sh).sum(), len(df), (df.rule_cagr > df.bh_cagr).sum(), len(df), (df.win > df["sum"]).sum(), len(df)))
ex_us = df.drop("us")
print("ex-us: sharpe better %d/%d, cagr better %d/%d, dd better %d/%d" % ((ex_us.rule_sh > ex_us.bh_sh).sum(), len(ex_us), (ex_us.rule_cagr > ex_us.bh_cagr).sum(), len(ex_us), (ex_us.rule_dd > ex_us.bh_dd).sum(), len(ex_us)))

# ---- equal-weight portfolio of the frozen rule across markets (local-currency, price-only: currency and dividends ignored)
R = pd.DataFrame(rets).sort_index()
B = pd.DataFrame(bhs).sort_index()
for a in ["1995-01-01"]:
    Rp, Bp = R.loc[a:], B.loc[a:]
    cols = [c for c in Rp if Rp[c].first_valid_index() is not None and Rp[c].first_valid_index() <= pd.Timestamp("1996-01-01")]
    pr = Rp[cols].fillna(0).mean(axis=1)   # closed-market days earn 0 (approximation)
    pb = Bp[cols].fillna(0).mean(axis=1)
    us = Rp["us"].fillna(0)
    rfx = rf_all.reindex(pr.index).ffill().fillna(0)
    print(f"\nequal-weight of {cols} since {a[:4]}")
    for name, x in [("us b&h (price)", Bp["us"].fillna(0)), ("us rule", us), ("ew b&h, all markets", pb), ("ew rule, all markets", pr)]:
        s = stats(x, rfx)
        print(f"{name:24s} cagr {s['cagr']:.1%} vol {s['vol']:.1%} sharpe {s['sharpe']:.2f} sortino {s['sortino']:.2f} maxdd {s['maxdd']:.1%}")
    pt, lo, hi, p = sharpe_diff_ci(pr, Bp["us"].fillna(0), rfx, n=1000)
    print(f"ew rule vs us b&h: dSharpe {pt:+.2f} ci [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
    pt, lo, hi, p = sharpe_diff_ci(pr, us, rfx, n=1000)
    print(f"ew rule vs us rule: dSharpe {pt:+.2f} ci [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
