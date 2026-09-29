"""71-year pooled series (1955-93 ^gspc + assumed 4% dividend, 1994+ spy total return) and signal-family test, judged per era."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred


def pooled():
    g = yf.download("^GSPC", start="1950-01-01", end="1994-02-01", auto_adjust=True, progress=False)["Close"].squeeze()
    old = pd.DataFrame({"r": g.pct_change() + 0.04 / TD}).loc["1955-01-01":]
    new = load("1993-02-01")[["r"]].loc["1994-02-01":]
    d = pd.concat([old, new]).dropna()
    d["px"] = (1 + d.r).cumprod()
    d["rf"] = (_fred("DTB3").reindex(d.index).ffill() / 100 / TD)
    return d.dropna()


def signals(px):
    s = {}
    for n in [100, 150, 200, 250]:
        s[f"sma{n}"] = (px > px.rolling(n).mean()).astype(float)
    for m in [63, 126, 252]:
        s[f"mom{m}"] = (px > px.shift(m)).astype(float)
    for n in [150, 200, 250]:
        s[f"band{n}"] = band(px, n, .03)
    return pd.DataFrame(s)


ERAS = {"1956-93": ("1956-01-01", "1993-12-31"), "1994-26": ("1994-02-01", None)}


def evaluate(cfg, d):
    rows = []
    for name, w in cfg.items():
        r, _ = run(w, d, borrow_spread=0.01)
        row = {"name": name}
        for e, (a, b) in ERAS.items():
            s = stats(r.loc[a:b], d.rf)
            row.update({f"{e} cagr": s["cagr"], f"{e} sh": s["sharpe"], f"{e} dd": s["maxdd"]})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("name")
    f = {c: "{:.1%}".format for c in df if c.endswith(("cagr", "dd"))} | {c: "{:.2f}".format for c in df if c.endswith("sh")}
    print(df.to_string(formatters=f))


if __name__ == "__main__":
    d = pooled(); S = signals(d.px)
    print(d.index.min().date(), d.index.max().date(), len(d))
    band_ens = S[["band150", "band200", "band250"]].mean(axis=1)
    sma_ens = S[["sma100", "sma150", "sma200", "sma250"]].mean(axis=1)
    mom_ens = S[["mom63", "mom126", "mom252"]].mean(axis=1)
    all_ens = S[["sma150", "sma200", "sma250", "mom126", "mom252"]].mean(axis=1)
    cfg = {
        "spy b&h": pd.Series(1.0, index=d.index),
        "plain sma200 2x": 2 * S.sma200,
        "band ensemble 2x (ref)": 2 * band_ens,
        "sma ensemble 2x": 2 * sma_ens,
        "mom ensemble 2x": 2 * mom_ens,
        "mom252 2x": 2 * S.mom252,
        "sma200 & mom252 both 2x": 2 * S.sma200 * S.mom252,
        "sma+mom vote 2x": 2 * all_ens,
        "sma+mom vote 2x, -0.5x short when 0": 2 * all_ens - 0.5 * (1 - all_ens),
        "sma+mom vote, 0-100% -> 0..2.5x": 2.5 * all_ens,
        "sma+mom vote, exit-only-if>=2/5 off": 2 * (all_ens >= 0.4).astype(float) * all_ens.clip(lower=0.6),
    }
    evaluate(cfg, d)
