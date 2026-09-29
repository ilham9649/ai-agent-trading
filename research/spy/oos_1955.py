"""out-of-sample: settings were chosen on 1994+. run on ^gspc 1955-1993. price index + assumed 4% dividend yield (approximation, applied to spy and strategy alike)."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from engine import _fred

g = yf.download("^GSPC", start="1950-01-01", end="1994-03-01", auto_adjust=True, progress=False)["Close"].squeeze()
DY = 0.04
d = pd.DataFrame({"px": g}); d["r"] = d.px.pct_change() + DY / TD
d["rf"] = (_fred("DTB3").reindex(d.index).ffill() / 100 / TD)
d = d.loc["1955-01-01":].dropna()
d["pxt"] = (1 + d.r).cumprod()          # synthetic total-return level for the trend signal
S = "1956-01-01"
ens = (band(d.pxt, 150, .03) + band(d.pxt, 200, .03) + band(d.pxt, 250, .03)) / 3
rows = []
for name, w in [("spy-equivalent b&h", pd.Series(1.0, index=d.index)), ("banded 200 3% 2x", 2 * band(d.pxt, 200, .03)), ("ensemble 2x", 2 * ens), ("ensemble 1.5x", 1.5 * ens), ("trend200 no band 2x", 2 * (d.pxt > d.pxt.rolling(200).mean()))]:
    r, _ = run(w, d, borrow_spread=0.01); r = r.loc[S:]
    a = stats(r, d.rf); mid = r.index[len(r) // 2]
    rows.append(dict(name=name, cagr=a["cagr"], vol=a["vol"], sharpe=a["sharpe"], maxdd=a["maxdd"], h1=stats(r.loc[:mid], d.rf)["cagr"], h2=stats(r.loc[mid:], d.rf)["cagr"]))
df = pd.DataFrame(rows).set_index("name")
print(d.index.min().date(), d.index.max().date(), "mean rf %.1f%%" % (d.rf.mean() * TD * 100))
print(df.to_string(formatters={c: "{:.1%}".format for c in ["cagr", "vol", "maxdd", "h1", "h2"]} | {"sharpe": "{:.2f}".format}))

print()
b = d.r.loc[S:]
for name, w in [("banded 200 3% 2x", 2 * band(d.pxt, 200, .03)), ("ensemble 2x", 2 * ens), ("trend200 no band 2x", 2 * (d.pxt > d.pxt.rolling(200).mean()))]:
    r, _ = run(w, d, borrow_spread=0.01); r = r.loc[S:]
    pt, lo, hi, p = sharpe_diff_ci(r, b, d.rf); a, t, beta = alpha(r, b, d.rf)
    print(f"{name:22s} dSharpe {pt:+.2f} ci [{lo:+.2f},{hi:+.2f}] p={p:.2f}  alpha {a:.1%} t={t:.1f} beta {beta:.2f}")
