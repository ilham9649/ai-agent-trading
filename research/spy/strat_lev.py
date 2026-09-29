"""goal: cagr above spy at similar or lower sharpe. leveraged spy, with and without trend/vol filters. borrow = rf + 1%."""
import numpy as np, pandas as pd
from engine import *
from strat_v import vm_weights

d = load("1993-02-01")
S = "1994-03-01"
one = pd.Series(1.0, index=d.index)
trend = (d.px > d.px.rolling(200).mean()).astype(float)
cfg = {"spy b&h": one}
for L in [1.25, 1.5, 2.0]:
    cfg[f"const {L}x"] = one * L
    cfg[f"trend200 {L}x"] = trend * L
for sig in [0.15, 0.20]:
    cfg[f"vol-target {sig:.0%} cap2 + trend"] = vm_weights(d, sig, 2.0, "trend", 21)
    cfg[f"vol-target {sig:.0%} cap2"] = vm_weights(d, sig, 2.0, "none", 63)
rows = []
for name, w in cfg.items():
    r, pos = run(w, d, borrow_spread=0.01)
    r = r.loc[S:]
    a, h1, h2 = stats(r, d.rf), stats(r.loc[:"2009-12-31"], d.rf), stats(r.loc["2010-01-01":], d.rf)
    rows.append(dict(name=name, cagr=a["cagr"], vol=a["vol"], sharpe=a["sharpe"], maxdd=a["maxdd"], cagr_h1=h1["cagr"], cagr_h2=h2["cagr"], dd_h1=h1["maxdd"], dd_h2=h2["maxdd"]))
df = pd.DataFrame(rows).set_index("name")
print(df.to_string(formatters={c: "{:.1%}".format for c in df if c != "sharpe"} | {"sharpe": "{:.2f}".format}))
