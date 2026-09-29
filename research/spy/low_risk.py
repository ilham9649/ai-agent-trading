"""lower-risk versions of the same rule: only the leverage scale changes (no new signals). french market 1927-2026, saturdays merged."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days
from industry import merge_sat

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
winter = pd.Series(d.index.month.isin([11, 12, 1, 2, 3, 4]), index=d.index).astype(float)
tom = tom_days(d.index, 1, 3)
sh = lambda w: w.shift(1).fillna(0)          # all weights decided at prior close; calendar known ahead
cfg = {
    "market b&h": sh(pd.Series(1.0, index=d.index)),
    "current best (max 3x)": (sh(ens * (1 + winter)) * (1 + tom)).clip(upper=3),
    "max 2x: season, no tom": sh(ens * (1 + winter)),
    "max 1.5x: tom 1.5x, season off": sh(ens) * (1 + 0.5 * tom),
    "max 1x: trend ensemble only": sh(ens),
    "max 1x: ens x (1 winter / 0.5 summer)": sh(ens * (0.5 + 0.5 * winter)),
    "max 0.75x: ens 0.75": sh(0.75 * ens),
}
E = {"1927-55": ("1927-07-01", "1955-12-31"), "1956-93": ("1956-01-01", "1993-12-31"), "1994-26": ("1994-01-01", None), "all": ("1927-07-01", None)}
rows = []
for name, w in cfg.items():
    r, _ = run(w, d, lag=0, borrow_spread=0.01)
    row = {"name": name}
    for e, (a, b) in E.items():
        s = stats(r.loc[a:b], d.rf); row |= {f"{e} cagr": s["cagr"], f"{e} sh": s["sharpe"], f"{e} dd": s["maxdd"]}
    rows.append(row)
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250)
print(df.to_string(formatters={c: ("{:.2f}" if c.endswith("sh") else "{:.1%}").format for c in df}))
