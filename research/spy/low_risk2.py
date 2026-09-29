"""lower-risk versions, judged in 4 eras incl. 2017-2026 (where the 3x rule trails spy on sharpe).
frozen before running: pass = cagr above spy AND max drawdown smaller than spy in EVERY era.
candidates (only leverage scale, calendar layers already tested; no new signals):
  A  ens x season (2x winter / 1x summer), no tom                          max 2x
  B  A x pre-holiday, cap 2                                                 max 2x
  C  ens x (1.5 winter / 1 summer) x tom x pre-holiday, cap 2               max 2x
  D  floor: (0.25 + 0.75 ens) x (1.5 winter / 1 summer), cap 1.5           never fully out; max 1.5x
  E  ens x tom x pre-holiday, cap 1.5, no season                            max 1.5x"""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import tom_days

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
nxt = idx[1:].values.astype("datetime64[D]")
ph = pd.Series(np.r_[np.busday_count(idx[:-1].values.astype("datetime64[D]"), nxt) > 1, False], index=idx).astype(float)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx).astype(float)
tom = tom_days(idx, 1, 3)
sh = lambda w: w.shift(1).fillna(0)
cfg = {
    "spy (market b&h)": sh(pd.Series(1.0, index=idx)),
    "current best, cap 3": (sh(ens * (1 + winter)) * (1 + tom) * (1 + ph)).clip(upper=3),
    "A ens x season": sh(ens * (1 + winter)),
    "B A x pre-holiday, cap 2": (sh(ens * (1 + winter)) * (1 + ph)).clip(upper=2),
    "C 1.5/1 season x tom x ph, cap 2": (sh(ens * (1 + 0.5 * winter)) * (1 + tom) * (1 + ph)).clip(upper=2),
    "D floor .25, 1.5/1 season, cap 1.5": sh((0.25 + 0.75 * ens) * (1 + 0.5 * winter)).clip(upper=1.5),
    "E ens x tom x ph, cap 1.5": (sh(ens) * (1 + tom) * (1 + ph)).clip(upper=1.5),
}
E = {"1927-55": ("1927-07-01", "1955-12-31"), "1956-93": ("1956-01-01", "1993-12-31"), "1994-16": ("1994-01-01", "2016-12-31"), "2017-26": ("2017-01-01", None)}
rows = []
for name, w in cfg.items():
    r, _ = run(w, d, lag=0, borrow_spread=0.01)
    row = {"name": name}
    for e, (a, b) in E.items():
        s = stats(r.loc[a:b], d.rf); row |= {f"{e} cagr": s["cagr"], f"{e} sh": s["sharpe"], f"{e} dd": s["maxdd"]}
    rows.append(row)
df = pd.DataFrame(rows).set_index("name")
m = df.loc["spy (market b&h)"]
df["pass"] = [all(df.loc[k, f"{e} cagr"] > m[f"{e} cagr"] and df.loc[k, f"{e} dd"] > m[f"{e} dd"] for e in E) for k in df.index]
pd.set_option("display.width", 250)
print(df.to_string(formatters={c: ("{:.2f}" if c.endswith("sh") else "{:.1%}").format for c in df if c != "pass"}))
