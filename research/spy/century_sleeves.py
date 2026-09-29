"""single aqr "century of factor premia" series (monthly, 1926+) as overlays on chosen rule E. run: uv run --with openpyxl python century_sleeves.py
frozen before running, no retuning after (6 configs declared together):
  sleeves = Commodities Market, Fixed income Market, All Macro Carry, All Macro Momentum, All Macro Value, All Macro Defensive
  overlay = E monthly + 0.5 x (sleeve excess return - 2%/yr haircut)
  pass    = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 1927-62, 1963-2007 AND 2008-2026 (to 2026-02)."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e
from macro_sleeve import mstats, boot

c = pd.read_excel(D / "Century-of-Factor-Premia-Monthly.xlsx", "Century of Factor Premia", header=18)
c = c[pd.to_datetime(c.Date, errors="coerce").notna()]
c.index = pd.to_datetime(c.Date).dt.to_period("M")
names = ["Commodities Market", "Fixed income Market", "All Macro Carry", "All Macro Momentum", "All Macro Value", "All Macro Defensive"]
ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, d.index), d, lag=0, borrow_spread=0.01)
M = lambda r: (1 + r).groupby(r.index.to_period("M")).prod() - 1
eM, rfM = M(rE), M(d.rf)
H = {"1927-62": ("1927-07", "1962-12"), "1963-2007": ("1963-01", "2007-12"), "2008-26": ("2008-01", None)}
rows = []
for n in names:
    s = pd.to_numeric(c[n], errors="coerce")
    ov = (eM + 0.5 * (s - 0.02 / 12)).dropna()
    row, ok = {"sleeve": n, "start": str(s.dropna().index[0]), "corr E": round(pd.concat([s, eM], axis=1).dropna().corr().iloc[0, 1], 2)}, True
    for e, (a, b) in H.items():
        x = ov.loc[a:b]; eb = eM.loc[x.index[0]:x.index[-1]]; rf = rfM.loc[x.index]
        s0, s1 = mstats(eb, rf), mstats(x, rf); dS = s1["sh"] - s0["sh"]; lo, hi = boot(x, eb, rf)
        row[f"{e} dSh"] = f"{dS:+.2f} [{lo:+.2f},{hi:+.2f}]"; row[f"{e} dd"] = f"{s0['dd']:.1%}->{s1['dd']:.1%}"
        ok &= dS >= 0.03 and s1["dd"] >= s0["dd"]
    row["verdict"] = "ADOPT" if ok else "REJECT"; rows.append(row)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40)
print(pd.DataFrame(rows).set_index("sleeve").to_string())
