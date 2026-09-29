"""two layers on rule E (market), declared together, frozen before running, no retuning after:
  C bond carry : + 0.5 x (10y bond - t-bill) only while the curve is steep (DGS10 - DTB3 > 1.0 pt at the prior close);
                 bond = synthetic 10y from dgs10 (as bond_overlay.py), 1962+. holdout 1963-2007.
  V vix stress : base = ens x 0.5 while vix (VIXCLS) > its trailing 252-day 90th percentile (prior close). 1990+. holdout 1991-2007.
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in the holdout AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
from strat_f import french
from strat_bc import rule_e

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, idx), d, lag=0, borrow_spread=0.01)
y = _fred("DGS10").reindex(idx).ffill() / 100
dmod = (1 - (1 + y / 2) ** -20) / y
br = (y.shift(1) / TD - dmod.shift(1) * y.diff())
steep = ((_fred("DGS10") - _fred("DTB3")).reindex(idx).ffill() > 1.0).astype(float).shift(1)
rC = (rE + 0.5 * steep * (br - d.rf)).loc["1963":]
vix = _fred("VIXCLS").reindex(idx).ffill()
hi = (vix > vix.rolling(252, min_periods=200).quantile(0.9)).astype(float).where(vix.rolling(252, min_periods=200).count() >= 200)
rV, _ = run(rule_e(ens * (1 - 0.5 * hi.fillna(0)), idx), d, lag=0, borrow_spread=0.01)
assert steep.loc["1963":].notna().all() and hi.loc["1991":].notna().all()
for name, rX, hold in [("C bond carry", rC, ("1963-01-01", "2007-12-31")), ("V vix stress", rV, ("1991-01-01", "2007-12-31"))]:
    ok, line = True, []
    for e, (a, b) in {"holdout": hold, "2008-26": ("2008-01-01", None), "2022 (info)": ("2022-01-01", "2022-12-31")}.items():
        s0, s1 = stats(rE.loc[a:b], d.rf), stats(rX.loc[a:b], d.rf)
        pt, lo, hi_, _ = sharpe_diff_ci(rX.loc[a:b], rE.loc[a:b], d.rf, n=1000)
        line.append(f"{e}: E {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%} dSh {pt:+.2f} [{lo:+.2f},{hi_:+.2f}]")
        if "info" not in e: ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {name}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
print(f"steep curve on {steep.loc['1963':].mean():.0%} of days; vix stress on {hi.loc['1991':].mean():.0%}")
