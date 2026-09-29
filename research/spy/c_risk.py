"""risk controls on user-chosen rule C, declared together, frozen before running, no retuning after:
  D1 drawdown control: while the strategy is > 10% below its own peak, cap exposure at 1.0x; release when back within 5% of the peak
     (state uses returns up to the prior close only)
  D2 double confirmation: leverage above 1x only when the market's 252-day return is also > 0; otherwise cap at 1.0x
  pass = vs C, dSharpe >= +0.03 AND max drawdown no worse than C, in 1927-62, 1963-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_c

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
wC = rule_c(ens, idx)
rC, _ = run(wC, d, lag=0, borrow_spread=0.01)

def dd_control(w, trig=0.10, rel=0.05, cap=1.0, cost_bp=2.0, spread=0.01):
    r, rf, wv = d.r.values, d.rf.values, w.values
    eq, peak, on, prev, out, pos = 1.0, 1.0, False, 0.0, np.zeros(len(r)), np.zeros(len(r))
    for t in range(len(r)):
        p = min(wv[t], cap) if on else wv[t]                     # decided from equity up to t-1
        rt = p * r[t] + max(1 - p, 0) * rf[t] + min(1 - p, 0) * (rf[t] + spread / TD) - abs(p - prev) * cost_bp / 1e4
        out[t], pos[t], prev = rt, p, p
        eq *= 1 + rt; peak = max(peak, eq)
        if not on and eq < peak * (1 - trig): on = True
        elif on and eq >= peak * (1 - rel): on = False
    return pd.Series(out, index=idx).iloc[1:], pd.Series(pos, index=idx)

rD1, pD1 = dd_control(wC)
r0, _ = run(wC, d, lag=0, borrow_spread=0.01)
assert abs(dd_control(wC, trig=9.9)[0] - r0).max() < 1e-12            # with the control never on, it equals run()
up = (d.px / d.px.shift(252) - 1 > 0).astype(float).shift(1).fillna(0)
wD2 = wC.where(up == 1, wC.clip(upper=1.0))
rD2, _ = run(wD2, d, lag=0, borrow_spread=0.01)
print(f"D1 control on {(pD1 < wC).mean():.0%} of days; D2 caps leverage on {((wD2 < wC)).mean():.0%} of days")
for name, rX in [("D1 drawdown control", rD1), ("D2 double confirmation", rD2)]:
    ok, line = True, []
    for e, (a, b) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
        s0, s1 = stats(rC.loc[a:b], d.rf), stats(rX.loc[a:b], d.rf)
        pt, lo, hi, _ = sharpe_diff_ci(rX.loc[a:b], rC.loc[a:b], d.rf, n=1000)
        line.append(f"{e}: C {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%} dSh {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ok &= pt >= 0.03 and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {name}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))
