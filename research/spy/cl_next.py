"""next ideas on CL (rule C on the low-vol industry basket), 3 configs declared together, frozen before running:
  S1 season switch : asset = market in nov-apr, low-vol basket in may-oct (weights = rule C)
  S2 lowvol + mom  : among the 20 lowest 60d-variance industries, the 10 with the best 12-1 month return
  S3 CL + carry    : CL + 0.5 x (synthetic 10y - t-bill) while DGS10 - DTB3 > 1 pt (as carry_vix.py); 1963+ (old period = 1963-2007 only)
  pass = cagr ABOVE CL's AND max drawdown no worse than CL's, in 1927-62 (not S3), 1963-2007 AND 2008-2026."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred
from strat_f import french
from strat_bc import rule_c

ind = french("49_Industry_Portfolios").mask(lambda x: x <= -0.99)
ff = french("F-F_Research_Data_Factors")
x = merge_sat(ind.join(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}), how="inner"))
R, d = x[ind.columns], x[["r", "rf"]].copy()
d["px"] = (1 + d.r).cumprod()
idx = d.index
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, index=idx)
var = R.rolling(60, min_periods=40).var()
lvl = (1 + R.fillna(0)).cumprod().where(R.notna())
mom = lvl.shift(21) / lvl.shift(252) - 1

def basket(sel_me):
    hold = sel_me.reindex(idx).shift(1).ffill().fillna(False).astype(bool)
    return (R.where(hold).mean(axis=1) - hold.astype(float).diff().abs().sum(axis=1).fillna(0) / 20 * 2e-4).fillna(0), hold

lv_rank = var[me].rank(axis=1, method="first")
b_cl, _ = basket(lv_rank <= 10)
b_s2, h_s2 = basket(mom[me].where(lv_rank <= 20).rank(axis=1, ascending=False, method="first") <= 10)
winter = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]), index=idx)
b_s1 = d.r.where(winter, b_cl)
wC = rule_c(ens, idx)
rCL, _ = run(wC, d.assign(r=b_cl), lag=0, borrow_spread=0.01)
y = _fred("DGS10").reindex(idx).ffill() / 100
dmod = (1 - (1 + y / 2) ** -20) / y
br = y.shift(1) / TD - dmod.shift(1) * y.diff()
steep = ((_fred("DGS10") - _fred("DTB3")).reindex(idx).ffill() > 1.0).astype(float).shift(1)
cfg = {"S1 season switch": run(wC, d.assign(r=b_s1), lag=0, borrow_spread=0.01)[0],
       "S2 lowvol + mom": run(wC, d.assign(r=b_s2), lag=0, borrow_spread=0.01)[0],
       "S3 CL + carry": (rCL + 0.5 * steep * (br - d.rf)).loc["1963":]}
assert (h_s2.sum(axis=1).loc["1928":] == 10).all()
for k, rX in cfg.items():
    ok, line = True, []
    per = {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}
    if k.startswith("S3"): per.pop("1927-62")
    for e, (a, bb) in per.items():
        s0, s1 = stats(rCL.loc[a:bb], d.rf), stats(rX.loc[a:bb], d.rf)
        line.append(f"{e}: CL {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
        ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
    print(f"##### {k}: {'ADOPT' if ok else 'REJECT'}\n  " + "\n  ".join(line))

# after the frozen verdict: S2 turnover and cost stress (CL at 2 bp as the reference)
to = h_s2.astype(float).diff().abs().sum(axis=1) / 20
_, h_cl = basket(lv_rank <= 10)
tcl = h_cl.astype(float).diff().abs().sum(axis=1) / 20
print(f"\nturnover per rebalance: S2 {to[to > 0].mean():.0%}, CL {tcl[tcl > 0].mean():.0%}")
for bp in [10, 25]:
    bs = (R.where(h_s2).mean(axis=1) - to.fillna(0) * bp * 1e-4).fillna(0)
    rs = run(wC, d.assign(r=bs), lag=0, borrow_spread=0.01)[0]
    print(f"S2 at {bp} bp: " + "  ".join(f"{e} {stats(rs.loc[a:b], d.rf)['cagr']:.1%}/{stats(rs.loc[a:b], d.rf)['sharpe']:.2f}/{stats(rs.loc[a:b], d.rf)['maxdd']:.1%}"
                                    for e, (a, b) in {"1927-62": ("1927-07-01", "1962-12-31"), "1963-2007": ("1963-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items()))
