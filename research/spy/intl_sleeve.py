"""international diversification next to R4. ex-us = french Developed_ex_US_3_Factors daily (Mkt-RF + RF, usd, 1990-07+).
frozen before running, no retuning after:
  intl rule = rule E (own band ensemble x tom x pre-holiday, cap 1.5) on the ex-us market
  combo = R4 + 0.25 x (intl rule - t-bill)
  pass = cagr ABOVE R4 AND max drawdown no worse than R4, in 1991-2007 AND 2008-2026."""
import io, zipfile, requests, numpy as np, pandas as pd
from engine import *
from strat_bc import rule_e
import s2_rates2 as r

p = D / "french_Developed_ex_US_3_Factors_daily.csv"
if not p.exists():
    z = zipfile.ZipFile(io.BytesIO(requests.get("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_3_Factors_Daily_CSV.zip",
                                                headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content))
    L = [l.split(",") for l in z.read(z.namelist()[0]).decode("latin1").splitlines() if l.strip()[:8].isdigit()]
    df = pd.DataFrame([[c.strip() for c in l] for l in L]).set_index(0).astype(float).div(100)
    df.index = pd.to_datetime(df.index, format="%Y%m%d"); df.columns = ["Mkt-RF", "SMB", "HML", "RF"]; df.to_csv(p)
x = pd.read_csv(p, index_col=0, parse_dates=True)
x = x[(x > -0.99).all(axis=1)]
di = pd.DataFrame({"r": x["Mkt-RF"] + x["RF"], "rf": x["RF"]}); di["px"] = (1 + di.r).cumprod()
ensi = (band(di.px, 150, .03) + band(di.px, 200, .03) + band(di.px, 250, .03)) / 3
ri = run(rule_e(ensi, di.index), di, lag=0, borrow_spread=0.01)[0]
# align to us trading days (compound non-us days into the next us day)
us = r.idx
g = pd.Series(us.searchsorted(ri.index), index=ri.index); ok_ = g < len(us)
ri_us = (1 + ri[ok_]).groupby(us[g[ok_]]).prod().reindex(us).fillna(1) - 1
rR4 = r.cfg["R4 R3 at half in hikes"]
combo = (rR4 + 0.25 * (ri_us - r.d.rf)).loc["1991":]
assert x.index[0] < pd.Timestamp("1990-08-01")
print(f"corr(R4, intl rule) since 1991: {rR4.loc['1991':].corr(ri_us.loc['1991':]):+.2f}")
ok = True
for e, (a, b) in {"1991-2007": ("1991-01-01", "2007-12-31"), "2008-26": ("2008-01-01", None)}.items():
    s0, s1, si = stats(rR4.loc[a:b], r.d.rf), stats(combo.loc[a:b], r.d.rf), stats(ri_us.loc[a:b], r.d.rf)
    print(f"{e}: intl rule alone {si['cagr']:.1%}/{si['sharpe']:.2f}/{si['maxdd']:.1%}   R4 {s0['cagr']:.1%}/{s0['sharpe']:.2f}/{s0['maxdd']:.1%} -> + 0.25 intl {s1['cagr']:.1%}/{s1['sharpe']:.2f}/{s1['maxdd']:.1%}")
    ok &= s1["cagr"] > s0["cagr"] and s1["maxdd"] >= s0["maxdd"]
print("ADOPT" if ok else "REJECT")
