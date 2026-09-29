"""low-volatility anomaly: hold the lowest-variance quintile of us stocks (french Portfolios_Formed_on_VAR, value-weighted
"Lo 20", monthly, 1963-07+) instead of the market, timed by the market trend ensemble at each month end.
frozen before running, no retuning after:
  L1   = ens(prior month end) x low-var excess return + t-bill                      (max 1x)
  L15  = 1.5 x ens(prior month end) x low-var excess, borrow at t-bill + 1%         (max 1.5x, like E's cap)
  pass = vs E (daily rule compounded monthly), dSharpe >= +0.03 AND max drawdown no worse than E, in 1964-2007 AND 2008-2026."""
import io, zipfile, requests, numpy as np, pandas as pd
from engine import *
from strat_f import french
from strat_bc import rule_e

p = D / "french_VAR_monthly.csv"
if not p.exists():
    z = zipfile.ZipFile(io.BytesIO(requests.get("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_VAR_CSV.zip",
                                                headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content))
    L = z.read(z.namelist()[0]).decode("latin1").splitlines()
    i = next(k for k, l in enumerate(L) if l.strip()[:6].isdigit()); hdr = [h.strip() for h in L[i - 1].split(",")]
    rows = [l.split(",") for l in L[i:] if l.strip()[:6].isdigit() and len(l.strip().split(",")[0]) == 6]
    rows = rows[:next(k for k in range(1, len(rows)) if rows[k][0].strip() <= rows[k - 1][0].strip())]   # first block only (value-weighted)
    df = pd.DataFrame(rows, columns=["date"] + hdr[1:]).set_index("date").astype(float).div(100)
    df.index = pd.PeriodIndex([f"{s.strip()[:4]}-{s.strip()[4:]}" for s in df.index], freq="M"); df.to_csv(p)
lv = pd.read_csv(p, index_col=0)["Lo 20"]; lv.index = pd.PeriodIndex(lv.index, freq="M")
assert lv.index[0] == pd.Period("1963-07", "M") and lv.abs().max() < 0.5

ff = french("F-F_Research_Data_Factors")
d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
d["px"] = (1 + d.r).cumprod()
ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
rE, _ = run(rule_e(ens, d.index), d, lag=0, borrow_spread=0.01)
M = lambda r: (1 + r).groupby(r.index.to_period("M")).prod() - 1
eM, rfM, mktM = M(rE), M(d.rf), M(d.r)
ensM = ens.groupby(ens.index.to_period("M")).last().shift(1)          # known at prior month end
X = pd.DataFrame({"E": eM, "rf": rfM, "mkt": mktM, "lv": lv, "s": ensM}).dropna()
X["L1"] = X.s * (X.lv - X.rf) + X.rf
X["L15"] = 1.5 * X.s * (X.lv - X.rf) + X.rf - (1.5 * X.s - 1).clip(lower=0) * 0.01 / 12
X["trend on mkt (same proxy)"] = X.s * (X.mkt - X.rf) + X.rf

def mstats(r, rf):
    ex = r - rf; w = (1 + r).cumprod()
    return {"cagr": w.iloc[-1] ** (12 / len(r)) - 1, "sh": ex.mean() / ex.std() * 12 ** .5, "dd": (w / w.cummax() - 1).min(), "vol": r.std() * 12 ** .5}

def boot(a, b, rf, n=2000, block=6, seed=0):
    rng = np.random.default_rng(seed); ea, eb = (a - rf).values, (b - rf).values; T = len(ea); out = []
    for _ in range(n):
        i = (rng.integers(0, T, T // block + 1)[:, None] + np.arange(block)).ravel()[:T] % T
        out.append(ea[i].mean() / ea[i].std() * 12 ** .5 - eb[i].mean() / eb[i].std() * 12 ** .5)
    return np.percentile(out, [2.5, 97.5])

res = {}
for e, (a, b) in {"1964-2007": ("1964-01", "2007-12"), "2008-26": ("2008-01", None)}.items():
    x = X.loc[a:b]; print(f"=== {e} ({x.index[0]}..{x.index[-1]}) ===")
    st = {k: mstats(x[k], x.rf) for k in ["mkt", "lv", "E", "trend on mkt (same proxy)", "L1", "L15"]}
    for k, s in st.items(): print(f"  {k:26s} cagr {s['cagr']:6.1%} sh {s['sh']:.2f} dd {s['dd']:6.1%} vol {s['vol']:5.1%}")
    for k in ["L1", "L15"]:
        lo, hi = boot(x[k], x.E, x.rf); dS = st[k]["sh"] - st["E"]["sh"]
        print(f"  {k} vs E: dSharpe {dS:+.2f} [{lo:+.2f},{hi:+.2f}], dd {st[k]['dd']:.1%} vs {st['E']['dd']:.1%}")
        res.setdefault(k, []).append(dS >= 0.03 and st[k]["dd"] >= st["E"]["dd"])
for k, v in res.items(): print(k, "ADOPT" if all(v) else "REJECT")
# result note: both REJECT vs daily E. but vs the same monthly proxy on the market, L1 is better in both eras
# (sh 0.54 vs 0.45, dd -20.6 vs -24.2; 0.95 vs 0.88, dd -14.5 vs -16.0). daily test follows in lowvol_ind.py.
