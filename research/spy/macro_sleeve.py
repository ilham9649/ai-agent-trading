"""diversifying sleeve from outside us stocks, added as an overlay on chosen rule E. run: uv run --with openpyxl python macro_sleeve.py
data: aqr data library (free, monthly, paper returns, gross of costs):
  A "All Macro Multi-style" (value+momentum+carry+defensive across equity indices, bonds, fx, commodities), 1926+
  B "TSMOM" (12-month time-series momentum, 58 futures), 1985+
frozen before running, no retuning after:
  overlay = E monthly return + 0.5 x (sleeve excess return - 2%/yr haircut for costs and fees); sleeves are self-financing
  pass = vs E, dSharpe >= +0.03 AND max drawdown no worse than E, in 2008-2026 AND every holdout
  (A: 1927-62 and 1963-2007; B: 1985-2007). monthly stats.
reality check after: real aqr funds (after fees, daily) AQMIX managed futures 2010+, QSPIX style premia 2013+."""
import numpy as np, pandas as pd, yfinance as yf
from engine import *
from strat_f import french
from strat_bc import rule_e

def mstats(r, rf):
    rf = rf.reindex(r.index); ex = r - rf; w = (1 + r).cumprod()
    return {"cagr": w.iloc[-1] ** (12 / len(r)) - 1, "sh": ex.mean() / ex.std() * 12 ** .5, "dd": (w / w.cummax() - 1).min(), "vol": r.std() * 12 ** .5}

def boot(a, b, rf, n=2000, block=6, seed=0):
    rng = np.random.default_rng(seed); ea, eb = (a - rf.reindex(a.index)).values, (b - rf.reindex(b.index)).values; L = len(ea); out = []
    for _ in range(n):
        i = (rng.integers(0, L, L // block + 1)[:, None] + np.arange(block)).ravel()[:L] % L
        out.append(ea[i].mean() / ea[i].std() * 12 ** .5 - eb[i].mean() / eb[i].std() * 12 ** .5)
    return np.percentile(out, [2.5, 97.5])


if __name__ == "__main__":
    ff = french("F-F_Research_Data_Factors")
    d = merge_sat(pd.DataFrame({"r": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}))
    d["px"] = (1 + d.r).cumprod()
    ens = (band(d.px, 150, .03) + band(d.px, 200, .03) + band(d.px, 250, .03)) / 3
    rE, _ = run(rule_e(ens, d.index), d, lag=0, borrow_spread=0.01)
    M = lambda r: (1 + r).groupby(r.index.to_period("M")).prod() - 1
    eM, rfM, mktM = M(rE), M(d.rf), M(d.r)

    c = pd.read_excel(D / "Century-of-Factor-Premia-Monthly.xlsx", "Century of Factor Premia", header=18)
    c = c[pd.to_datetime(c.Date, errors="coerce").notna()]
    c.index = pd.to_datetime(c.Date).dt.to_period("M")
    t = pd.read_excel(D / "Time-Series-Momentum-Factors-Monthly.xlsx", "TSMOM Factors", header=17, index_col=0).dropna(how="all")
    t.index = pd.to_datetime(t.index).to_period("M")
    S = {"A macro multi-style": c["All Macro Multi-style"].astype(float), "B tsmom": t["TSMOM"].astype(float)}
    assert S["A macro multi-style"].index[0] == pd.Period("1926-07", "M") and S["B tsmom"].index[0] == pd.Period("1985-01", "M")

    H = {"A macro multi-style": {"1927-62": ("1927-07", "1962-12"), "1963-2007": ("1963-01", "2007-12")}, "B tsmom": {"1985-2007": ("1985-01", "2007-12")}}
    for name, s in S.items():
        ov = (eM + 0.5 * (s - 0.02 / 12)).dropna()
        print(f"\n##### {name}: sleeve alone sharpe {mstats(s.dropna() + rfM.reindex(s.dropna().index), rfM)['sh']:.2f}, "
              f"corr with E {pd.concat([s, eM], axis=1).dropna().corr().iloc[0, 1]:+.2f}")
        ok = True
        for e, (a, b) in (H[name] | {"2008-26": ("2008-01", None)}).items():
            x = {"spy": mktM.loc[a:b], "E": eM.loc[a:b], "E + 0.5 sleeve": ov.loc[a:b]}
            x = {k: v.loc[ov.loc[a:b].index[0]:ov.loc[a:b].index[-1]] for k, v in x.items()}
            print(f"=== {e} ({x['E'].index[0]}..{x['E'].index[-1]}) ===")
            st = {k: mstats(v, rfM) for k, v in x.items()}
            for k, v in st.items(): print(f"  {k:16s} cagr {v['cagr']:6.1%} sh {v['sh']:.2f} dd {v['dd']:6.1%} vol {v['vol']:5.1%}")
            dS = st["E + 0.5 sleeve"]["sh"] - st["E"]["sh"]; lo, hi = boot(x["E + 0.5 sleeve"], x["E"], rfM)
            print(f"  dSharpe {dS:+.2f} [{lo:+.2f},{hi:+.2f}]")
            ok &= dS >= 0.03 and st["E + 0.5 sleeve"]["dd"] >= st["E"]["dd"]
        print("ADOPT" if ok else "REJECT")

    print("\n##### reality check: real aqr mutual funds (after fees), daily, overlay 0.5 x (fund - t-bill) on E, no extra haircut")
    for tk in ["AQMIX", "QSPIX"]:
        f = yf.download(tk, period="max", progress=False, auto_adjust=True)["Close"].squeeze().pct_change().dropna()
        f = f.reindex(d.index).dropna(); r0 = rE.loc[f.index]
        x = {"spy": d.r.loc[f.index], "E": r0, f"E + 0.5 {tk}": r0 + 0.5 * (f - d.rf.loc[f.index]), f"{tk} alone": f}
        print(f"=== {tk} {f.index[0].date()}..{f.index[-1].date()} ===")
        for k, v in x.items():
            s = stats(v, d.rf); print(f"  {k:16s} cagr {s['cagr']:6.1%} sh {s['sharpe']:.2f} dd {s['maxdd']:6.1%}")
        pt, lo, hi, p = sharpe_diff_ci(x[f"E + 0.5 {tk}"], r0, d.rf, n=1000)
        print(f"  dSharpe vs E {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
