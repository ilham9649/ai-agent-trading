"""new information: (1) ken french daily factors 1963+ (survivorship-free), (2) tradable factor etfs vs spy."""
import io, zipfile, requests
import numpy as np, pandas as pd, yfinance as yf
from engine import *

U = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_daily_CSV.zip"


def french(name):
    p = D / f"french_{name}.csv"
    if not p.exists():
        z = zipfile.ZipFile(io.BytesIO(requests.get(U.format(name), timeout=60, headers={"User-Agent": "Mozilla/5.0"}).content))
        t = z.read(z.namelist()[0]).decode("latin1")
        lines = t.splitlines()
        i = next(k for k, l in enumerate(lines) if l.strip()[:8].isdigit())
        hdr = lines[i - 1].split(",")
        rows = []
        for l in lines[i:]:
            if not l.strip()[:8].isdigit():
                break
            rows.append(l.split(","))
        df = pd.DataFrame(rows, columns=["date"] + [h.strip() for h in hdr[1:]]).set_index("date")
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.astype(float).div(100).to_csv(p)
    return pd.read_csv(p, index_col=0, parse_dates=True)


def sh(x, rf=0.0):
    x = x - rf
    return x.mean() / x.std() * np.sqrt(TD)


if __name__ == "__main__":
    ff = french("F-F_Research_Data_5_Factors_2x3").join(french("F-F_Momentum_Factor"), how="inner")
    ff.columns = [c.strip() for c in ff.columns]
    print(ff.columns.tolist(), ff.index.min().date(), ff.index.max().date())
    mkt = ff["Mkt-RF"]
    fac = ["SMB", "HML", "RMW", "CMA", "Mom"]
    for a, b in [("1963", "1989"), ("1990", "2007"), ("2008", "2026")]:
        s = ff.loc[a:b]
        print(a, b, "sharpe:", {c: round(sh(s[c]), 2) for c in ["Mkt-RF"] + fac})
    # market + equal-vol factor overlay (long-short, costs ignored -> upper bound)
    s = ff.loc["1990":]
    vol = s[fac].std()
    ov = (s[fac] / vol * 0.05 / np.sqrt(TD)).mean(axis=1)  # each factor scaled to 5% vol, averaged
    for L in [0.0, 1.0]:
        print("mkt-rf + overlay L=%.0f (no costs): sharpe %.2f" % (L, sh(mkt.loc["1990":] + L * ov)))
    # tradable factor etfs vs spy, real prices
    tk = ["SPY", "MTUM", "QUAL", "VLUE", "USMV", "RSP", "IWM", "SPHQ"]
    px = yf.download(tk, start="2013-07-01", auto_adjust=True, progress=False)["Close"].dropna(how="any")
    rf = pd.read_csv(D / "fred_DTB3.csv", index_col=0, parse_dates=True).iloc[:, 0].pipe(pd.to_numeric, errors="coerce").reindex(px.index).ffill() / 100 / TD
    r = px.pct_change().dropna()
    print("etfs", px.index.min().date(), "->", px.index.max().date())
    for c in r:
        eq = (1 + r[c]).cumprod()
        print(f"{c:5s} cagr {eq.iloc[-1] ** (TD / len(r)) - 1:6.1%} sharpe {sh(r[c], rf.loc[r.index]):.2f} maxdd {(eq / eq.cummax() - 1).min():6.1%}")
