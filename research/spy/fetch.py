"""download all research inputs to data/*.csv"""
import io, pathlib
import pandas as pd, requests, yfinance as yf

out = pathlib.Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

for tk, name in [("SPY", "spy"), ("^GSPC", "gspc"), ("^VIX", "vix"), ("^VIX3M", "vix3m"), ("^VIX9D", "vix9d"),
                 ("SHY", "shy"), ("IEF", "ief"), ("HYG", "hyg"), ("TLT", "tlt"), ("GLD", "gld")]:
    df = yf.download(tk, start="1990-01-01", auto_adjust=True, progress=False)
    df.columns = [c[0] for c in df.columns]
    df.to_csv(out / f"{name}.csv")
    print(name, len(df), df.index.min().date(), df.index.max().date())

for sid in ["VIXCLS", "VXVCLS", "BAMLH0A0HYM2", "DTB3", "DGS10", "DGS2", "T10Y3M"]:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=30)
    df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    df.to_csv(out / f"fred_{sid}.csv")
    s = df.iloc[:, 0].replace(".", pd.NA).dropna()
    print(sid, len(s), s.index.min().date(), s.index.max().date())
