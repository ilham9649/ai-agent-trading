"""hypothesis b (event windows) and c (credit-led regime), plus a trend baseline."""
import numpy as np, pandas as pd
from engine import *
from engine import _fred, _yf


def tom_days(idx, n_last=1, n_first=3):
    """1.0 on the last n_last and first n_first trading days of each month (return days)."""
    s = pd.Series(idx, index=idx)
    ym = idx.to_period("M")
    k_first = s.groupby(ym).cumcount()
    k_last = s.groupby(ym).cumcount(ascending=False)
    return ((k_first < n_first) | (k_last < n_last)).astype(float)


def fomc_days(idx):
    f = pd.to_datetime(pd.read_csv(D / "fomc.csv").date)
    return pd.Series(idx.isin(f), index=idx).astype(float)


def event_weights(d, tom=True, fomc=True, L=1.0, n_last=1, n_first=3):
    e = pd.Series(0.0, index=d.index)
    if tom:
        e = np.maximum(e, tom_days(d.index, n_last, n_first))
    if fomc:
        e = np.maximum(e, fomc_days(d.index))
    return e * L


def credit_weights(d, win=63, zwin=252, zthr=-1.0, div=True, series="BAA10Y"):
    if series == "hyg":
        s = -np.log(_yf("hyg") / _yf("ief")).reindex(d.index).ffill()  # rising = spread widening
    else:
        s = _fred(series).reindex(d.index).ffill()
    m = -(s - s.shift(win))
    z = (m - m.rolling(zwin).mean()) / m.rolling(zwin).std()
    w = (z > zthr).astype(float)
    if div:
        q = d.px.pct_change(win)
        qz = (q - q.rolling(zwin).mean()) / q.rolling(zwin).std()
        w = w.where(~((z - qz) < -1.5), w.clip(upper=0.5))
    return w.where(z.notna(), 0.0)


def trend_weights(d, n=200):
    return (d.px > d.px.rolling(n).mean()).astype(float)


if __name__ == "__main__":
    d = load("1993-02-01")
    rows = [report("spy b&h", d.r.loc["1994-06-01":], d.loc["1994-06-01":])]
    for name, w, lag in [
        ("b tom L1", event_weights(d, fomc=False), 0),
        ("b fomc L1", event_weights(d, tom=False), 0),
        ("b tom+fomc L1", event_weights(d), 0),
        ("b tom+fomc L2", event_weights(d, L=2), 0),
        ("c baa10y", credit_weights(d), 1),
        ("c baa10y no div", credit_weights(d, div=False), 1),
        ("c hyg/ief", credit_weights(d, series="hyg"), 1),
        ("trend 200d", trend_weights(d), 1),
    ]:
        r, pos = run(w, d, lag=lag)
        r = r.loc["1994-06-01":] if "hyg" not in name else r.loc["2008-06-01":]
        dd = d.loc[r.index[0]:]
        rows.append(report(name + f" [{pos.loc[r.index].mean():.0%} inv]", r, dd, dd.r))
    print(table(rows))


def preholiday_days(idx):
    """1.0 on the last trading day before a weekday market closure (also flags a few unscheduled closures)."""
    d = idx.values.astype("datetime64[D]")
    return pd.Series(np.r_[np.busday_count(d[:-1], d[1:]) > 1, False], index=idx).astype(float)


def rule_e(base, idx):
    """chosen rule E: trend base (decided at prior close) x tom boost x pre-holiday boost, cap 1.5x. calendar known ahead, so lag 0."""
    return (base.shift(1).fillna(0) * (1 + tom_days(idx, 1, 3)) * (1 + preholiday_days(idx))).clip(upper=1.5)
