"""shared data loader, execution model, metrics. every strategy returns target weights known at close t."""
import pathlib
import numpy as np, pandas as pd

D = pathlib.Path(__file__).parent / "data"
TD = 252


def _yf(name, col="Close"):
    return pd.read_csv(D / f"{name}.csv", index_col=0, parse_dates=True)[col]


def _fred(sid):
    s = pd.read_csv(D / f"fred_{sid}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def load(start="2006-07-17"):
    """daily frame on spy trading days. spy is total return (adjusted close)."""
    spy = _yf("spy")
    d = pd.DataFrame({"px": spy, "hi": _yf("spy", "High"), "lo": _yf("spy", "Low"), "op": _yf("spy", "Open")})
    d["r"] = d.px.pct_change()
    d["vix"] = _yf("vix").reindex(d.index).ffill()
    d["vix3m"] = _yf("vix3m").reindex(d.index).ffill()
    d["rf"] = (_fred("DTB3").reindex(d.index).ffill() / 100 / TD).fillna(0)
    return d.loc[start:]


def run(w, d, lag=1, cost_bp=2.0, borrow_spread=0.005):
    """w: target weight in spy known at close t. earns spy return from t+lag. cash earns rf.
    leverage above 1 pays rf + borrow_spread. cost = cost_bp per unit turnover."""
    pos = w.shift(lag).reindex(d.index).fillna(0.0)
    r = pos * d.r + (1 - pos).clip(lower=0) * d.rf + (1 - pos).clip(upper=0) * (d.rf + borrow_spread / TD)
    r = r - pos.diff().abs().fillna(0) * cost_bp / 1e4
    return r.iloc[1:], pos


def stats(r, rf):
    ex = r - rf.reindex(r.index)
    eq = (1 + r).cumprod()
    yrs = len(r) / TD
    dd = eq / eq.cummax() - 1
    down = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TD)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, vol=r.std() * np.sqrt(TD), sharpe=ex.mean() * TD / (r.std() * np.sqrt(TD)),
                sortino=ex.mean() * TD / down, maxdd=dd.min(), calmar=cagr / -dd.min())


def alpha(r, bench, rf):
    """annualised jensen alpha and t-stat vs benchmark, daily excess returns."""
    y, x = (r - rf.reindex(r.index)).align(bench - rf.reindex(bench.index), join="inner")
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    res = y.values - X @ b
    se = np.sqrt(np.diag(res.var(ddof=2) * np.linalg.inv(X.T @ X)))
    return b[0] * TD, b[0] / se[0], b[1]


def sharpe_diff_ci(r, bench, rf, n=2000, block=21, seed=0):
    """circular block bootstrap of sharpe(r) - sharpe(bench). returns (point, lo, hi, p_le_0)."""
    rng = np.random.default_rng(seed)
    a, b = (r - rf.reindex(r.index)).align(bench - rf.reindex(bench.index), join="inner")
    a, b = a.values, b.values
    T = len(a)
    nb = -(-T // block)
    sh = lambda x: x.mean() / x.std() * np.sqrt(TD)
    out = np.empty(n)
    for i in range(n):
        idx = (rng.integers(0, T, nb)[:, None] + np.arange(block)).ravel()[:T] % T
        out[i] = sh(a[idx]) - sh(b[idx])
    return sh(a) - sh(b), *np.percentile(out, [2.5, 97.5]), (out <= 0).mean()


def report(name, r, d, bench_r=None):
    s = stats(r, d.rf)
    bench_r = d.r.iloc[1:] if bench_r is None else bench_r
    a, t, beta = alpha(r, bench_r, d.rf)
    pt, lo, hi, p = sharpe_diff_ci(r, bench_r, d.rf)
    return dict(name=name, **s, alpha=a, alpha_t=t, beta=beta, dSharpe=pt, ci_lo=lo, ci_hi=hi, p=p)


def table(rows):
    df = pd.DataFrame(rows).set_index("name")
    pd.set_option("display.width", 220)
    fmt = {c: "{:.1%}".format for c in ["cagr", "vol", "maxdd", "alpha"]}
    fmt.update({c: "{:.2f}".format for c in ["sharpe", "sortino", "calmar", "alpha_t", "beta", "dSharpe", "ci_lo", "ci_hi", "p"]})
    return df.to_string(formatters=fmt)


def band(px, n, w):
    """hysteresis trend state: on when px > sma*(1+w), off when px < sma*(1-w), else unchanged."""
    m = px.rolling(n).mean().values; on, out = 0.0, []
    for p, mm in zip(px.values, m):
        if np.isnan(mm): out.append(0.0); continue
        if p > mm * (1 + w): on = 1.0
        elif p < mm * (1 - w): on = 0.0
        out.append(on)
    return pd.Series(out, index=px.index)
