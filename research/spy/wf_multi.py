"""train 2005-2015 / test 2016-2026 on the multi-asset trend book grid + deflated sharpe."""
import itertools
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from engine import *
from strat_m import assets, multi_weights, run_multi


def dsr(r, rf, sr_trials):
    """deflated sharpe (bailey/lopez de prado): prob true sharpe > best-of-n-trials expectation. daily units."""
    x = (r - rf.reindex(r.index)).dropna()
    T, sr = len(x), x.mean() / x.std()
    n = len(sr_trials)
    g = 0.5772156649
    sr0 = np.std(sr_trials) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    s3, k4 = skew(x), kurtosis(x, fisher=False)
    return norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - s3 * sr + (k4 - 1) / 4 * sr ** 2))


if __name__ == "__main__":
    d, px = assets()
    sets = {"all4": ("SPY", "IEF", "TLT", "GLD"), "spy+ief+gld": ("SPY", "IEF", "GLD"), "spy+ief": ("SPY", "IEF"), "spy+gld": ("SPY", "GLD")}
    grid = list(itertools.product(sets, [0.10, 0.12, 0.15], [1.5, 2.0], [150, 200, 250]))
    R = {}
    for g in grid:
        R[g] = run_multi(multi_weights(px, d, sma=g[3], sig=g[1], cap=g[2], names=sets[g[0]]), px, d)
    tr, te = slice("2006-06-01", "2015-12-31"), slice("2016-01-01", None)
    spy = d.r.iloc[1:]
    sh = lambda r, s: stats(r.loc[s], d.rf)["sharpe"]
    df = pd.DataFrame({"train": {g: sh(r, tr) for g, r in R.items()}, "test": {g: sh(r, te) for g, r in R.items()},
                       "cagr_test": {g: stats(r.loc[te], d.rf)["cagr"] for g, r in R.items()},
                       "full": {g: sh(r, slice("2006-06-01", None)) for g, r in R.items()}})
    print(f"spy sharpe: train {sh(spy, tr):.2f} test {sh(spy, te):.2f} full {sh(spy, slice('2006-06-01', None)):.2f}; spy cagr test {stats(spy.loc[te], d.rf)['cagr']:.1%}")
    best = df.train.idxmax()
    print("best on train:", best, df.loc[best].round(3).to_dict())
    print("corr(train, test sharpe) across grid: %.2f" % df.train.corr(df.test))
    print("share of grid beating spy on test sharpe: %.0f%%; on both: %.0f%%" % ((df.test > sh(spy, te)).mean() * 100, ((df.test > sh(spy, te)) & (df.train > sh(spy, tr))).mean() * 100))
    print(df.groupby(level=0).mean().round(3))
    full = slice("2006-06-01", None)
    trials = np.array([stats(r.loc[full], d.rf)["sharpe"] / np.sqrt(TD) for r in R.values()])
    for g in [best, df.full.idxmax()]:
        print("dsr", g, round(dsr(R[g].loc[full], d.rf, trials), 3))
    print("dsr spy (n=1 baseline, sr0=0 not applicable) skip")
