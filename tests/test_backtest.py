import math

from trading_agent.backtest.engine import Backtest, deflated_sharpe
from trading_agent.backtest.feed import is_oos_split, synthetic_series


def test_backtest_runs_and_metrics_finite(settings):
    series = synthetic_series(settings, bars=600)
    res = Backtest(settings).run(series, n_trials=1, sample="full")
    assert res.bars > 100
    for v in (res.total_return_pct, res.cagr_pct, res.sharpe, res.sortino, res.expectancy_usd):
        assert math.isfinite(v)
    assert res.max_drawdown_pct >= 0.0
    assert 0.0 <= res.deflated_sharpe <= 1.0
    assert res.num_trades >= 0


def test_is_oos_split_lengths(settings):
    series = synthetic_series(settings, bars=400)
    in_s, oos = is_oos_split(series, oos_frac=0.25)
    for sym in series:
        assert len(in_s[sym]) + len(oos[sym]) == len(series[sym])
        assert len(oos[sym]) == 100  # 25% of 400


def test_deflated_sharpe_is_a_probability():
    # more trials -> lower DSR for the same Sharpe (selection bias penalized)
    dsr1, _ = deflated_sharpe(per_obs_sharpe=0.05, n_obs=500, n_trials=1, skew=0.0, kurt=3.0)
    dsr50, _ = deflated_sharpe(per_obs_sharpe=0.05, n_obs=500, n_trials=50, skew=0.0, kurt=3.0)
    assert 0.0 <= dsr1 <= 1.0 and 0.0 <= dsr50 <= 1.0
    assert dsr50 < dsr1
