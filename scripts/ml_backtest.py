#!/usr/bin/env python3
"""Walk-forward ML forecasting backtest.

Trains a gradient-boosting directional model on past daily bars, trades its forecast
on held-out OOS years. Reports net-of-cost OOS metrics vs a buy-and-hold benchmark
(the honest question: does timing beat just holding the asset?).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series, is_oos_split  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.ml.forecast import ForecastStrategy  # noqa: E402

TOTAL = 2500
OOS_FRAC = 0.5
THRESHOLDS = [0.55, 0.60]


def main() -> int:
    base = load_settings()
    st = base.model_copy(update={
        "strategy": base.strategy.model_copy(update={"timeframe": "1d"}),
        "execution": base.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
    })
    syms = st.symbols
    series = {sym: binance_series(st, sym, interval="1d", total=TOTAL) for sym in syms}

    print(f"Daily history { {s: len(series[s]) for s in syms} }, OOS {int((1-OOS_FRAC)*100)}% (~3.4y)\n")
    for thr in THRESHOLDS:
        print(f"=== ForecastStrategy  P(up)>={thr} ===")
        print(f"{'asset':11} {'OOS ret':>9} {'B&H':>9} {'sharpe':>7} {'trades':>7} {'win':>5} {'maxDD':>6} {'DSR':>5}")
        cret = cbh = 0.0
        for sym in syms:
            is_s, oos = is_oos_split({sym: series[sym]}, OOS_FRAC)
            r = Backtest(st, ForecastStrategy(st, prob_threshold=thr)).run(oos, n_trials=4, sample="OOS")
            bh = (float(oos[sym]["close"].iloc[-1]) / float(oos[sym]["close"].iloc[0]) - 1) * 100
            cret += r.total_return_pct; cbh += bh
            print(f"{sym:11} {r.total_return_pct:+8.2f}% {bh:+8.2f}% {r.sharpe:7.2f} "
                  f"{r.num_trades:7d} {r.win_rate:4.0f}% {r.max_drawdown_pct:5.1f}% {r.deflated_sharpe:5.2f}")
        n = len(syms)
        print(f"{'COMBINED':11} {cret/n:+8.2f}% {cbh/n:+8.2f}%   (avg per asset)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
