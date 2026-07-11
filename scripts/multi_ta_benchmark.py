#!/usr/bin/env python3
"""Multi-indicator ensemble vs buy-and-hold (full cycle, standard-timing engine).

Full-exposure long/flat. Reports ABSOLUTE return vs B&H + drawdown, per asset, across
vote thresholds. Paper-only. Honest bar: beats B&H in dollars on BOTH assets robustly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.multi_ta import MultiTA  # noqa: E402
from trading_agent.strategy.trend_long_flat import TrendLongFlat  # noqa: E402

TOTAL = 3000
THRESHOLDS = [4, 5, 6]


def aggressive(base):
    return base.model_copy(update={
        "strategy": base.strategy.model_copy(update={"name": "multi_ta", "timeframe": "1d",
                                                      "stop_atr_mult": 10.0, "take_profit_atr_mult": 100.0}),
        "execution": base.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
        "risk": base.risk.model_copy(update={"sizing_mode": "fixed_fraction", "fixed_fraction_pct": 100.0,
                                              "max_position_notional_pct": 100.0, "max_deployed_pct": 100.0,
                                              "max_concurrent_positions": 1, "max_trades_per_day": 20,
                                              "daily_loss_cap_pct": 100.0, "drawdown_alert_pct": 100.0,
                                              "drawdown_kill_pct": 100.0}),
    })


def main() -> int:
    base = load_settings()
    st = aggressive(base)
    syms = base.symbols
    series = {sym: binance_series(st, sym, interval="1d", total=TOTAL) for sym in syms}
    bh = {sym: (series[sym]["close"].to_numpy(dtype=float)[-1] / series[sym]["close"].to_numpy(dtype=float)[0] - 1) * 100
          for sym in syms}
    print(f"Full-cycle daily (~{TOTAL} bars), full exposure. B&H: { {s: f'{bh[s]:+.0f}%' for s in syms} }\n")

    print(f"{'strategy':18} {'asset':11} {'return':>9} {'vs B&H':>9} {'maxDD':>7} {'trades':>6} {'beats?':>7}")
    print("-" * 64)
    # single-indicator reference
    for sym in syms:
        r = Backtest(st, TrendLongFlat(st, sma_len=200)).run({sym: series[sym]}, n_trials=1, sample="full")
        print(f"{'trend SMA200':18} {sym:11} {r.total_return_pct:+8.0f}% {r.total_return_pct-bh[sym]:+8.0f}% "
              f"{r.max_drawdown_pct:6.1f}% {r.num_trades:6d} {'YES' if r.total_return_pct > bh[sym] else 'no':>7}")
    for thr in THRESHOLDS:
        for sym in syms:
            r = Backtest(st, MultiTA(st, threshold=thr)).run({sym: series[sym]}, n_trials=1, sample="full")
            print(f"{'multi-TA t='+str(thr):18} {sym:11} {r.total_return_pct:+8.0f}% {r.total_return_pct-bh[sym]:+8.0f}% "
                  f"{r.max_drawdown_pct:6.1f}% {r.num_trades:6d} {'YES' if r.total_return_pct > bh[sym] else 'no':>7}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
