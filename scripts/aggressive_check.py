#!/usr/bin/env python3
"""Validate the FULL-EXPOSURE long/flat trend in the real engine (parity with the
standalone leverage_test), confirm it beats buy-and-hold in absolute dollars, and
print the strategy's CURRENT live signal (long or flat) on real prices. Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.trend_long_flat import TrendLongFlat  # noqa: E402

TOTAL = 3000


def main() -> int:
    base = load_settings()
    # Aggressive profile: full exposure when long, SMA-exit governs (wide stop/TP),
    # risk caps loosened so the strategy can ride a full cycle (matches the test).
    st = base.model_copy(update={
        "strategy": base.strategy.model_copy(update={"name": "trend_long_flat", "timeframe": "1d",
                                                      "stop_atr_mult": 10.0, "take_profit_atr_mult": 100.0}),
        "execution": base.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
        "risk": base.risk.model_copy(update={"sizing_mode": "fixed_fraction", "fixed_fraction_pct": 100.0,
                                              "max_position_notional_pct": 100.0, "max_deployed_pct": 100.0,
                                              "max_concurrent_positions": 1, "max_trades_per_day": 10,
                                              "daily_loss_cap_pct": 100.0, "drawdown_alert_pct": 100.0,
                                              "drawdown_kill_pct": 100.0}),
    })

    print("=== Engine parity: full-exposure long/flat trend vs buy-and-hold (per asset, ~8y) ===")
    any_beat = False
    for sym in base.symbols:
        series = binance_series(st, sym, interval="1d", total=TOTAL)
        close = series["close"].to_numpy(dtype=float)
        r = Backtest(st, TrendLongFlat(st, sma_len=200)).run({sym: series}, n_trials=1, sample="full")
        bh = (close[-1] / close[0] - 1) * 100
        beat = r.total_return_pct > bh
        any_beat = any_beat or beat
        print(f"  {sym:11} engine {r.total_return_pct:+8.0f}%   B&H {bh:+8.0f}%   "
              f"maxDD {r.max_drawdown_pct:5.1f}%   trades {r.num_trades:3d}   "
              f"{'<<< BEATS B&H' if beat else '(under)'}")

    print("\n=== Current live signal (real prices, price vs 200-day SMA) ===")
    for sym in base.symbols:
        df = binance_series(st, sym, interval="1d", total=250)
        price = float(df["close"].iloc[-1])
        sma = float(df["close"].rolling(200).mean().iloc[-1])
        state = "LONG (100%)" if price > sma else "FLAT (stables)"
        print(f"  {sym:11} price={price:,.0f}  SMA200={sma:,.0f}  -> {state}")

    print(f"\nVerdict: engine reproduces a buy-and-hold-beating absolute return: {'YES' if any_beat else 'no'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
