#!/usr/bin/env python3
"""Long/flat trend backtest over a FULL cycle (incl. the 2022 bear) vs buy-and-hold.

The honest question: does staying long in uptrends and flat in downtrends capture a
solid fraction of buy-and-hold's return at a MUCH smaller max drawdown (so $500
survives)? Wide stops (3*ATR); the SMA cross is the real exit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.trend_long_flat import TrendLongFlat  # noqa: E402

TOTAL = 3000  # ~8y daily (full cycle incl. 2022 bear)
SMA_LENS = [100, 150, 200]


def bnh_maxdd(close: np.ndarray) -> tuple[float, float]:
    ret = (close[-1] / close[0] - 1) * 100
    running = np.maximum.accumulate(close)
    dd = ((close - running) / running).min() * 100
    return ret, abs(dd)


def main() -> int:
    base = load_settings()
    st = base.model_copy(update={
        "strategy": base.strategy.model_copy(update={"timeframe": "1d", "stop_atr_mult": 3.0,
                                                      "take_profit_atr_mult": 20.0}),
        "execution": base.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
        # Fuller sizing for a long/flat trend strategy: the SMA cross is the real risk
        # control, so we deploy a meaningful fraction (not the 1%-over-wide-stop minimum).
        "risk": base.risk.model_copy(update={"sizing_mode": "fixed_fraction",
                                             "fixed_fraction_pct": 45.0,
                                             "max_position_notional_pct": 60.0,
                                             "max_deployed_pct": 100.0}),
    })
    syms = st.symbols
    series = {sym: binance_series(st, sym, interval="1d", total=TOTAL) for sym in syms}
    print(f"Full-cycle daily history { {s: len(series[s]) for s in syms} }\n")

    for sma in SMA_LENS:
        print(f"=== TrendLongFlat  SMA={sma}  (3*ATR stop) ===")
        print(f"{'asset':11} {'strat':>9} {'B&H':>9} {'stratDD':>8} {'B&H_DD':>8} {'trades':>7} {'DSR':>5}")
        cs = cb = csd = cbd = 0.0
        for sym in syms:
            close = series[sym]["close"].to_numpy(dtype=float)
            r = Backtest(st, TrendLongFlat(st, sma_len=sma)).run({sym: series[sym]}, n_trials=3, sample="full")
            bh_ret, bh_dd = bnh_maxdd(close)
            cs += r.total_return_pct; cb += bh_ret; csd += r.max_drawdown_pct; cbd += bh_dd
            print(f"{sym:11} {r.total_return_pct:+8.2f}% {bh_ret:+8.1f}% {r.max_drawdown_pct:7.1f}% "
                  f"{bh_dd:7.1f}% {r.num_trades:7d} {r.deflated_sharpe:5.2f}")
        n = len(syms)
        print(f"{'AVG':11} {cs/n:+8.2f}% {cb/n:+8.1f}% {csd/n:7.1f}% {cbd/n:7.1f}%\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
