#!/usr/bin/env python3
"""Out-of-sample check for the multi-TA ensemble vs buy-and-hold.

If the full-cycle edge survives on UNSEEN data, it's more trustworthy. 50/50 split.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series, is_oos_split  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.multi_ta import MultiTA  # noqa: E402

TOTAL = 3000
OOS_FRAC = 0.5


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
    print(f"Multi-TA out-of-sample (last {int(OOS_FRAC*100)}% of ~{TOTAL} daily bars)\n")
    print(f"{'threshold':>9} {'asset':11} {'OOS strat':>10} {'OOS B&H':>9} {'maxDD':>7} {'trades':>6} {'beats?':>7}")
    for thr in [4, 5]:
        for sym in base.symbols:
            series = binance_series(st, sym, interval="1d", total=TOTAL)
            _, oos = is_oos_split({sym: series}, OOS_FRAC)
            r = Backtest(st, MultiTA(st, threshold=thr)).run(oos, n_trials=1, sample="OOS")
            c = oos[sym]["close"].to_numpy(dtype=float)
            bh = (c[-1] / c[0] - 1) * 100
            print(f"{thr:>9} {sym:11} {r.total_return_pct:+9.0f}% {bh:+8.0f}% "
                  f"{r.max_drawdown_pct:6.1f}% {r.num_trades:6d} {'YES' if r.total_return_pct > bh else 'no':>7}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
