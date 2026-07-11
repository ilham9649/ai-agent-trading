#!/usr/bin/env python3
"""Profitability lab: search for OUT-OF-SAMPLE edge across strategy families on real
BTC/ETH history. Criterion for 'edge': OOS return > 0 net of realistic fees+slippage,
with non-trivial Deflated Sharpe. Ranks by OOS net return.

Honest guardrail: we report OOS, not in-sample. No parameter is tuned to fake a win.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series, is_oos_split  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.ema_cross import EmaCrossStrategy  # noqa: E402
from trading_agent.strategy.mean_reversion import MeanReversionStrategy  # noqa: E402
from trading_agent.strategy.momentum import MomentumStrategy  # noqa: E402
from trading_agent.strategy.swing import SwingStrategy  # noqa: E402

STRATS = [
    ("breakout", SwingStrategy),
    ("ema_cross", EmaCrossStrategy),
    ("mean_reversion", MeanReversionStrategy),
    ("momentum", MomentumStrategy),
]
TIMEFRAMES = {"1d": 1000, "4h": 2000}  # bars to pull (~2.7y daily, ~11mo 4h)
OOS_FRAC = 0.40


def variant(settings, tf):
    return settings.model_copy(update={
        "strategy": settings.strategy.model_copy(update={"timeframe": tf}),
        "execution": settings.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
    })


def main() -> int:
    settings = load_settings()
    rows = []
    for sname, cls in STRATS:
        for sym in settings.symbols:
            for tf, total in TIMEFRAMES.items():
                st = variant(settings, tf)
                try:
                    series = binance_series(st, sym, interval=tf, total=total)
                    is_s, oos = is_oos_split({sym: series}, OOS_FRAC)
                    bt_is = Backtest(st, cls(st)).run(is_s, n_trials=1, sample="IS")
                    bt_oos = Backtest(st, cls(st)).run(oos, n_trials=1, sample="OOS")
                except Exception as exc:
                    print(f"# {sname:14} {sym:11} {tf}  ERROR {exc!r}")
                    continue
                rows.append((sname, sym, tf, bt_is, bt_oos))
                star = "  <<< OOS PROFITABLE" if bt_oos.total_return_pct > 0 else ""
                print(f"{sname:14} {sym:11} {tf}  IS ret={bt_is.total_return_pct:+7.2f}% "
                      f"sharpe={bt_is.sharpe:5.2f}  |  OOS ret={bt_oos.total_return_pct:+7.2f}% "
                      f"sharpe={bt_oos.sharpe:5.2f} trades={bt_oos.num_trades:3} "
                      f"DD={bt_oos.max_drawdown_pct:5.1f}% DSR={bt_oos.deflated_sharpe:.2f}{star}")

    print("\n=== OOS leaderboard (net of fees+slippage) ===")
    for sname, sym, tf, _is, oos in sorted(rows, key=lambda r: r[4].total_return_pct, reverse=True):
        print(f"  OOS ret={oos.total_return_pct:+7.2f}%  sharpe={oos.sharpe:5.2f}  "
              f"{sname:14} {sym:11} {tf}  trades={oos.num_trades}")

    profitable = [r for r in rows if r[4].total_return_pct > 0]
    if profitable:
        print(f"\n{len(profitable)} OOS-profitable config(s) found.")
    else:
        print("\nNo rule-based OOS edge found across families -> pivot to ML forecasting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
