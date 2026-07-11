#!/usr/bin/env python3
"""Profit search on LONG daily history (more trades = real significance).

Pulls ~7y of daily BTC/ETH, develops on the first half, tests OOS on the second
(~3.4y). Runs trend families + a 2-of-3 ensemble. Reports per-asset and the
equal-weight COMBINED OOS result with total trade count (significance).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series, is_oos_split  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.ema_cross import EmaCrossStrategy  # noqa: E402
from trading_agent.strategy.momentum import MomentumStrategy  # noqa: E402
from trading_agent.strategy.swing import SwingStrategy  # noqa: E402
from trading_agent.strategy.ensemble import EnsembleStrategy  # noqa: E402

TOTAL = 2500   # ~7y daily
OOS_FRAC = 0.5


def strat_factories(st):
    return {
        "breakout": lambda: SwingStrategy(st),
        "ema_cross": lambda: EmaCrossStrategy(st),
        "momentum": lambda: MomentumStrategy(st),
        "ensemble_2of3": lambda: EnsembleStrategy(st, [SwingStrategy(st), EmaCrossStrategy(st), MomentumStrategy(st)], 2),
    }


def main() -> int:
    st = load_settings().model_copy(update={
        "strategy": load_settings().strategy.model_copy(update={"timeframe": "1d"}),
        "execution": load_settings().execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
    })
    syms = st.symbols
    series = {sym: binance_series(st, sym, interval="1d", total=TOTAL) for sym in syms}

    print(f"Daily history: { {sym: len(series[sym]) for sym in syms} }  (OOS {int((1-OOS_FRAC)*100)}%)\n")
    print(f"{'strategy':16} {'asset':11} {'OOS ret':>9} {'sharpe':>7} {'trades':>7} {'maxDD':>6} {'DSR':>5}")
    print("-" * 60)

    rows = []
    for name, fac in strat_factories(st).items():
        per_asset = {}
        for sym in syms:
            is_s, oos = is_oos_split({sym: series[sym]}, OOS_FRAC)
            r = Backtest(st, fac()).run(oos, n_trials=4, sample="OOS")
            per_asset[sym] = r
            print(f"{name:16} {sym:11} {r.total_return_pct:+8.2f}% {r.sharpe:7.2f} "
                  f"{r.num_trades:7d} {r.max_drawdown_pct:5.1f}% {r.deflated_sharpe:5.2f}")
        combined_ret = sum(per_asset[s].total_return_pct for s in syms) / len(syms)
        combined_trades = sum(per_asset[s].num_trades for s in syms)
        avg_sharpe = sum(per_asset[s].sharpe for s in syms) / len(syms)
        rows.append((name, combined_ret, avg_sharpe, combined_trades, per_asset))

    print("\n=== Combined (equal-weight BTC+ETH) OOS over ~3.4y ===")
    print(f"{'strategy':16} {'OOS ret':>9} {'sharpe':>7} {'total trades':>13}")
    for name, cret, csh, ctr, _ in sorted(rows, key=lambda r: r[1], reverse=True):
        print(f"{name:16} {cret:+8.2f}% {csh:7.2f} {ctr:13d}")

    best = max(rows, key=lambda r: r[1])
    print(f"\nBest combined OOS: {best[0]} {best[1]:+.2f}% over {best[3]} trades.")
    if best[1] > 0 and best[3] >= 10:
        print(f"VERDICT: positive OOS edge with >=10 trades — credible (modest) profitability.")
    elif best[1] > 0:
        print(f"VERDICT: positive but thin ({best[3]} trades) — needs more data / ML to confirm.")
    else:
        print("VERDICT: no OOS edge -> pivot to ML forecasting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
