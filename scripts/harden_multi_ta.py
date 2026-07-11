#!/usr/bin/env python3
"""HARDEN multi-TA: more assets (ETH/BTC/SOL/BNB) + multiple sequential regime windows.

Robustness bar: multi-TA must beat buy-and-hold in a MAJORITY of windows across a
MAJORITY of assets, not just one lucky slice. Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402
from trading_agent.strategy.multi_ta import MultiTA  # noqa: E402

TOTAL = 3000
CHUNKS = 5
THR = 4


def aggressive(base, universe):
    return base.model_copy(update={
        "strategy": base.strategy.model_copy(update={"name": "multi_ta", "timeframe": "1d",
                                                      "stop_atr_mult": 10.0, "take_profit_atr_mult": 100.0}),
        "execution": base.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
        "risk": base.risk.model_copy(update={"sizing_mode": "fixed_fraction", "fixed_fraction_pct": 100.0,
                                              "max_position_notional_pct": 100.0, "max_deployed_pct": 100.0,
                                              "max_concurrent_positions": 1, "max_trades_per_day": 20,
                                              "daily_loss_cap_pct": 100.0, "drawdown_alert_pct": 100.0,
                                              "drawdown_kill_pct": 100.0}),
        "universe": universe,
    })


def main() -> int:
    base = load_settings()
    extra = [
        UniverseItem(symbol="SOL-USDC", base="SOL", quote="USDC", coingecko_id="solana", decimals=6),
        UniverseItem(symbol="BNB-USDC", base="BNB", quote="USDC", coingecko_id="binancecoin", decimals=18),
    ]
    st = aggressive(base, list(base.universe) + extra)

    asset_full_beats = 0
    total_windows = 0
    window_beats = 0
    for sym in st.symbols:
        series = binance_series(st, sym, interval="1d", total=TOTAL)
        close = series["close"].to_numpy(dtype=float)
        r = Backtest(st, MultiTA(st, threshold=THR)).run({sym: series}, n_trials=1, sample="full")
        bh = (close[-1] / close[0] - 1) * 100
        full_beat = r.total_return_pct > bh
        asset_full_beats += int(full_beat)
        print(f"\n### {sym}  FULL: strat {r.total_return_pct:+7.0f}% vs B&H {bh:+7.0f}%  "
              f"{'BEAT' if full_beat else 'under'}  (DD {r.max_drawdown_pct:.0f}%, {r.num_trades} trades)")

        n = len(series)
        cs = n // CHUNKS
        for k in range(CHUNKS):
            chunk = series.iloc[k * cs:(k + 1) * cs].reset_index(drop=True)
            if len(chunk) < 210:
                continue
            rc = Backtest(st, MultiTA(st, threshold=THR)).run({sym: chunk}, n_trials=1, sample=f"w{k}")
            cc = chunk["close"].to_numpy(dtype=float)
            bhc = (cc[-1] / cc[0] - 1) * 100
            beat = rc.total_return_pct > bhc
            total_windows += 1
            window_beats += int(beat)
            print(f"   w{k}: strat {rc.total_return_pct:+7.0f}% vs B&H {bhc:+7.0f}%  "
                  f"{'BEAT' if beat else 'under'}  (DD {rc.max_drawdown_pct:.0f}%, {rc.num_trades} trades)")

    na = len(st.symbols)
    print(f"\n=== ROBUSTNESS ===")
    print(f"Full-cycle beats B&H on {asset_full_beats}/{na} assets.")
    print(f"Window-level: beats B&H in {window_beats}/{total_windows} windows "
          f"({window_beats/total_windows*100:.0f}%).")
    robust = asset_full_beats >= na - 1 and window_beats >= total_windows * 0.6
    print(f"Verdict: {'ROBUST edge (beats B&H across assets & most windows)' if robust else 'NOT robust (edge is patchy) — honest fail'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
