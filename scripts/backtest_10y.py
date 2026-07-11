#!/usr/bin/env python3
"""10-year backtest (2015-2025, daily bars) — multi-strategy comparison.

Fetches ~3650 real daily OHLCV bars from Binance for BTC and ETH,
then runs every deterministic strategy through the same backtest engine
used in production (same Guard, PaperExecutor, Portfolio).

  uv run python scripts/backtest_10y.py

No API key required. GLM overlay is excluded by design (forward-tested separately).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_agent.backtest.engine import Backtest, BacktestResult  # noqa: E402
from trading_agent.backtest.feed import binance_series, is_oos_split  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.ema_cross import EmaCrossStrategy  # noqa: E402
from trading_agent.strategy.mean_reversion import MeanReversionStrategy  # noqa: E402
from trading_agent.strategy.momentum import MomentumStrategy  # noqa: E402
from trading_agent.strategy.multi_ta import MultiTA  # noqa: E402
from trading_agent.strategy.trend_long_flat import TrendLongFlat  # noqa: E402

BARS = 3650          # ~10 years daily
OOS_FRAC = 0.3       # last 3 years = out-of-sample
N_TRIALS = 6         # number of strategies tested (for Deflated Sharpe)

DIVIDER = "─" * 110


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def bnh(close: np.ndarray) -> tuple[float, float, float]:
    """Buy-and-hold return %, max drawdown %, CAGR% (assumes daily bars)."""
    ret = (close[-1] / close[0] - 1) * 100
    running = np.maximum.accumulate(close)
    dd = float(abs(((close - running) / running).min()) * 100)
    years = len(close) / 365.25
    cagr = ((close[-1] / close[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    return ret, dd, cagr


def fmt_result(name: str, sym: str, r: BacktestResult,
               bh_ret: float, bh_dd: float, bh_cagr: float) -> str:
    pf = f"{r.profit_factor:5.2f}" if r.profit_factor != float("inf") else "  inf"
    cap = r.end_equity
    return (
        f"  {name:<22} {sym:<12}"
        f"  ret={r.total_return_pct:+8.2f}%  cagr={r.cagr_pct:+6.2f}%"
        f"  maxDD={r.max_drawdown_pct:5.1f}%"
        f"  sharpe={r.sharpe:5.2f}  sortino={r.sortino:5.2f}"
        f"  win={r.win_rate:4.1f}%  PF={pf}"
        f"  trades={r.num_trades:4d}  fees=${r.total_fees:6.2f}"
        f"  DSR={r.deflated_sharpe:.2f}"
        f"  equity=${cap:.0f}"
        f"  │ B&H ret={bh_ret:+8.2f}% cagr={bh_cagr:+5.2f}% DD={bh_dd:5.1f}%"
    )


def fmt_oos(r_is: BacktestResult, r_oos: BacktestResult) -> str:
    flag = "✓" if r_oos.sharpe >= r_is.sharpe * 0.5 else "⚠"
    return (
        f"    IS : ret={r_is.total_return_pct:+7.2f}%  cagr={r_is.cagr_pct:+6.2f}%"
        f"  maxDD={r_is.max_drawdown_pct:5.1f}%  sharpe={r_is.sharpe:5.2f}"
        f"  trades={r_is.num_trades:3d}\n"
        f"    OOS: ret={r_oos.total_return_pct:+7.2f}%  cagr={r_oos.cagr_pct:+6.2f}%"
        f"  maxDD={r_oos.max_drawdown_pct:5.1f}%  sharpe={r_oos.sharpe:5.2f}"
        f"  trades={r_oos.num_trades:3d}  {flag} (OOS Sharpe >= 50% IS)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Strategy factory  (one fresh instance per run — engine is stateful)
# ──────────────────────────────────────────────────────────────────────────────

def make_strategies(st):
    """Return list of (label, strategy_instance) built from settings."""
    trend_st = st.model_copy(update={
        "strategy": st.strategy.model_copy(update={
            "stop_atr_mult": 3.0,
            "take_profit_atr_mult": 20.0,
        }),
        "execution": st.execution.model_copy(update={
            "slippage_bps": 10,
            "stop_slippage_bps": 30,
        }),
        "risk": st.risk.model_copy(update={
            "sizing_mode": "fixed_fraction",
            "fixed_fraction_pct": 45.0,
            "max_position_notional_pct": 60.0,
            "max_deployed_pct": 100.0,
        }),
    })
    return [
        ("TrendLongFlat-SMA200",   trend_st, TrendLongFlat(trend_st, sma_len=200)),
        ("TrendLongFlat-SMA150",   trend_st, TrendLongFlat(trend_st, sma_len=150)),
        ("TrendLongFlat-SMA100",   trend_st, TrendLongFlat(trend_st, sma_len=100)),
        ("EmaCross",               st,       EmaCrossStrategy(st)),
        ("MeanReversion",          st,       MeanReversionStrategy(st)),
        ("Momentum",               st,       MomentumStrategy(st)),
        ("MultiTA",                st,       MultiTA(st)),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    base = load_settings()

    print(DIVIDER)
    print(f"  10-YEAR BACKTEST  ({BARS} daily bars, ~{BARS/365:.1f}y)  |  "
          f"OOS={OOS_FRAC*100:.0f}% (last ~3y)  |  starting equity=${base.starting_equity_usd:.0f}")
    print(DIVIDER)

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    print("\n  [1/3] Fetching historical data from Binance …")
    series: dict[str, pd.DataFrame] = {}
    for sym in base.symbols:
        print(f"        {sym} … ", end="", flush=True)
        df = binance_series(base, sym, interval="1d", total=BARS)
        series[sym] = df
        ts_start = pd.to_datetime(df["ts"].iloc[0], unit="ms").strftime("%Y-%m-%d")
        ts_end   = pd.to_datetime(df["ts"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
        print(f"{len(df)} bars  ({ts_start} → {ts_end})")
        time.sleep(0.4)   # gentle rate-limit

    in_s, oos = is_oos_split(series, OOS_FRAC)
    print(f"\n  IS  : {len(list(in_s.values())[0])} bars  "
          f"(~{len(list(in_s.values())[0])/365:.1f}y)")
    print(f"  OOS : {len(list(oos.values())[0])} bars  "
          f"(~{len(list(oos.values())[0])/365:.1f}y)\n")

    # ── 2. Full-period results ─────────────────────────────────────────────────
    print(DIVIDER)
    print("  [2/3] FULL PERIOD — all strategies vs Buy-and-Hold")
    print(DIVIDER)

    for sym in base.symbols:
        close = series[sym]["close"].to_numpy(dtype=float)
        bh_ret, bh_dd, bh_cagr = bnh(close)

        print(f"\n  {'━'*50}  {sym}  {'━'*50}")
        for label, st, strat in make_strategies(base):
            r = Backtest(st, strat).run({sym: series[sym]}, n_trials=N_TRIALS, sample="full")
            print(fmt_result(label, sym, r, bh_ret, bh_dd, bh_cagr))

    # ── 3. IS / OOS split ─────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  [3/3] IN-SAMPLE vs OUT-OF-SAMPLE  (overfitting check)")
    print(DIVIDER)

    for sym in base.symbols:
        print(f"\n  {'━'*50}  {sym}  {'━'*50}")
        for label, st, strat in make_strategies(base):
            r_is  = Backtest(st, strat).run({sym: in_s[sym]},  n_trials=N_TRIALS, sample="in-sample")
            strat2 = make_strategies(base)[[x[0] for x in make_strategies(base)].index(label)][2]
            r_oos = Backtest(st, strat2).run({sym: oos[sym]}, n_trials=N_TRIALS, sample="out-of-sample")
            print(f"\n  ── {label} ({sym}) ──")
            print(fmt_oos(r_is, r_oos))

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  SUMMARY — best strategy per metric (full period, across all symbols)")
    print(DIVIDER)

    records = []
    for sym in base.symbols:
        for label, st, strat in make_strategies(base):
            r = Backtest(st, strat).run({sym: series[sym]}, n_trials=N_TRIALS, sample="full")
            records.append({
                "strategy": label, "symbol": sym,
                "ret%": round(r.total_return_pct, 2),
                "cagr%": round(r.cagr_pct, 2),
                "maxDD%": round(r.max_drawdown_pct, 2),
                "sharpe": round(r.sharpe, 3),
                "DSR": round(r.deflated_sharpe, 3),
                "trades": r.num_trades,
                "equity$": round(r.end_equity, 2),
            })

    df_summary = pd.DataFrame(records)
    df_summary["ret/dd"] = (df_summary["ret%"] / df_summary["maxDD%"].replace(0, np.nan)).round(2)

    # Best by Sharpe across symbols
    by_sharpe = df_summary.groupby("strategy")["sharpe"].mean().sort_values(ascending=False)
    by_ret    = df_summary.groupby("strategy")["ret%"].mean().sort_values(ascending=False)
    by_dd     = df_summary.groupby("strategy")["maxDD%"].mean().sort_values()
    by_dsr    = df_summary.groupby("strategy")["DSR"].mean().sort_values(ascending=False)

    print("\n  Avg Sharpe (↑ better):")
    for s, v in by_sharpe.items():
        print(f"    {s:<25} {v:+.3f}")
    print("\n  Avg Return % (↑ better):")
    for s, v in by_ret.items():
        print(f"    {s:<25} {v:+.2f}%")
    print("\n  Avg Max Drawdown % (↓ better):")
    for s, v in by_dd.items():
        print(f"    {s:<25} {v:.1f}%")
    print("\n  Avg Deflated Sharpe (↑ better, selection-bias-adjusted):")
    for s, v in by_dsr.items():
        print(f"    {s:<25} {v:.3f}")

    print(f"\n{DIVIDER}")
    print("  NOTE: Backtest tests the DETERMINISTIC SHELL only. GLM overlay is")
    print("  forward-tested in paper mode. Green backtest = necessary, not sufficient.")
    print(DIVIDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
