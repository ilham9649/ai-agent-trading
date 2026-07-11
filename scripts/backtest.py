#!/usr/bin/env python3
"""Backtest the deterministic swing shell.

  uv run python scripts/backtest.py --simulated --bars 1500           # offline
  uv run python scripts/backtest.py --bars 400 --oos 0.3               # CoinGecko history
  uv run python scripts/backtest.py --simulated --trials 50 --oos 0.3  # deflated Sharpe + IS/OOS

NOTE: this backtests the DETERMINISTIC shell only. The GLM overlay is forward-tested
separately (it cannot be reliably backtested). A green backtest is necessary but not
sufficient — always forward/paper-test before live.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.backtest.engine import Backtest, BacktestResult  # noqa: E402
from trading_agent.backtest.feed import coingecko_series, is_oos_split, synthetic_series  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402


def fmt(r: BacktestResult) -> str:
    pf = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
    return (
        f"[{r.sample:<12}] bars={r.bars:<5} trades={r.num_trades:<3} "
        f"ret={r.total_return_pct:+7.2f}%  cagr={r.cagr_pct:+6.2f}%  "
        f"maxDD={r.max_drawdown_pct:5.2f}%  sharpe={r.sharpe:5.2f}  sortino={r.sortino:5.2f}  "
        f"win={r.win_rate:4.1f}%  PF={pf:>5}  exp={r.expectancy_usd:+6.2f}  "
        f"fees=${r.total_fees:6.2f}  DSR={r.deflated_sharpe:.2f}@{r.n_trials}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backtest the deterministic swing shell")
    p.add_argument("--simulated", action="store_true", help="use synthetic data (offline)")
    p.add_argument("--bars", type=int, default=1500)
    p.add_argument("--symbol", default=None, help="single symbol (default: whole universe)")
    p.add_argument("--trials", type=int, default=1, help="# strategy variants tried (for Deflated Sharpe)")
    p.add_argument("--oos", type=float, default=0.3, help="OOS fraction (0 to disable IS/OOS split)")
    args = p.parse_args(argv)

    settings = load_settings()
    syms = [args.symbol] if args.symbol else None
    series = (
        synthetic_series(settings, bars=args.bars, symbols=syms)
        if args.simulated
        else coingecko_series(settings, bars=args.bars, symbols=syms)
    )

    print(fmt(Backtest(settings).run(series, n_trials=args.trials, sample="full")))
    if args.oos > 0:
        in_s, oos = is_oos_split(series, args.oos)
        print(fmt(Backtest(settings).run(in_s, n_trials=args.trials, sample="in-sample")))
        print(fmt(Backtest(settings).run(oos, n_trials=args.trials, sample="out-of-sample")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
