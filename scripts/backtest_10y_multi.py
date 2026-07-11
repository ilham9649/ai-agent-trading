#!/usr/bin/env python3
"""20-asset multi-crypto backtest: RobustTrend vs TrendLongFlat-SMA200 vs Buy-and-Hold.

Fetches real daily OHLCV from Binance (no auth) for top-20 cryptos by market cap,
backtests two strategies through the same engine used in production,
and reports per-asset + portfolio-level results with IS/OOS split.

  uv run python scripts/backtest_10y_multi.py

Bars per asset (best-effort, up to BARS):
  BTC, ETH, BNB, XRP, ADA, DOGE, LINK, LTC, TRX, XLM: ~3000+ bars (2017+)
  SOL, AVAX, DOT, MATIC, ATOM, NEAR, UNI: ~1500-2500 bars (2020+)
  SHIB, ICP, TON: ~1000-1500 bars (2021+)

Portfolio backtest truncates to the shortest common period.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_agent.backtest.engine import Backtest, BacktestResult  # noqa: E402
from trading_agent.backtest.feed import binance_series, is_oos_split  # noqa: E402
from trading_agent.config import Settings, UniverseItem, load_settings  # noqa: E402
from trading_agent.strategy.robust_trend import RobustTrend  # noqa: E402
from trading_agent.strategy.trend_long_flat import TrendLongFlat  # noqa: E402

BARS = 3650          # request up to 10 years daily
OOS_FRAC = 0.3       # last ~3 years = out-of-sample
N_TRIALS = 2         # strategies compared (for Deflated Sharpe)
DIV = "─" * 115

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE_20 = [
    # Original top-10 (longer history: 2017+)
    UniverseItem(symbol="BTC-USDC",  base="BTC",  quote="USDC", coingecko_id="bitcoin",          decimals=8),
    UniverseItem(symbol="ETH-USDC",  base="ETH",  quote="USDC", coingecko_id="ethereum",         decimals=18),
    UniverseItem(symbol="BNB-USDC",  base="BNB",  quote="USDC", coingecko_id="binancecoin",      decimals=18),
    UniverseItem(symbol="SOL-USDC",  base="SOL",  quote="USDC", coingecko_id="solana",           decimals=9),
    UniverseItem(symbol="XRP-USDC",  base="XRP",  quote="USDC", coingecko_id="ripple",           decimals=6),
    UniverseItem(symbol="AVAX-USDC", base="AVAX", quote="USDC", coingecko_id="avalanche-2",      decimals=18),
    UniverseItem(symbol="DOGE-USDC", base="DOGE", quote="USDC", coingecko_id="dogecoin",         decimals=8),
    UniverseItem(symbol="ADA-USDC",  base="ADA",  quote="USDC", coingecko_id="cardano",          decimals=6),
    UniverseItem(symbol="LINK-USDC", base="LINK", quote="USDC", coingecko_id="chainlink",        decimals=18),
    UniverseItem(symbol="DOT-USDC",  base="DOT",  quote="USDC", coingecko_id="polkadot",         decimals=10),
    # Extended top-20 (mix of 2017+ and 2020+)
    UniverseItem(symbol="TRX-USDC",  base="TRX",  quote="USDC", coingecko_id="tron",             decimals=6),
    UniverseItem(symbol="MATIC-USDC",base="MATIC",quote="USDC", coingecko_id="matic-network",    decimals=18),
    UniverseItem(symbol="LTC-USDC",  base="LTC",  quote="USDC", coingecko_id="litecoin",         decimals=8),
    UniverseItem(symbol="ATOM-USDC", base="ATOM", quote="USDC", coingecko_id="cosmos",           decimals=6),
    UniverseItem(symbol="NEAR-USDC", base="NEAR", quote="USDC", coingecko_id="near",             decimals=24),
    UniverseItem(symbol="UNI-USDC",  base="UNI",  quote="USDC", coingecko_id="uniswap",          decimals=18),
    UniverseItem(symbol="XLM-USDC",  base="XLM",  quote="USDC", coingecko_id="stellar",          decimals=7),
    UniverseItem(symbol="SHIB-USDC", base="SHIB", quote="USDC", coingecko_id="shiba-inu",        decimals=18),
    UniverseItem(symbol="ICP-USDC",  base="ICP",  quote="USDC", coingecko_id="internet-computer",decimals=8),
    UniverseItem(symbol="TON-USDC",  base="TON",  quote="USDC", coingecko_id="the-open-network", decimals=9),
]


# ── Settings factories ────────────────────────────────────────────────────────

def robust_settings(base: Settings) -> Settings:
    """RobustTrend settings: vol_target sizing, 7 concurrent positions, wide stops."""
    return base.model_copy(update={
        "universe": UNIVERSE_20,
        "strategy": base.strategy.model_copy(update={
            "name": "robust_trend",
            "stop_atr_mult": 2.5,
            "take_profit_atr_mult": 6.0,
            "donchian_period": 20,
            "volume_filter_mult": 1.0,
        }),
        "risk": base.risk.model_copy(update={
            "sizing_mode": "vol_target",
            "target_annual_vol_pct": 40.0,
            "max_concurrent_positions": 7,
            "max_position_notional_pct": 15.0,
            "max_deployed_pct": 90.0,
            "max_trades_per_day": 20,
        }),
    })


def trend_settings(base: Settings) -> Settings:
    """TrendLongFlat settings: fixed_fraction sizing (as verified in prior backtest)."""
    return base.model_copy(update={
        "universe": UNIVERSE_20,
        "strategy": base.strategy.model_copy(update={
            "name": "trend_long_flat",
            "stop_atr_mult": 3.0,
            "take_profit_atr_mult": 20.0,
        }),
        "risk": base.risk.model_copy(update={
            "sizing_mode": "fixed_fraction",
            "fixed_fraction_pct": 40.0,
            "max_concurrent_positions": 7,
            "max_position_notional_pct": 15.0,
            "max_deployed_pct": 90.0,
            "max_trades_per_day": 20,
        }),
        "execution": base.execution.model_copy(update={
            "slippage_bps": 10,
            "stop_slippage_bps": 30,
        }),
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def bnh(close: np.ndarray) -> tuple[float, float, float]:
    """Buy-and-hold return%, max-DD%, CAGR% (daily bars assumed)."""
    if len(close) < 2:
        return 0.0, 0.0, 0.0
    ret = (close[-1] / close[0] - 1) * 100
    running = np.maximum.accumulate(close)
    dd = float(abs(((close - running) / running).min()) * 100)
    years = max(len(close) / 365.25, 0.01)
    cagr = ((close[-1] / close[0]) ** (1 / years) - 1) * 100
    return ret, dd, cagr


def calmar(r: BacktestResult) -> float:
    return r.cagr_pct / r.max_drawdown_pct if r.max_drawdown_pct > 0 else 0.0


def fmt_row(label: str, sym: str, r: BacktestResult,
            bh_ret: float, bh_dd: float, bh_cagr: float) -> str:
    cal = calmar(r)
    return (
        f"  {label:<22} {sym:<12}"
        f"  ret={r.total_return_pct:+8.2f}%  cagr={r.cagr_pct:+6.2f}%"
        f"  DD={r.max_drawdown_pct:5.1f}%  cal={cal:5.2f}"
        f"  sharpe={r.sharpe:5.2f}  DSR={r.deflated_sharpe:.2f}"
        f"  trades={r.num_trades:4d}  eq=${r.end_equity:.0f}"
        f"  │ B&H {bh_ret:+7.1f}% cagr={bh_cagr:+5.1f}% DD={bh_dd:5.1f}%"
    )


def fmt_oos(label: str, r_is: BacktestResult, r_oos: BacktestResult) -> str:
    flag = "✓" if (r_is.sharpe <= 0 or r_oos.sharpe >= r_is.sharpe * 0.5) else "⚠"
    return (
        f"  {label:<22}  "
        f"IS  cagr={r_is.cagr_pct:+6.2f}%  DD={r_is.max_drawdown_pct:5.1f}%"
        f"  sharpe={r_is.sharpe:5.2f}  trades={r_is.num_trades:3d}  │  "
        f"OOS cagr={r_oos.cagr_pct:+6.2f}%  DD={r_oos.max_drawdown_pct:5.1f}%"
        f"  sharpe={r_oos.sharpe:5.2f}  trades={r_oos.num_trades:3d}  {flag}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    base = load_settings()
    rst = robust_settings(base)
    tst = trend_settings(base)
    universe = UNIVERSE_20

    print(DIV)
    print(f"  {len(universe)}-ASSET CRYPTO BACKTEST  (up to {BARS} daily bars ~10y)")
    print(f"  OOS={OOS_FRAC*100:.0f}% (last ~3y)  |  equity=${base.starting_equity_usd:.0f}")
    print(f"  Strategies: RobustTrend (Donchian20×10 + SMA100, vol_target)")
    print(f"              TrendLongFlat-SMA200 (fixed_fraction 40%)")
    print(DIV)

    # ── 1. Fetch all data in parallel ─────────────────────────────────────────
    print(f"\n  [1/4] Fetching data from Binance (parallel) …")

    def _fetch(u: UniverseItem) -> tuple[str, pd.DataFrame | None]:
        try:
            df = binance_series(rst, u.symbol, interval="1d", total=BARS)
            return u.symbol, df
        except Exception as exc:
            print(f"        {u.symbol:<14}  FAILED: {exc}")
            return u.symbol, None

    series: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch, u): u for u in universe}
        for fut in as_completed(futures):
            sym, df = fut.result()
            if df is not None:
                series[sym] = df
                t0 = pd.to_datetime(df["ts"].iloc[0],  unit="ms").strftime("%Y-%m-%d")
                t1 = pd.to_datetime(df["ts"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
                print(f"        {sym:<14}  {len(df):4d} bars  ({t0} → {t1})")

    syms = [u.symbol for u in universe if u.symbol in series]
    min_bars = min(len(series[s]) for s in syms)
    max_bars = max(len(series[s]) for s in syms)
    print(f"\n  Bars range: {min_bars} – {max_bars}  "
          f"(portfolio backtest uses {min_bars} bars ~{min_bars/365:.1f}y)")

    # ── 2. Per-asset comparison ───────────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  [2/4] PER-ASSET: RobustTrend vs TrendLongFlat-SMA200 vs Buy-and-Hold")
    print(DIV)

    per_asset: list[dict] = []
    for sym in syms:
        df = series[sym]
        close = df["close"].to_numpy(dtype=float)
        bh_ret, bh_dd, bh_cagr = bnh(close)

        r_rt = Backtest(rst, RobustTrend(rst)).run({sym: df}, n_trials=N_TRIALS, sample="full")
        r_tl = Backtest(tst, TrendLongFlat(tst, sma_len=200)).run({sym: df}, n_trials=N_TRIALS, sample="full")

        print(f"\n  {'━'*40}  {sym}  ({'{'}{len(df)} bars{'}'})  {'━'*40}")
        print(fmt_row("RobustTrend",         sym, r_rt, bh_ret, bh_dd, bh_cagr))
        print(fmt_row("TrendLongFlat-SMA200", sym, r_tl, bh_ret, bh_dd, bh_cagr))

        per_asset.append(dict(
            sym=sym, bars=len(df),
            rt_ret=r_rt.total_return_pct, rt_cagr=r_rt.cagr_pct,
            rt_dd=r_rt.max_drawdown_pct, rt_sharpe=r_rt.sharpe, rt_dsr=r_rt.deflated_sharpe,
            rt_trades=r_rt.num_trades, rt_eq=r_rt.end_equity, rt_cal=calmar(r_rt),
            tl_ret=r_tl.total_return_pct, tl_cagr=r_tl.cagr_pct,
            tl_dd=r_tl.max_drawdown_pct, tl_sharpe=r_tl.sharpe, tl_dsr=r_tl.deflated_sharpe,
            tl_trades=r_tl.num_trades, tl_eq=r_tl.end_equity, tl_cal=calmar(r_tl),
            bh_ret=bh_ret, bh_cagr=bh_cagr, bh_dd=bh_dd,
        ))

    # ── 3. IS / OOS per asset (with pass-rate tally) ──────────────────────────
    print(f"\n{DIV}")
    print(f"  [3/4] IN-SAMPLE vs OUT-OF-SAMPLE (overfitting check)")
    print(f"  IS=first {(1-OOS_FRAC)*100:.0f}%  OOS=last {OOS_FRAC*100:.0f}%   ✓=OOS Sharpe ≥50% IS")
    print(DIV)

    rt_oos_pass = tl_oos_pass = 0
    for sym in syms:
        df = series[sym]
        in_s, oos = is_oos_split({sym: df}, OOS_FRAC)

        r_rt_is  = Backtest(rst, RobustTrend(rst)).run(in_s, n_trials=N_TRIALS, sample="IS")
        r_rt_oos = Backtest(rst, RobustTrend(rst)).run(oos, n_trials=N_TRIALS, sample="OOS")
        r_tl_is  = Backtest(tst, TrendLongFlat(tst, sma_len=200)).run(in_s, n_trials=N_TRIALS, sample="IS")
        r_tl_oos = Backtest(tst, TrendLongFlat(tst, sma_len=200)).run(oos, n_trials=N_TRIALS, sample="OOS")

        print(f"\n  {sym}")
        print(fmt_oos("RobustTrend",          r_rt_is, r_rt_oos))
        print(fmt_oos("TrendLongFlat-SMA200",  r_tl_is, r_tl_oos))

        if r_rt_is.sharpe <= 0 or r_rt_oos.sharpe >= r_rt_is.sharpe * 0.5:
            rt_oos_pass += 1
        if r_tl_is.sharpe <= 0 or r_tl_oos.sharpe >= r_tl_is.sharpe * 0.5:
            tl_oos_pass += 1

    # ── 4. Portfolio backtest (all assets together) ───────────────────────────
    print(f"\n{DIV}")
    print(f"  [4/4] PORTFOLIO BACKTEST — all assets traded simultaneously")
    print(f"  (min common period: {min_bars} bars ~{min_bars/365:.1f}y)")
    print(DIV)

    r_rt_port = Backtest(rst, RobustTrend(rst)).run(series, n_trials=N_TRIALS, sample="portfolio")
    r_tl_port = Backtest(tst, TrendLongFlat(tst, sma_len=200)).run(series, n_trials=N_TRIALS, sample="portfolio")

    avg_bh_ret  = np.mean([d["bh_ret"]  for d in per_asset])
    avg_bh_cagr = np.mean([d["bh_cagr"] for d in per_asset])
    avg_bh_dd   = np.mean([d["bh_dd"]   for d in per_asset])

    print(f"\n  {'Strategy':<25}  {'Return':>9}  {'CAGR':>8}  {'MaxDD':>7}  "
          f"{'Calmar':>7}  {'Sharpe':>8}  {'DSR':>5}  {'Trades':>7}  {'End equity':>12}")
    print(f"  {'─'*25}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*12}")

    def prow(name: str, r: BacktestResult) -> None:
        cal = calmar(r)
        print(f"  {name:<25}  {r.total_return_pct:+8.2f}%  {r.cagr_pct:+7.2f}%  "
              f"{r.max_drawdown_pct:6.1f}%  {cal:7.2f}  {r.sharpe:+7.2f}  "
              f"{r.deflated_sharpe:5.2f}  {r.num_trades:7d}  ${r.end_equity:11.2f}")

    prow("RobustTrend",          r_rt_port)
    prow("TrendLongFlat-SMA200", r_tl_port)
    print(f"  {'EqualWeight B&H':<25}  {avg_bh_ret:+8.1f}%  {avg_bh_cagr:+7.1f}%  "
          f"{avg_bh_dd:6.1f}%  {'—':>7}  {'—':>8}  {'—':>5}  {'—':>7}  {'—':>12}")

    # ── Summary scorecard ─────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  SCORECARD — RobustTrend vs TrendLongFlat-SMA200  (avg across {len(syms)}/{len(universe)} assets)")
    print(DIV)

    df_pa = pd.DataFrame(per_asset)
    metrics = [
        ("Avg CAGR %",        "rt_cagr",   "tl_cagr",   "higher"),
        ("Avg Max DD %",      "rt_dd",     "tl_dd",     "lower"),
        ("Avg Sharpe",        "rt_sharpe", "tl_sharpe", "higher"),
        ("Avg Calmar",        "rt_cal",    "tl_cal",    "higher"),
        ("Avg DSR",           "rt_dsr",    "tl_dsr",    "higher"),
        ("Avg trades/asset",  "rt_trades", "tl_trades", "—"),
    ]
    print(f"\n  {'Metric':<24} {'RobustTrend':>14} {'TrendLF-200':>14}  Winner")
    print(f"  {'─'*24} {'─'*14} {'─'*14}  {'─'*10}")
    for label, rc, tc, direction in metrics:
        rv, tv = df_pa[rc].mean(), df_pa[tc].mean()
        if direction == "higher":
            win = "RobustTrend" if rv > tv else "TrendLF-200"
        elif direction == "lower":
            win = "RobustTrend" if rv < tv else "TrendLF-200"
        else:
            win = "—"
        print(f"  {label:<24} {rv:14.3f} {tv:14.3f}  {win}")

    rt_wins = sum(1 for d in per_asset if d["rt_sharpe"] > d["tl_sharpe"])
    n = len(syms)
    print(f"\n  RobustTrend has higher Sharpe on {rt_wins}/{n} assets")
    print(f"  OOS robustness: RobustTrend {rt_oos_pass}/{n}  |  TrendLF-200 {tl_oos_pass}/{n}")

    print(f"\n{DIV}")
    print("  NOTE: Backtest = deterministic shell only. GLM overlay is forward-tested in paper mode.")
    print("  Green OOS = necessary but not sufficient. Always paper-trade before going live.")
    print(DIV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
