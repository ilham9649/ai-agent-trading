#!/usr/bin/env python3
"""EDGE test: does vol-targeted momentum beat buy-and-hold on a RISK-ADJUSTED basis?

Fair comparison — both get the same capital; we compare Sharpe/Sortino/Calmar
(the legitimate definition of 'edge over B&H'), not just raw return. Full ~8y cycle.
Paper-only. SUCCESS BAR: strategy Sharpe AND Calmar > buy-and-hold.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from trading_agent.backtest.engine import Backtest  # noqa: E402
from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.strategy.trend_long_flat import TrendLongFlat  # noqa: E402

TOTAL = 3000
TARGETS = [30, 40, 50]


def bnh_metrics(close: np.ndarray) -> dict:
    c = np.asarray(close, float)
    rets = np.diff(c) / c[:-1]
    n = len(rets)
    ret = (c[-1] / c[0] - 1) * 100
    years = n / 365
    cagr = ((c[-1] / c[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    mean, std = rets.mean(), rets.std(ddof=1)
    sharpe = mean / std * math.sqrt(365) if std > 0 else 0.0
    downside = rets[rets < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = mean / dstd * math.sqrt(365) if dstd > 0 else 0.0
    rmax = np.maximum.accumulate(c)
    dd = abs(((c - rmax) / rmax).min() * 100)
    calmar = cagr / dd if dd > 0 else 0.0
    return dict(ret=ret, cagr=cagr, sharpe=sharpe, sortino=sortino, dd=dd, calmar=calmar)


def strat_metrics(r, close_len) -> dict:
    dd = r.max_drawdown_pct or 1e-9
    return dict(ret=r.total_return_pct, cagr=r.cagr_pct, sharpe=r.sharpe,
                sortino=r.sortino, dd=r.max_drawdown_pct, calmar=r.cagr_pct / dd,
                trades=r.num_trades)


def main() -> int:
    base = load_settings()
    syms = base.symbols
    series = {sym: binance_series(base, sym, interval="1d", total=TOTAL) for sym in syms}
    bnh = {sym: bnh_metrics(series[sym]["close"].to_numpy()) for sym in syms}
    print(f"Full-cycle daily ({ {s: len(series[s]) for s in syms} })\n")

    for tgt in TARGETS:
        st = base.model_copy(update={
            "strategy": base.strategy.model_copy(update={"name": "trend_long_flat", "timeframe": "1d",
                                                          "stop_atr_mult": 3.0, "take_profit_atr_mult": 20.0}),
            "execution": base.execution.model_copy(update={"slippage_bps": 10, "stop_slippage_bps": 30}),
            "risk": base.risk.model_copy(update={"sizing_mode": "vol_target", "target_annual_vol_pct": float(tgt),
                                                  "max_position_notional_pct": 100.0, "max_deployed_pct": 100.0}),
        })
        print(f"=== TrendLongFlat + vol_target {tgt}% ===")
        print(f"{'asset':11} {'side':>5} {'ret':>8} {'cagr':>6} {'sharpe':>6} {'sortino':>7} {'calmar':>6} {'maxDD':>6} {'trades':>6}")
        agg_s = {k: 0.0 for k in ["ret", "cagr", "sharpe", "sortino", "calmar", "dd"]}
        agg_b = {k: 0.0 for k in agg_s}
        for sym in syms:
            r = Backtest(st, TrendLongFlat(st)).run({sym: series[sym]}, n_trials=3, sample="full")
            sm = strat_metrics(r, len(series[sym]))
            bm = bnh[sym]
            for k in agg_s:
                agg_s[k] += sm[k]; agg_b[k] += bm[k]
            print(f"{sym:11} {'STR':>5} {sm['ret']:+7.1f}% {sm['cagr']:+5.1f}% {sm['sharpe']:6.2f} "
                  f"{sm['sortino']:7.2f} {sm['calmar']:6.2f} {sm['dd']:5.1f}% {sm['trades']:6d}")
            print(f"{'':11} {'B&H':>5} {bm['ret']:+7.1f}% {bm['cagr']:+5.1f}% {bm['sharpe']:6.2f} "
                  f"{bm['sortino']:7.2f} {bm['calmar']:6.2f} {bm['dd']:5.1f}%")
        n = len(syms)
        edge_sharpe = agg_s["sharpe"] / n > agg_b["sharpe"] / n
        edge_calmar = agg_s["calmar"] / n > agg_b["calmar"] / n
        print(f"{'AVG':11} STR sharpe={agg_s['sharpe']/n:.2f} calmar={agg_s['calmar']/n:.2f} dd={agg_s['dd']/n:.1f}%  |  "
              f"B&H sharpe={agg_b['sharpe']/n:.2f} calmar={agg_b['calmar']/n:.2f} dd={agg_b['dd']/n:.1f}%")
        print(f"  -> beats B&H on Sharpe? {'YES' if edge_sharpe else 'no'}   on Calmar? {'YES' if edge_calmar else 'no'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
