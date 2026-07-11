#!/usr/bin/env python3
"""UNLEVERED multi-asset trend portfolio vs buy-and-hold, targeting LOW drawdown.

Each asset gets a fixed 1/N slice when its multi-TA signal is bullish, else stables.
Unlevered (sum of weights <= 1). Diversification + trend filter -> flat in correlated
crashes (low DD) while capturing upside across assets. Paper-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402
from trading_agent.strategy.multi_ta_signal import multi_ta_signal  # noqa: E402

ASSETS = ["WETH-USDC", "cbBTC-USDC", "SOL-USDC", "BNB-USDC"]
TOTAL = 3000
FEE = 0.001


def metrics(eq: np.ndarray) -> tuple[float, float]:
    rmax = np.maximum.accumulate(eq)
    dd = abs(((eq - rmax) / rmax).min() * 100)
    return (eq[-1] / eq[0] - 1) * 100, dd


def run(w: np.ndarray, rets: np.ndarray) -> np.ndarray:
    port_ret = (w * rets).sum(axis=1)
    turnover = np.abs(np.diff(w, axis=0, prepend=w[:1])).sum(axis=1)
    eq = np.insert(np.cumprod(1 + port_ret - turnover * FEE), 0, 1.0)
    return eq


def main() -> int:
    base = load_settings()
    base.universe = list(base.universe) + [
        UniverseItem(symbol="SOL-USDC", base="SOL", quote="USDC", coingecko_id="solana", decimals=6),
        UniverseItem(symbol="BNB-USDC", base="BNB", quote="USDC", coingecko_id="binancecoin", decimals=18),
    ]
    series = {s: binance_series(base, s, interval="1d", total=TOTAL) for s in ASSETS}
    minlen = min(len(series[s]) for s in ASSETS)
    for s in ASSETS:
        series[s] = series[s].tail(minlen).reset_index(drop=True)

    T = minlen - 1
    sig_mat = np.column_stack([multi_ta_signal(series[s], 4).to_numpy()[:T] for s in ASSETS])  # (T, N)
    rets = np.column_stack([np.diff(series[s]["close"].to_numpy(dtype=float)) /
                            series[s]["close"].to_numpy(dtype=float)[:-1] for s in ASSETS])  # (T, N)
    N = len(ASSETS)

    # portfolio: fixed 1/N per active asset
    w_fixed = sig_mat / N
    eq_fixed = run(w_fixed, rets)
    # portfolio: equal-weight among active (sum=1)
    rowsum = sig_mat.sum(axis=1, keepdims=True)
    w_active = np.where(rowsum > 0, sig_mat / np.maximum(rowsum, 1.0), 0.0)
    eq_active = run(w_active, rets)
    # equal-weight B&H basket (rebalanced)
    eq_basket = run(np.full((T, N), 1.0 / N), rets)
    # single-asset B&H (constant 100% in one asset)
    singles = {ASSETS[i]: run(np.eye(N)[i][None, :] * np.ones((T, 1)), rets) for i in range(N)}

    print(f"Unlevered portfolios over {minlen} daily bars (common range), fee {FEE*100:.1f}%/turnover\n")
    print(f"{'strategy':34} {'return':>9} {'maxDD':>7}")
    for name, eq in [("PORTFOLIO fixed 1/N trend", eq_fixed),
                     ("PORTFOLIO equal-active trend", eq_active),
                     ("B&H equal-weight basket (rebal)", eq_basket)]:
        ret, dd = metrics(eq)
        print(f"{name:34} {ret:+8.0f}% {dd:6.1f}%")
    print("-" * 52)
    for s in ASSETS:
        ret, dd = metrics(singles[s])
        print(f"{'B&H ' + s:34} {ret:+8.0f}% {dd:6.1f}%")

    pf_ret, pf_dd = metrics(eq_fixed)
    basket_ret, basket_dd = metrics(eq_basket)
    btc_ret = metrics(singles["cbBTC-USDC"])[0]
    print(f"\nPORTFOLIO fixed-1/N: beats B&H basket? {'YES' if pf_ret>basket_ret else 'no'} "
          f"({pf_ret:+.0f}% vs {basket_ret:+.0f}%); beats BTC B&H? {'YES' if pf_ret>btc_ret else 'no'}; "
          f"maxDD {pf_dd:.0f}% (basket {basket_dd:.0f}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
