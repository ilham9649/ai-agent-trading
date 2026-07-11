"""Shared indicator computation + Strategy protocol.

All strategies expose ``analyze(symbol, df_window, in_position) -> (MarketSnapshot, Signal)``
where ``df_window`` is OHLCV up to (and not including) the bar being traded — so any
indicator computed on the window is point-in-time correct (no look-ahead).
"""
from __future__ import annotations

import math
from typing import Protocol

import pandas as pd

from decimal import Decimal

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Signal, Trend, utcnow


def _D(x):
    return Decimal(str(x))


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def compute_indicators(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Rich indicator set on an OHLCV window: EMAs, ATR, Donchian, RSI, ROC, vol avg."""
    s = settings.strategy
    d = df.copy()
    d["ema_fast"] = d["close"].ewm(span=s.ema_fast, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=s.ema_slow, adjust=False).mean()
    d["ema_trend"] = d["close"].ewm(span=50, adjust=False).mean()
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - d["close"].shift(1)).abs(),
            (d["low"] - d["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.rolling(s.atr_period).mean()
    d["donchian_high"] = d["high"].shift(1).rolling(s.donchian_period).max()
    d["donchian_low"] = d["low"].shift(1).rolling(s.donchian_period).min()

    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    loss_nz = loss.replace(0, pd.NA)
    rs = gain / loss_nz
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    d["roc"] = d["close"].pct_change(10)
    d["vol_avg"] = d["volume"].rolling(20).mean() if "volume" in d else 0.0
    return d


def snapshot_from_row(symbol: str, row: pd.Series) -> MarketSnapshot:
    price = _D(str(float(row["close"])))
    atr = _D(str(float(row["atr"]))) if _finite(row.get("atr")) else price * _D("0.01")
    ema_fast = _D(str(float(row["ema_fast"]))) if _finite(row.get("ema_fast")) else price
    ema_slow = _D(str(float(row["ema_slow"]))) if _finite(row.get("ema_slow")) else price
    dhigh = _D(str(float(row["donchian_high"]))) if _finite(row.get("donchian_high")) else price * _D("1.01")
    dlow = _D(str(float(row["donchian_low"]))) if _finite(row.get("donchian_low")) else price * _D("0.99")
    trend: Trend = "up" if ema_fast > ema_slow else ("down" if ema_fast < ema_slow else "range")
    return MarketSnapshot(
        symbol=symbol, price=price, atr=atr, ema_fast=ema_fast, ema_slow=ema_slow,
        donchian_high=dhigh, donchian_low=dlow, trend=trend, ts=utcnow(),
    )


class Strategy(Protocol):
    name: str
    need: int

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool,
                position: Position | None = None) -> tuple[MarketSnapshot, Signal]: ...
