"""Long/flat trend-following: LONG when close > SMA(N), in stables otherwise.

The most robust documented 'edge' available to retail in crypto is the secular drift
plus drawdown control — ride bull markets, exit to stables in bears. Uses a wide stop
(set via settings: stop_atr_mult ~3) so normal pullbacks don't shake it out; the SMA
cross is the real risk control. Designed to capture most of buy-and-hold's return at
a fraction of its drawdown over full cycles.
"""
from __future__ import annotations

import math

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Side, Signal
from trading_agent.strategy.base import _D, compute_indicators, snapshot_from_row


class TrendLongFlat:
    def __init__(self, settings: Settings, sma_len: int = 200):
        self.s = settings
        self.sma_len = sma_len
        self.need = sma_len + 5
        self.name = f"trend_longflat_sma{sma_len}"

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool,
                position: Position | None = None) -> tuple[MarketSnapshot, Signal]:
        if len(df) < self.need:
            return snapshot_from_row(symbol, df.iloc[-1].to_dict() if len(df) else pd.Series({"close": 0})), \
                Signal(action="hold", symbol=symbol, rationale="warmup")
        ind = compute_indicators(df, self.s)
        snap = snapshot_from_row(symbol, ind.iloc[-1])
        sma = df["close"].rolling(self.sma_len).mean().iloc[-1]
        close = float(df["close"].iloc[-1])
        if sma is None or (isinstance(sma, float) and math.isnan(sma)):
            return snap, Signal(action="hold", symbol=symbol)
        bull = close > float(sma)
        price, atr = snap.price, snap.atr
        if not in_position:
            if bull:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=price - _D("3") * atr, take_profit=price + _D("20") * atr,
                    rationale=f"close>{self.sma_len}-SMA (uptrend)",
                )
            return snap, Signal(action="hold", symbol=symbol)
        if not bull:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale=f"close<{self.sma_len}-SMA (trend broke)")
        return snap, Signal(action="hold", symbol=symbol)
