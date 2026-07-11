"""EMA-cross trend-following: long when fast>slow AND price>long-EMA (trend filter)."""
from __future__ import annotations

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Side, Signal
from trading_agent.strategy.base import _D, compute_indicators, snapshot_from_row


class EmaCrossStrategy:
    name = "ema_cross"

    def __init__(self, settings: Settings):
        self.s = settings
        self.need = 55

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool,
                position: Position | None = None) -> tuple[MarketSnapshot, Signal]:
        if len(df) < self.need:
            return snapshot_from_row(symbol, df.iloc[-1].to_dict() if len(df) else pd.Series({"close": 0})), \
                Signal(action="hold", symbol=symbol, rationale="insufficient history")
        d = compute_indicators(df, self.s)
        last = d.iloc[-1]
        snap = snapshot_from_row(symbol, last)
        price = snap.price
        atr = snap.atr
        aligned = last["ema_fast"] > last["ema_slow"] and last["close"] > last["ema_trend"]
        if not in_position:
            if aligned:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=price - _D(self.s.strategy.stop_atr_mult) * atr,
                    take_profit=price + _D(self.s.strategy.take_profit_atr_mult) * atr,
                    rationale="ema_fast>slow & price>trend",
                )
            return snap, Signal(action="hold", symbol=symbol)
        # exit on cross-down or trend break
        if last["ema_fast"] < last["ema_slow"] or last["close"] < last["ema_trend"]:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale="ema cross-down / trend break")
        return snap, Signal(action="hold", symbol=symbol)
