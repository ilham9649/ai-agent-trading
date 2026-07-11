"""RSI mean-reversion: buy oversold pullbacks WITHIN an uptrend (close>trend-EMA),
exit on RSI recovery. Avoids catching falling knives by requiring the trend filter."""
from __future__ import annotations

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Side, Signal
from trading_agent.strategy.base import _D, compute_indicators, snapshot_from_row

RSI_BUY = 30
RSI_EXIT = 55


class MeanReversionStrategy:
    name = "mean_reversion"

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
        price, atr = snap.price, snap.atr
        uptrend = last["close"] > last["ema_trend"]
        if not in_position:
            if last["rsi"] < RSI_BUY and uptrend:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=price - _D(self.s.strategy.stop_atr_mult) * atr,
                    take_profit=price + _D(self.s.strategy.take_profit_atr_mult) * atr,
                    rationale=f"rsi<{RSI_BUY} pullback in uptrend",
                )
            return snap, Signal(action="hold", symbol=symbol)
        if last["rsi"] > RSI_EXIT or not uptrend:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale=f"rsi>{RSI_EXIT} / trend broke")
        return snap, Signal(action="hold", symbol=symbol)
