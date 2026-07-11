"""ROC momentum: long on positive 10-bar rate-of-change above the trend-EMA,
exit when momentum rolls over."""
from __future__ import annotations

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Side, Signal
from trading_agent.strategy.base import _D, compute_indicators, snapshot_from_row

ROC_BUY = 0.02  # +2% over 10 bars


class MomentumStrategy:
    name = "momentum"

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
        roc = last["roc"]
        uptrend = last["close"] > last["ema_trend"]
        if not in_position:
            if roc is not None and pd.notna(roc) and roc > ROC_BUY and uptrend:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=price - _D(self.s.strategy.stop_atr_mult) * atr,
                    take_profit=price + _D(self.s.strategy.take_profit_atr_mult) * atr,
                    rationale=f"roc={roc:.3f}>{ROC_BUY} & uptrend",
                )
            return snap, Signal(action="hold", symbol=symbol)
        if (roc is None or pd.isna(roc) or roc < 0) or not uptrend:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale="momentum rolled over")
        return snap, Signal(action="hold", symbol=symbol)
