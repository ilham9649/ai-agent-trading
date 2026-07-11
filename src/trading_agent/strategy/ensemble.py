"""Ensemble: enter only when N of M sub-strategies agree (higher-conviction trades).
Reduces false signals at the cost of fewer trades — good when individual strategies
are noisy but point the same way on real setups."""
from __future__ import annotations

from decimal import Decimal
from typing import Sequence

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Side, Signal
from trading_agent.strategy.base import snapshot_from_row


class EnsembleStrategy:
    def __init__(self, settings: Settings, strategies: Sequence, min_agree: int = 2):
        self.s = settings
        self.strategies = list(strategies)
        self.min_agree = min_agree
        self.name = f"ensemble_{min_agree}of{len(self.strategies)}"
        self.need = max(getattr(st, "need", getattr(st, "_need", 55)) for st in self.strategies)

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool,
                position: Position | None = None) -> tuple[MarketSnapshot, Signal]:
        votes: list[str] = []
        stops: list[Decimal] = []
        snap = None
        for strat in self.strategies:
            snap, sig = strat.analyze(symbol, df, in_position)
            votes.append(sig.action)
            if sig.action == "enter" and sig.stop_price > 0:
                stops.append(sig.stop_price)
        if snap is None:
            snap = snapshot_from_row(symbol, df.iloc[-1].to_dict() if len(df) else pd.Series({"close": 0}))
        price, atr = snap.price, snap.atr

        if not in_position:
            n_enter = sum(1 for a in votes if a == "enter")
            if n_enter >= self.min_agree:
                # tightest of the agreeing stops (most conservative)
                stop = min(stops) if stops else price * Decimal("0.98")
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=stop, take_profit=price + Decimal(str(self.s.strategy.take_profit_atr_mult)) * atr,
                    rationale=f"{n_enter}/{len(self.strategies)} strategies agree",
                )
            return snap, Signal(action="hold", symbol=symbol)

        n_exit = sum(1 for a in votes if a == "exit")
        if n_exit >= self.min_agree:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale=f"{n_exit}/{len(self.strategies)} strategies say exit")
        return snap, Signal(action="hold", symbol=symbol)
