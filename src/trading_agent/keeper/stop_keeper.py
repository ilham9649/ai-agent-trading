"""Stop-loss / take-profit keeper.

On-chain DEXes have no native server-side stop orders, so a keeper watches prices
and emits market-sell intents when a position's stop (or take-profit) is breached.
These intents are risk-reducing: they bypass the GLM gate but still route through
the Guard (which always approves exits) and the Executor. In live mode an
OpenZeppelin Defender Sentinel runs as an independent fallback if this keeper
ever goes silent.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from trading_agent.config import Settings
from trading_agent.models import DecisionSource, OrderIntent, OrderType, Position, Side


class StopKeeper:
    def __init__(self, settings: Settings):
        self.s = settings

    def check(
        self,
        positions: list[Position],
        prices: dict[str, Decimal],
        cycle_id: str,
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for p in positions:
            px = prices.get(p.symbol)
            if px is None or p.qty <= 0:
                continue
            if px <= p.stop_price and p.stop_price > 0:
                intents.append(self._sell(p, "stop-loss hit", cycle_id))
            elif p.take_profit is not None and px >= p.take_profit:
                intents.append(self._sell(p, "take-profit hit", cycle_id))
        return intents

    @staticmethod
    def _sell(p: Position, rationale: str, cycle_id: str) -> OrderIntent:
        return OrderIntent(
            symbol=p.symbol,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            qty=p.qty,
            stop_price=Decimal("0"),
            source=DecisionSource.KEEPER,
            rationale=rationale,
            cycle_id=cycle_id,
        )
