"""Paper executor — simulates CowSwap-like fills against the current price.

Models taker slippage (wider on keeper/kill sells to guarantee exit) and a ~0.1%
AMM fee. Never touches the chain. The real CowSwap executor (gasless, structural
MEV protection, `cow-py`) replaces this in live mode behind the same interface.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

from trading_agent.config import Settings
from trading_agent.models import DecisionSource, Fill, OrderIntent, Side, utcnow

_D = lambda x: Decimal(str(x))
AMM_FEE_BPS = 10  # 0.1%


class PaperExecutor:
    def __init__(self, settings: Settings):
        self.s = settings

    def mode(self) -> str:
        return "paper"

    def submit(self, intent: OrderIntent, ref_price: Decimal) -> Fill:
        wide = intent.source in (DecisionSource.KEEPER, DecisionSource.KILLSWITCH)
        bps = _D(self.s.execution.stop_slippage_bps if wide else self.s.execution.slippage_bps)
        if intent.side == Side.BUY:
            price = ref_price * (Decimal("10000") + bps) / Decimal("10000")
        else:
            price = ref_price * (Decimal("10000") - bps) / Decimal("10000")
        qty = intent.qty
        fee = (qty * price * _D(AMM_FEE_BPS) / Decimal("10000")).quantize(Decimal("0.000001"))
        return Fill(
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            price=price.quantize(Decimal("0.00000001")),
            fee=fee,
            ts=utcnow(),
            cycle_id=intent.cycle_id,
            source=intent.source,
            order_id=f"paper-{uuid.uuid4().hex[:12]}",
        )
