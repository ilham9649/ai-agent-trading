"""Executor interface. Paper simulates fills; Live (CowSwap/KMS) drops in later."""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from trading_agent.models import Fill, OrderIntent


class Executor(Protocol):
    def submit(self, intent: OrderIntent, ref_price: Decimal) -> Fill: ...
    def mode(self) -> str: ...
