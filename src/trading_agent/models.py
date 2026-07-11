"""Typed contracts every layer speaks. Pydantic v2; Decimal for all money/qty."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- enums
class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class DecisionSource(str, Enum):
    STRATEGY = "strategy"
    GLM_OVERLAY = "glm_overlay"
    KEEPER = "keeper"
    KILLSWITCH = "killswitch"


class GuardReason(str, Enum):
    APPROVED = "approved"
    HALTED_TODAY = "halted_today"
    KILLED = "killed"
    DAILY_LOSS_CAP = "daily_loss_cap"
    DRAWDOWN_KILL = "drawdown_kill"
    MAX_POSITIONS = "max_positions"
    MAX_DEPLOYED = "max_deployed"
    POSITION_TOO_LARGE = "position_too_large"
    INVALID_STOP = "invalid_stop"
    INVALID_SYMBOL = "invalid_symbol"
    INVALID_QTY = "invalid_qty"
    LEVERAGE_CAP = "leverage_cap"
    DUST = "dust"
    TRADE_COUNT_CAP = "trade_count_cap"
    OVERLAY_DENIED = "overlay_denied"


class EventKind(str, Enum):
    OBSERVATION = "observation"
    SIGNAL = "signal"
    GATE = "gate"
    GUARD_DECISION = "guard_decision"
    ORDER_REQUEST = "order_request"
    FILL = "fill"
    STOP_EVENT = "stop_event"
    RISK_BREACH = "risk_breach"
    KILLSWITCH = "killswitch"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


Trend = Literal["up", "down", "range"]
Action = Literal["enter", "exit", "hold"]


# --------------------------------------------------------------------------- market
class MarketSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    price: Decimal
    atr: Decimal
    ema_fast: Decimal
    ema_slow: Decimal
    donchian_high: Decimal
    donchian_low: Decimal
    trend: Trend
    ts: datetime


# --------------------------------------------------------------------------- strategy output
class Signal(BaseModel):
    """Output of the deterministic strategy — a *proposal*, not an order."""
    action: Action
    symbol: str
    side: Optional[Side] = None
    entry_price: Decimal = Decimal("0")
    stop_price: Decimal = Decimal("0")
    take_profit: Decimal = Decimal("0")
    rationale: str = ""


# --------------------------------------------------------------------------- GLM gate output
class GateDecision(BaseModel):
    """Output of the GLM regime/sentiment overlay."""
    approve: bool
    size_factor: float = Field(default=1.0, ge=0.0, le=1.5)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    model: str = ""
    escalated: bool = False
    stub: bool = False  # True when no API key (paper-mode neutral gate)


# --------------------------------------------------------------------------- orders / fills
class OrderIntent(BaseModel):
    """A guard-approved (or pending-guard) instruction to execute."""
    symbol: str
    side: Side
    order_type: OrderType = OrderType.MARKET
    qty: Decimal  # base-asset quantity
    stop_price: Decimal
    take_profit: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    leverage: int = 1
    source: DecisionSource
    rationale: str = ""
    cycle_id: str = ""


class Fill(BaseModel):
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    fee: Decimal
    ts: datetime
    cycle_id: str
    source: DecisionSource
    order_id: str


class Position(BaseModel):
    symbol: str
    side: Side
    qty: Decimal
    entry_price: Decimal
    stop_price: Decimal
    take_profit: Optional[Decimal] = None
    entry_ts: datetime
    realized_pnl: Decimal = Decimal("0")


class ClosedTrade(BaseModel):
    symbol: str
    side: Side
    qty: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    fee: Decimal
    opened_ts: datetime
    closed_ts: datetime


# --------------------------------------------------------------------------- guard
class GuardVerdict(BaseModel):
    approved: bool
    reason: GuardReason
    intent: Optional[OrderIntent] = None
    message: str = ""
    clamped_from_qty: Optional[Decimal] = None  # if size was reduced


# --------------------------------------------------------------------------- observation
class Observation(BaseModel):
    """The single compact blob the brain sees each cycle."""
    ts: datetime
    equity: Decimal
    free_capital: Decimal
    risk_budget_remaining_today: Decimal
    daily_pnl: Decimal
    drawdown_pct: Decimal
    positions: list[Position]
    markets: list[MarketSnapshot]
    recent_trades: list[ClosedTrade]
    halted: bool
    killed: bool
    trades_today: int


# --------------------------------------------------------------------------- audit log row
class AuditEvent(BaseModel):
    id: int
    ts: datetime
    cycle_id: str
    kind: EventKind
    payload: dict
    llm_raw: Optional[str] = None
    prev_hash: str
    hash: str
