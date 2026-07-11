"""Portfolio state — cash, positions, equity, PnL, drawdown, daily caps.

In-memory authoritative state for the running session; every Fill is also written
to the AuditLog, and the whole portfolio can be rebuilt from the log via
``Portfolio.from_audit_log`` (crash recovery / verification).
"""
from __future__ import annotations

from datetime import datetime, date, timezone
from decimal import Decimal
from typing import Optional

from trading_agent.config import Settings
from trading_agent.models import (
    ClosedTrade,
    Fill,
    Position,
)
from trading_agent.risk.guardrail import RiskContext
from trading_agent.state.audit_log import AuditLog
from trading_agent.models import EventKind, Side, utcnow

_D = lambda x: Decimal(str(x))


class Portfolio:
    def __init__(self, settings: Settings):
        self.s = settings
        self.cash = _D(settings.starting_equity_usd)  # quote (USDC)
        self.positions: dict[str, Position] = {}
        self.realized_pnl_total = Decimal("0")
        self.high_water_mark = _D(settings.starting_equity_usd)
        self._day: Optional[date] = None
        self.day_start_equity = _D(settings.starting_equity_usd)
        self.trades_today = 0
        self.halted = False
        self.killed = False
        self.recent_trades: list[ClosedTrade] = []
        self._roll_day(utcnow())

    # ---- marking ----------------------------------------------------------
    def position_value(self, prices: dict[str, Decimal]) -> Decimal:
        return sum((p.qty * prices.get(p.symbol, p.entry_price) for p in self.positions.values()), Decimal("0"))

    def equity(self, prices: dict[str, Decimal]) -> Decimal:
        return self.cash + self.position_value(prices)

    def unrealized_pnl(self, prices: dict[str, Decimal]) -> Decimal:
        return sum(
            (p.qty * (prices.get(p.symbol, p.entry_price) - p.entry_price) for p in self.positions.values()),
            Decimal("0"),
        )

    def daily_pnl(self, prices: dict[str, Decimal]) -> Decimal:
        return self.equity(prices) - self.day_start_equity

    def drawdown_pct(self, prices: dict[str, Decimal]) -> Decimal:
        eq = self.equity(prices)
        if self.high_water_mark <= 0:
            return Decimal("0")
        dd = (self.high_water_mark - eq) / self.high_water_mark * Decimal("100")
        return dd if dd > 0 else Decimal("0")

    # ---- daily roll + risk flags -----------------------------------------
    def _roll_day(self, now: datetime) -> None:
        today = now.astimezone(timezone.utc).date()
        if self._day != today:
            self._day = today
            self.day_start_equity = self.equity({}) if self.positions == {} else self.day_start_equity
            # On a fresh roll, re-anchor day_start to current realized equity (cash + cost basis)
            self.day_start_equity = self.cash + sum(
                (p.qty * p.entry_price for p in self.positions.values()), Decimal("0")
            )
            self.trades_today = 0
            self.halted = False  # new day clears the daily halt (killed stays until manual re-arm)

    def update_risk_flags(self, prices: dict[str, Decimal], now: Optional[datetime] = None) -> None:
        """Set halted/killed from current PnL. Called every cycle after marking."""
        self._roll_day(now or utcnow())
        eq = self.equity(prices)
        if eq > self.high_water_mark:
            self.high_water_mark = eq

        daily = self.daily_pnl(prices)
        day_start = self.day_start_equity if self.day_start_equity > 0 else Decimal("1")
        daily_pct = -daily / day_start * Decimal("100")
        if daily_pct >= _D(self.s.risk.daily_loss_cap_pct):
            self.halted = True

        if self.drawdown_pct(prices) >= _D(self.s.risk.drawdown_kill_pct):
            self.killed = True

    def manual_rearm(self) -> None:
        self.killed = False

    # ---- fills ------------------------------------------------------------
    def apply_fill(
        self,
        fill: Fill,
        stop_price: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Optional[ClosedTrade]:
        """Apply a Fill. For BUY opens/adds a long; for SELL closes (full or partial).
        Returns a ClosedTrade if a position was closed."""
        closed: Optional[ClosedTrade] = None
        if fill.side == Side.BUY:
            cost = fill.qty * fill.price + fill.fee
            self.cash -= cost
            if fill.symbol in self.positions:
                p = self.positions[fill.symbol]
                blended = (p.entry_price * p.qty + fill.price * fill.qty) / (p.qty + fill.qty)
                p.entry_price = blended
                p.qty = p.qty + fill.qty
                if stop_price:
                    p.stop_price = min(p.stop_price, stop_price)
            else:
                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    side=Side.BUY,
                    qty=fill.qty,
                    entry_price=fill.price,
                    stop_price=stop_price or Decimal("0"),
                    take_profit=take_profit,
                    entry_ts=fill.ts,
                )
            self.trades_today += 1
        else:  # SELL -> close
            if fill.symbol not in self.positions:
                return None
            p = self.positions[fill.symbol]
            close_qty = fill.qty if fill.qty > 0 else p.qty
            close_qty = min(close_qty, p.qty)
            proceeds = close_qty * fill.price - fill.fee
            self.cash += proceeds
            pnl = close_qty * (fill.price - p.entry_price) - fill.fee
            self.realized_pnl_total += pnl
            closed = ClosedTrade(
                symbol=fill.symbol,
                side=Side.BUY,
                qty=close_qty,
                entry_price=p.entry_price,
                exit_price=fill.price,
                pnl=pnl,
                fee=fill.fee,
                opened_ts=p.entry_ts,
                closed_ts=fill.ts,
            )
            self.recent_trades.insert(0, closed)
            self.recent_trades = self.recent_trades[:10]
            p.qty = p.qty - close_qty
            if p.qty <= 0:
                del self.positions[fill.symbol]
        return closed

    # ---- risk context for the guard --------------------------------------
    def risk_context(self, prices: dict[str, Decimal]) -> RiskContext:
        return RiskContext(
            equity=self.equity(prices),
            free_capital=self.cash,
            positions=list(self.positions.values()),
            daily_pnl=self.daily_pnl(prices),
            drawdown_pct=self.drawdown_pct(prices),
            trades_today=self.trades_today,
            halted=self.halted,
            killed=self.killed,
        )

    # ---- replay from audit log -------------------------------------------
    @classmethod
    def from_audit_log(cls, settings: Settings, log: AuditLog) -> "Portfolio":
        pf = cls(settings)
        for ev in log.iter_events():
            if ev.kind != EventKind.FILL:
                continue
            payload = ev.payload
            fill = Fill(
                symbol=payload["symbol"],
                side=Side(payload["side"]),
                qty=Decimal(payload["qty"]),
                price=Decimal(payload["price"]),
                fee=Decimal(payload["fee"]),
                ts=_parse_ts(payload["ts"]),
                cycle_id=payload.get("cycle_id", ""),
                source=payload.get("source", "strategy"),
                order_id=payload.get("order_id", ""),
            )
            stop = Decimal(payload["stop_price"]) if payload.get("stop_price") else None
            tp = Decimal(payload["take_profit"]) if payload.get("take_profit") else None
            pf.apply_fill(fill, stop_price=stop, take_profit=tp)
        return pf


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)
