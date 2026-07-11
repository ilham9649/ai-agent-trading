"""The RISK/GUARDRAIL layer — pure deterministic code.

This is the load-bearing safety wall. The LLM never calls this; the agent loop
runs every proposed action through here before any order is signed. The LLM can
propose a size; CODE clamps and may veto. Settings are read from immutable config
the model cannot touch at runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from trading_agent.config import Settings
from trading_agent.models import (
    DecisionSource,
    GuardReason,
    GuardVerdict,
    GateDecision,
    MarketSnapshot,
    OrderIntent,
    OrderType,
    Position,
    Side,
    Signal,
)

# Reject positions whose notional is below this (dust / not worth the gas & fee).
MIN_NOTIONAL_USD = Decimal("10")


def _D(x) -> Decimal:
    return Decimal(str(x))


def _reject(reason: GuardReason, message: str = "") -> GuardVerdict:
    return GuardVerdict(approved=False, reason=reason, message=message)


def _approve(intent: Optional[OrderIntent], message: str = "") -> GuardVerdict:
    return GuardVerdict(approved=True, reason=GuardReason.APPROVED, intent=intent, message=message)


@dataclass
class RiskContext:
    """Snapshot of portfolio state the guard reads. Built by the agent loop."""
    equity: Decimal
    free_capital: Decimal
    positions: list[Position]
    daily_pnl: Decimal
    drawdown_pct: Decimal
    trades_today: int
    halted: bool
    killed: bool


# --------------------------------------------------------------------------- pure math
def compute_stop_distance_pct(entry: Decimal, atr: Decimal, settings: Settings) -> Decimal:
    """stop distance as a % of entry: max(stop_atr_mult*ATR, min_stop_pct)."""
    if entry <= 0:
        return Decimal("0")
    atr_stop_pct = (_D(settings.strategy.stop_atr_mult) * atr / entry) * Decimal("100")
    return max(atr_stop_pct, _D(settings.strategy.min_stop_pct))


def compute_take_profit(entry: Decimal, atr: Decimal, settings: Settings) -> Decimal:
    return entry + _D(settings.strategy.take_profit_atr_mult) * atr


def risk_based_notional(equity: Decimal, stop_distance_pct: Decimal, settings: Settings) -> Decimal:
    """Notional implied by risking risk_pct of equity over the stop distance."""
    risk_amount = equity * _D(settings.risk.max_risk_per_trade_pct) / Decimal("100")
    return risk_amount / (stop_distance_pct / Decimal("100"))


# --------------------------------------------------------------------------- guard
class Guardrail:
    def __init__(self, settings: Settings):
        self.s = settings
        self.bars_per_year = {"1h": 8760, "4h": 2190, "1d": 365}.get(settings.strategy.timeframe, 365)

    def evaluate(
        self,
        signal: Signal,
        gate: GateDecision,
        market: MarketSnapshot,
        ctx: RiskContext,
        cycle_id: str,
    ) -> GuardVerdict:
        s = self.s

        if ctx.killed:
            return _reject(GuardReason.KILLED, "Account killed; manual re-arm required.")
        if signal.action == "enter" and not gate.approve:
            return _reject(GuardReason.OVERLAY_DENIED, f"GLM overlay denied: {gate.rationale}")

        # EXIT / close: allow even when halted (closing reduces risk).
        if signal.action == "exit":
            return self._exit_intent(signal, market, cycle_id)
        if signal.action != "enter":
            return _approve(None, "hold — no action")

        # 2) Entry gates
        if ctx.halted:
            return _reject(GuardReason.HALTED_TODAY, "Halted for the day (daily-loss cap).")
        if ctx.drawdown_pct >= _D(s.risk.drawdown_kill_pct):
            return _reject(GuardReason.DRAWDOWN_KILL,
                           f"Drawdown {ctx.drawdown_pct:.2f}% >= kill {s.risk.drawdown_kill_pct}%.")
        # NOTE: the daily-loss cap is enforced via the `halted` flag (set by the
        # portfolio/agent when realized+unrealized daily PnL hits -daily_loss_cap_pct).
        if signal.symbol not in s.symbols:
            return _reject(GuardReason.INVALID_SYMBOL, f"{signal.symbol} not in universe.")
        if any(p.symbol == signal.symbol for p in ctx.positions):
            return _reject(GuardReason.MAX_POSITIONS, f"Already in {signal.symbol} (one position per symbol).")
        if len(ctx.positions) >= s.risk.max_concurrent_positions:
            return _reject(GuardReason.MAX_POSITIONS, f"Max {s.risk.max_concurrent_positions} concurrent positions.")
        if ctx.trades_today >= s.risk.max_trades_per_day:
            return _reject(GuardReason.TRADE_COUNT_CAP, f"Max {s.risk.max_trades_per_day} trades/day reached.")

        entry = market.price
        if entry <= 0:
            return _reject(GuardReason.INVALID_QTY, "No valid market price.")

        # 3) Authoritative stop + take-profit (strategy's are advisory only)
        stop_pct = compute_stop_distance_pct(entry, market.atr, s)
        if stop_pct <= 0:
            return _reject(GuardReason.INVALID_STOP, "Could not compute a positive stop distance.")
        stop = (entry * (Decimal("100") - stop_pct) / Decimal("100")).quantize(Decimal("0.000001"))
        tp = compute_take_profit(entry, market.atr, s)

        # 4) Position size, then CLAMP to caps + available capital
        if s.risk.sizing_mode == "vol_target":
            per_bar_vol = (market.atr / market.price) if market.price > 0 else Decimal("0")
            ann_vol = float(per_bar_vol) * math.sqrt(self.bars_per_year)
            weight = (s.risk.target_annual_vol_pct / 100.0) / ann_vol if ann_vol > 0 else 0.0
            notional = ctx.equity * _D(min(1.0, weight))  # cap at 1x (no leverage)
        elif s.risk.sizing_mode == "fixed_fraction":
            notional = ctx.equity * _D(s.risk.fixed_fraction_pct) / Decimal("100")
        else:
            notional = risk_based_notional(ctx.equity, stop_pct, s)
        cap_notional = ctx.equity * _D(s.risk.max_position_notional_pct) / Decimal("100")
        max_deploy = ctx.equity * _D(s.risk.max_deployed_pct) / Decimal("100")
        deployed = self._deployed_notional(ctx)
        notional = min(notional, cap_notional, ctx.free_capital, max_deploy - deployed)

        if notional < MIN_NOTIONAL_USD:
            return _reject(GuardReason.DUST, f"Notional {notional:.2f} below floor {MIN_NOTIONAL_USD}.")

        qty = (notional / entry).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        if qty <= 0:
            return _reject(GuardReason.INVALID_QTY, "Computed qty <= 0.")

        intent = OrderIntent(
            symbol=signal.symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=qty,
            stop_price=stop,
            take_profit=tp,
            leverage=min(1, s.risk.leverage_cap),  # spot
            source=DecisionSource.STRATEGY,
            rationale=signal.rationale,
            cycle_id=cycle_id,
        )
        return _approve(intent, "approved")

    # -- exits ----------------------------------------------------------------
    def _exit_intent(self, signal: Signal, market: MarketSnapshot, cycle_id: str) -> GuardVerdict:
        if market.price <= 0:
            return _reject(GuardReason.INVALID_QTY, "No valid market price for exit.")
        intent = OrderIntent(
            symbol=signal.symbol,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("0"),  # executor fills full position qty on SELL
            stop_price=Decimal("0"),
            source=DecisionSource.STRATEGY,
            rationale=signal.rationale or "strategy exit",
            cycle_id=cycle_id,
        )
        return _approve(intent, "exit approved")

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _deployed_notional(ctx: RiskContext) -> Decimal:
        return sum((p.qty * p.entry_price for p in ctx.positions), Decimal("0"))
