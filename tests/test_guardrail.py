from datetime import datetime, timezone
from decimal import Decimal

from trading_agent.models import GateDecision, MarketSnapshot, Side, Signal
from trading_agent.risk.guardrail import (
    Guardrail,
    RiskContext,
    compute_stop_distance_pct,
)


def snap(price, atr, symbol="WETH-USDC"):
    return MarketSnapshot(
        symbol=symbol, price=Decimal(str(price)), atr=Decimal(str(atr)),
        ema_fast=Decimal(str(price)), ema_slow=Decimal(str(price)),
        donchian_high=Decimal(str(price)), donchian_low=Decimal(str(price)),
        trend="up", ts=datetime.now(timezone.utc),
    )


def ctx(equity=500, free=500, positions=None, daily=0, dd=0, trades=0, halted=False, killed=False):
    return RiskContext(
        equity=Decimal(str(equity)), free_capital=Decimal(str(free)),
        positions=positions or [], daily_pnl=Decimal(str(daily)),
        drawdown_pct=Decimal(str(dd)), trades_today=trades,
        halted=halted, killed=killed,
    )


def enter(symbol="WETH-USDC", price=3000):
    return Signal(action="enter", symbol=symbol, side=Side.BUY, entry_price=Decimal(str(price)))


def test_sizing_clamped_to_position_cap(settings):
    # atr=40 -> stop_pct=1.5*40/3000*100=2%; risk notional=500*1%/2%=250; cap=125 -> 125
    g = Guardrail(settings)
    v = g.evaluate(enter(), GateDecision(approve=True, size_factor=1.0), snap(3000, 40), ctx(), "t")
    assert v.approved, v.message
    assert abs(float(v.intent.qty) - 125 / 3000) < 1e-6          # qty = 125/3000
    assert abs(float(v.intent.stop_price) - 2940.0) < 1e-3        # stop = 3000*0.98


def test_reject_when_halted(settings):
    assert not Guardrail(settings).evaluate(enter(), GateDecision(approve=True), snap(3000, 40), ctx(halted=True), "t").approved


def test_reject_when_killed(settings):
    v = Guardrail(settings).evaluate(enter(), GateDecision(approve=True), snap(3000, 40), ctx(killed=True), "t")
    assert not v.approved and v.reason.value == "killed"


def test_reject_overlay_denied(settings):
    v = Guardrail(settings).evaluate(enter(), GateDecision(approve=False), snap(3000, 40), ctx(), "t")
    assert not v.approved and v.reason.value == "overlay_denied"


def test_reject_unknown_symbol(settings):
    g = Guardrail(settings)
    v = g.evaluate(enter("DOGE-USDC", 1), GateDecision(approve=True), snap(1, 0.05, "DOGE-USDC"), ctx(), "t")
    assert not v.approved and v.reason.value == "invalid_symbol"


def test_stop_distance_floor_and_atr(settings):
    assert float(compute_stop_distance_pct(Decimal("100"), Decimal("0.01"), settings)) == 1.0  # floor
    assert abs(float(compute_stop_distance_pct(Decimal("3000"), Decimal("40"), settings)) - 2.0) < 1e-6
