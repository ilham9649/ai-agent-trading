from datetime import datetime, timezone
from decimal import Decimal

from trading_agent.models import DecisionSource, EventKind, Fill, Side
from trading_agent.state.audit_log import AuditLog
from trading_agent.state.portfolio import Portfolio


def _fill(side, qty, price, fee="0.3", symbol="WETH-USDC"):
    return Fill(symbol=symbol, side=side, qty=Decimal(str(qty)), price=Decimal(str(price)),
                fee=Decimal(fee), ts=datetime.now(timezone.utc), cycle_id="t",
                source=DecisionSource.STRATEGY, order_id="o1")


def test_buy_open_and_equity(settings):
    pf = Portfolio(settings)
    assert pf.cash == Decimal("500")
    assert pf.apply_fill(_fill(Side.BUY, "0.1", "3000", "0.3"), stop_price=Decimal("2900")) is None
    assert "WETH-USDC" in pf.positions
    assert abs(float(pf.cash) - (500 - 300 - 0.3)) < 1e-6               # 199.7
    assert abs(float(pf.equity({"WETH-USDC": Decimal("3000")})) - 499.7) < 1e-6


def test_sell_close_realizes_pnl(settings):
    pf = Portfolio(settings)
    pf.apply_fill(_fill(Side.BUY, "0.1", "3000", "0.3"), stop_price=Decimal("2900"))
    closed = pf.apply_fill(_fill(Side.SELL, "0.1", "3300", "0.3"))
    assert closed is not None
    assert abs(float(closed.pnl) - 29.7) < 1e-6                          # 0.1*(3300-3000)-0.3
    assert "WETH-USDC" not in pf.positions


def test_replay_from_log_matches(tmp_path, settings):
    log = AuditLog(tmp_path / "p.sqlite")
    pf = Portfolio(settings)
    f1 = _fill(Side.BUY, "0.1", "3000", "0.3")
    pf.apply_fill(f1, stop_price=Decimal("2900"))
    log.append(EventKind.FILL, {**f1.model_dump(mode="json"), "stop_price": "2900"}, "c1")
    f2 = _fill(Side.SELL, "0.1", "3300", "0.3")
    pf.apply_fill(f2)
    log.append(EventKind.FILL, {**f2.model_dump(mode="json")}, "c1")
    log.close()

    rebuilt = Portfolio.from_audit_log(settings, AuditLog(tmp_path / "p.sqlite"))
    assert abs(float(rebuilt.cash) - float(pf.cash)) < 1e-6
    assert abs(float(rebuilt.realized_pnl_total) - float(pf.realized_pnl_total)) < 1e-6
