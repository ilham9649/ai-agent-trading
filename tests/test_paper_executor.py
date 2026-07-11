from decimal import Decimal

from trading_agent.execution.paper import PaperExecutor
from trading_agent.models import DecisionSource, OrderIntent, Side


def _intent(side, qty, source=DecisionSource.STRATEGY):
    return OrderIntent(symbol="WETH-USDC", side=side, qty=Decimal(str(qty)),
                       stop_price=Decimal("0"), source=source, cycle_id="t")


def test_buy_slippage_and_fee(settings):
    ex = PaperExecutor(settings)
    fill = ex.submit(_intent(Side.BUY, 0.1), Decimal("3000"))
    assert abs(float(fill.price) - 3015.0) < 1e-3          # 3000*(1+50bps)
    assert abs(float(fill.fee) - 0.3015) < 1e-4             # 0.1% of 0.1*3015


def test_keeper_sell_uses_wide_slippage(settings):
    ex = PaperExecutor(settings)
    fill = ex.submit(_intent(Side.SELL, 0.1, DecisionSource.KEEPER), Decimal("3000"))
    assert abs(float(fill.price) - 2940.0) < 1e-3           # 3000*(1-200bps)
