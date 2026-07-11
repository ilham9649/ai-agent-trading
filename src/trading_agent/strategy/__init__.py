from trading_agent.config import Settings
from trading_agent.strategy.ema_cross import EmaCrossStrategy
from trading_agent.strategy.mean_reversion import MeanReversionStrategy
from trading_agent.strategy.momentum import MomentumStrategy
from trading_agent.strategy.swing import SwingStrategy
from trading_agent.strategy.trend_long_flat import TrendLongFlat


def build_strategy(settings: Settings):
    """Select the strategy from settings.strategy.name (default: swing)."""
    name = getattr(settings.strategy, "name", "swing")
    if name == "trend_long_flat":
        return TrendLongFlat(settings)
    if name == "ema_cross":
        return EmaCrossStrategy(settings)
    if name == "mean_reversion":
        return MeanReversionStrategy(settings)
    if name == "momentum":
        return MomentumStrategy(settings)
    return SwingStrategy(settings)


__all__ = [
    "build_strategy",
    "SwingStrategy",
    "EmaCrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "TrendLongFlat",
]
