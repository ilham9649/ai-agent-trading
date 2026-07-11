"""Multi-indicator (multi-technical) ensemble: 8 indicators vote, long on consensus.

No single indicator is robust, so combine: long/medium/short trend, golden cross,
MACD, ROC, RSI, and a volatility-regime filter each cast +/-1; go long when the
score clears a threshold, flat otherwise. Full-exposure (the SMA/score exit is the
risk control). Designed to time better than a single SMA -> a more robust edge.
"""
from __future__ import annotations

import math

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import Position, Side, Signal
from trading_agent.strategy.base import _D, compute_indicators, snapshot_from_row


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class MultiTA:
    def __init__(self, settings: Settings, threshold: int = 5):
        self.s = settings
        self.threshold = threshold
        self.need = 205
        self.name = f"multi_ta_t{threshold}"

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool,
                position: Position | None = None):
        if len(df) < self.need:
            snap = snapshot_from_row(symbol, df.iloc[-1].to_dict() if len(df) else pd.Series({"close": 0}))
            return snap, Signal(action="hold", symbol=symbol, rationale="warmup")

        close = df["close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        roc = close.pct_change(10)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, pd.NA)
        rsi = (100 - 100 / (1 + gain / loss)).fillna(50)
        realvol = close.pct_change(1).rolling(20).std()
        vol_med = realvol.rolling(100, min_periods=20).median()

        ind = compute_indicators(df, self.s)
        snap = snapshot_from_row(symbol, ind.iloc[-1])
        price, atr = snap.price, snap.atr
        i = len(df) - 1

        def vote(a, b):
            av, bv = a.iloc[i], (b.iloc[i] if hasattr(b, "iloc") else b)
            if _nan(av) or _nan(bv):
                return 0
            return 1 if av > bv else -1

        score = (
            vote(close, sma200) + vote(close, sma50) + vote(sma50, sma200)
            + vote(close, ema20) + vote(macd, macd_sig) + vote(roc, 0) + vote(rsi, 50)
        )
        rv, vm = realvol.iloc[i], vol_med.iloc[i]
        if not (_nan(rv) or _nan(vm)):
            score += 1 if rv < vm else -1

        bull = score >= self.threshold
        if not in_position:
            if bull:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=price - _D("10") * atr, take_profit=price + _D("100") * atr,
                    rationale=f"multi-TA score {score}/8 >= {self.threshold}",
                )
            return snap, Signal(action="hold", symbol=symbol)
        if not bull:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale=f"score {score} < {self.threshold}")
        return snap, Signal(action="hold", symbol=symbol)
