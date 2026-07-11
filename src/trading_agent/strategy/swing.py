"""Deterministic swing strategy (4H Donchian breakout + EMA trend + ATR stops).

This is the CORE the GLM overlays. It outputs a *proposal* Signal; the Guard
recomputes/validates stops and sizes from risk — the strategy's stop/size are
advisory only.
"""
from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import Action, MarketSnapshot, Side, Signal, Trend, utcnow

_D = lambda x: Decimal(str(x))
_NAN = float("nan")


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class SwingStrategy:
    def __init__(self, settings: Settings):
        self.s = settings
        self._need = max(
            self.s.strategy.donchian_period,
            self.s.strategy.ema_slow,
            self.s.strategy.atr_period,
        ) + 2

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.s.strategy.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.s.strategy.ema_slow, adjust=False).mean()
        df["donchian_high"] = df["high"].shift(1).rolling(self.s.strategy.donchian_period).max()
        df["donchian_low"] = df["low"].shift(1).rolling(self.s.strategy.donchian_period).min()
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(self.s.strategy.atr_period).mean()
        return df

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool) -> tuple[MarketSnapshot, Signal]:
        price = _D(str(float(df["close"].iloc[-1]))) if len(df) else Decimal("0")

        if len(df) < self._need:
            snap = MarketSnapshot(
                symbol=symbol, price=price, atr=Decimal("0"), ema_fast=price, ema_slow=price,
                donchian_high=price, donchian_low=price, trend="range", ts=utcnow(),
            )
            return snap, Signal(action="hold", symbol=symbol, rationale="insufficient history")

        d = self._indicators(df)
        last = d.iloc[-1]
        prev = d.iloc[-2]

        atr_f = last["atr"]
        atr = _D(str(float(atr_f))) if _finite(atr_f) else price * _D("0.01")
        ema_fast = _D(str(float(last["ema_fast"]))) if _finite(last["ema_fast"]) else price
        ema_slow = _D(str(float(last["ema_slow"]))) if _finite(last["ema_slow"]) else price
        dhigh = _D(str(float(last["donchian_high"]))) if _finite(last["donchian_high"]) else price * _D("1.01")
        dlow = _D(str(float(last["donchian_low"]))) if _finite(last["donchian_low"]) else price * _D("0.99")
        trend: Trend = "up" if ema_fast > ema_slow else ("down" if ema_fast < ema_slow else "range")

        snap = MarketSnapshot(
            symbol=symbol, price=price, atr=atr, ema_fast=ema_fast, ema_slow=ema_slow,
            donchian_high=dhigh, donchian_low=dlow, trend=trend, ts=utcnow(),
        )

        # ---- signal ----
        prev_dhigh = prev["donchian_high"]
        broke_out = _finite(prev_dhigh) and float(last["close"]) > float(prev_dhigh)
        uptrend = ema_fast > ema_slow
        vol_ok = True
        if d["volume"].mean() > 0:
            vol_ok = float(last["volume"]) > d["volume"].mean() * self.s.strategy.volume_filter_mult

        stop = price - _D(self.s.strategy.stop_atr_mult) * atr
        tp = price + _D(self.s.strategy.take_profit_atr_mult) * atr

        if not in_position:
            if broke_out and uptrend and vol_ok:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=stop, take_profit=tp,
                    rationale="breakout above donchian high with uptrend",
                )
            return snap, Signal(action="hold", symbol=symbol)

        # in position -> manage exit. (Take-profit is enforced by the keeper;
        # the strategy exits on trend break.)
        trend_break = price < ema_slow
        if trend_break:
            return snap, Signal(
                action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                rationale="trend break (close < EMA slow)",
            )
        return snap, Signal(action="hold", symbol=symbol)
