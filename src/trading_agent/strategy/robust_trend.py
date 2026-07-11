"""Robust multi-asset trend-following: Donchian breakout + SMA trend filter.

Designed to generalise across assets WITHOUT per-asset parameter tuning:

  Entry  : close breaks above N-bar Donchian HIGH (using previous bars only)
           AND  close > SMA(trend_sma)  — long-term uptrend confirmed
  Exit   : close falls below M-bar Donchian LOW (M < N — asymmetric)
           OR   close < SMA(trend_sma)  — trend has broken
  Stop   : hard stop at entry − atr_stop × ATR  (recomputed by Guard)
  TP     : 6 × ATR wide target (let trends run, Guard enforces)
  Sizing : vol_target (position scales with 1/realized_vol) — configured in Risk

Anti-overfitting principles:
  - Identical parameters for every asset in the universe
  - Donchian channels are parameter-robust (20 ≈ 40 ≈ 55 all behave similarly)
  - SMA(100) trend filter is the most commonly cited regime filter
  - Minimal rules: 2 entry conditions, 2 exit conditions

Typical holding period: 2-8 weeks on daily bars — swing/trend following.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models import Position, Side, Signal, utcnow, MarketSnapshot
from trading_agent.strategy.base import _D, _finite, compute_indicators, snapshot_from_row


class RobustTrend:
    def __init__(
        self,
        settings: Settings,
        entry_period: int = 20,   # N-bar Donchian high for entry
        exit_period: int = 10,    # M-bar Donchian low for exit  (M < N)
        trend_sma: int = 100,     # long-term trend filter
        atr_stop: float = 2.5,    # hard stop distance in ATRs
    ):
        self.s = settings
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.trend_sma = trend_sma
        self.atr_stop = atr_stop
        self.need = trend_sma + entry_period + 5
        self.name = f"robust_trend_d{entry_period}x{exit_period}_sma{trend_sma}"

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared indicators via base + strategy-specific Donchian windows."""
        d = compute_indicators(df, self.s)
        # entry_period / exit_period are independent of settings.donchian_period
        d["don_entry"] = d["high"].shift(1).rolling(self.entry_period).max()
        d["don_exit"]  = d["low"].shift(1).rolling(self.exit_period).min()
        d["sma_trend"] = d["close"].rolling(self.trend_sma).mean()
        return d

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        in_position: bool,
        position: Position | None = None,  # unused; satisfies Strategy Protocol
    ) -> tuple[MarketSnapshot, Signal]:
        if len(df) < self.need:
            price = _D(str(float(df["close"].iloc[-1]))) if len(df) else Decimal("0")
            snap = MarketSnapshot(
                symbol=symbol, price=price, atr=price * _D("0.01"),
                ema_fast=price, ema_slow=price,
                donchian_high=price * _D("1.01"), donchian_low=price * _D("0.99"),
                trend="range", ts=utcnow(),
            )
            return snap, Signal(action="hold", symbol=symbol, rationale="warmup")

        ind = self._indicators(df)
        last = ind.iloc[-1]
        snap = snapshot_from_row(symbol, last)

        close = float(last["close"])

        # ── Trend regime filter ──────────────────────────────────
        sma_trend = last["sma_trend"]
        if not _finite(sma_trend):
            return snap, Signal(action="hold", symbol=symbol, rationale="warmup-sma")
        in_uptrend = close > float(sma_trend)

        # ── Donchian breakout levels ─────────────────────────────
        don_entry = last["don_entry"]   # high of prev entry_period bars
        don_exit  = last["don_exit"]    # low  of prev exit_period  bars

        if not in_position:
            if in_uptrend and _finite(don_entry) and close > float(don_entry):
                return snap, Signal(
                    action="enter",
                    symbol=symbol,
                    side=Side.BUY,
                    entry_price=snap.price,
                    stop_price=snap.price - _D(str(self.atr_stop)) * snap.atr,
                    take_profit=snap.price + _D("6") * snap.atr,
                    rationale=(
                        f"breakout>{float(don_entry):.4f} "
                        f"& close>{self.trend_sma}-SMA"
                    ),
                )
            return snap, Signal(action="hold", symbol=symbol)

        # ── Exit conditions (in position) ────────────────────────
        if not in_uptrend:
            return snap, Signal(
                action="exit", symbol=symbol, side=Side.SELL, entry_price=snap.price,
                rationale=f"trend break: close<{self.trend_sma}-SMA",
            )
        if _finite(don_exit) and close < float(don_exit):
            return snap, Signal(
                action="exit", symbol=symbol, side=Side.SELL, entry_price=snap.price,
                rationale=f"donchian exit: close<{self.exit_period}-bar low",
            )
        return snap, Signal(action="hold", symbol=symbol)
