"""Vectorized multi-indicator signal (shared by the leveraged sim scripts).

Returns a Series of 1.0/0.0: 1.0 when the PRIOR bar's 8-indicator score >= threshold
(no look-ahead — uses score.shift(1)). Same 8 indicators as MultiTA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def multi_ta_signal(df: pd.DataFrame, threshold: int = 4) -> pd.Series:
    close = df["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    roc = close.pct_change(10)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rsi = (100 - 100 / (1 + gain / loss)).fillna(50)
    realvol = close.pct_change(1).rolling(20).std()
    vol_med = realvol.rolling(100, min_periods=20).median()

    def b(a, c):
        return (a > c).astype(float) * 2 - 1

    score = (b(close, sma200) + b(close, sma50) + b(sma50, sma200) + b(close, ema20)
             + b(macd, macd_sig) + b(roc, 0.0) + b(rsi, 50.0)
             + (realvol < vol_med).astype(float) * 2 - 1)
    return (score.shift(1) >= threshold).astype(float)
