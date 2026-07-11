"""Feature engineering for return-direction forecasting. All features are LAGGED
(past-only) so prediction has no look-ahead. The target (forward return sign) is
used ONLY as a historical training label — never for the row being predicted."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_20", "rsi", "ema_dist", "roc", "vol_ratio", "range_pct",
]


def make_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    ret1 = close.pct_change(1)
    d["ret_1"] = ret1
    d["ret_2"] = close.pct_change(2)
    d["ret_5"] = close.pct_change(5)
    d["ret_10"] = close.pct_change(10)
    d["ret_20"] = close.pct_change(20)
    d["vol_10"] = ret1.rolling(10).std()
    d["vol_20"] = ret1.rolling(20).std()
    d["range_pct"] = (d["high"] - d["low"]) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + gain / loss)).fillna(50)

    d["ema_dist"] = close / close.ewm(span=50, adjust=False).mean() - 1
    d["roc"] = close.pct_change(10)
    if "volume" in d:
        d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean().replace(0, np.nan)
    else:
        d["vol_ratio"] = 1.0

    d["fwd_ret"] = close.shift(-horizon) / close - 1
    d["target"] = (d["fwd_ret"] > 0).astype("Int64")
    return d
