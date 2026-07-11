"""ForecastStrategy — a walk-forward ML strategy.

Each bar it (re)trains a gradient-boosted classifier on PAST bars (labels = sign of
forward return) and trades the forecast: go long when P(up) >= threshold, go flat
when P(up) < 0.5. The model retrains every ``retrain_every`` bars, training only on
rows whose forward-return label is known — strictly no look-ahead.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from trading_agent.config import Settings
from trading_agent.models import MarketSnapshot, Position, Side, Signal
from trading_agent.ml.features import FEATURE_COLS, make_features
from trading_agent.strategy.base import _D, compute_indicators, snapshot_from_row


class ForecastStrategy:
    name = "forecast_gbdt"

    def __init__(self, settings: Settings, horizon: int = 5, retrain_every: int = 60,
                 prob_threshold: float = 0.55, min_train: int = 250):
        self.s = settings
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.prob_threshold = prob_threshold
        self.min_train = min_train
        self.need = 70
        self.model = None
        self._trained_at = -1
        self.last_prob = 0.5

    def _train(self, feat: pd.DataFrame) -> None:
        # train only on rows whose forward label is known (exclude last `horizon`)
        train = feat.iloc[: len(feat) - self.horizon].dropna(subset=FEATURE_COLS + ["target"])
        train = train[train["target"].notna()]
        if len(train) < self.min_train or train["target"].nunique() < 2:
            self.model = None
            return
        X = train[FEATURE_COLS].fillna(0).to_numpy(dtype=float)
        y = train["target"].astype(int).to_numpy()
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=3,
            l2_regularization=0.1, random_state=0,
        )
        m.fit(X, y)
        self.model = m

    def analyze(self, symbol: str, df: pd.DataFrame, in_position: bool,
                position: Position | None = None) -> tuple[MarketSnapshot, Signal]:
        if len(df) < self.need:
            return snapshot_from_row(symbol, df.iloc[-1].to_dict() if len(df) else pd.Series({"close": 0})), \
                Signal(action="hold", symbol=symbol, rationale="warmup")

        ind = compute_indicators(df, self.s)
        snap = snapshot_from_row(symbol, ind.iloc[-1])
        feat = make_features(df, self.horizon)
        cur_idx = len(feat) - 1

        if self.model is None or (cur_idx - self._trained_at) >= self.retrain_every:
            self._train(feat)
            self._trained_at = cur_idx

        if self.model is None:
            return snap, Signal(action="hold", symbol=symbol, rationale="model not trained")

        x = feat.iloc[-1][FEATURE_COLS].fillna(0).to_numpy(dtype=float).reshape(1, -1)
        prob_up = float(self.model.predict_proba(x)[0, 1])
        self.last_prob = prob_up

        price, atr = snap.price, snap.atr
        if not in_position:
            if prob_up >= self.prob_threshold:
                return snap, Signal(
                    action="enter", symbol=symbol, side=Side.BUY, entry_price=price,
                    stop_price=price - _D(self.s.strategy.stop_atr_mult) * atr,
                    take_profit=price + _D(self.s.strategy.take_profit_atr_mult) * atr,
                    rationale=f"forecast P(up)={prob_up:.2f}>={self.prob_threshold}",
                )
            return snap, Signal(action="hold", symbol=symbol)
        if prob_up < 0.50:
            return snap, Signal(action="exit", symbol=symbol, side=Side.SELL, entry_price=price,
                                rationale=f"forecast P(up)={prob_up:.2f}<0.5")
        return snap, Signal(action="hold", symbol=symbol)
