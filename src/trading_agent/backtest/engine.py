"""Backtest engine — replays historical OHLCV through the SAME components used live
(strategy + guard + paper executor + keeper + portfolio), so backtest logic ≈ live logic.

Backtests the DETERMINISTIC SHELL only. The GLM overlay is deliberately excluded —
LLMs cannot be reliably backtested (they 'cannot forget the future', are
nondeterministic, and weren't there historically). The overlay is forward-tested
via paper trading against the deterministic baseline.

No-lookahead: at bar t the signal is computed from bars[0..t-1] and fills happen at
bar t's close. Stops/TP are checked intrabar against bar.low/bar.high (pessimistic:
if both trigger in one bar, the stop is assumed first).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from trading_agent.config import Settings
from trading_agent.execution.paper import PaperExecutor
from trading_agent.models import ClosedTrade, DecisionSource, Fill, GateDecision, OrderIntent, Side
from trading_agent.risk.guardrail import Guardrail
from trading_agent.state.portfolio import Portfolio
from trading_agent.strategy.swing import SwingStrategy

_D = lambda x: Decimal(str(x))


# ----------------------------- normal dist helpers (no scipy) -----------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return x


def deflated_sharpe(per_obs_sharpe: float, n_obs: int, n_trials: int,
                    skew: float, kurt: float) -> tuple[float, float]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio -> (DSR probability, SR0).
    per_obs_sharpe is the NON-annualized Sharpe of per-bar returns."""
    gamma = 0.5772156649015329
    if n_trials > 1:
        sr0 = ((1 - gamma) * _norm_ppf(1 - 1 / n_trials) +
               gamma * _norm_ppf(1 - 1 / (n_trials * math.e))) / math.sqrt(max(1, n_obs))
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * per_obs_sharpe + (kurt - 1) / 4 * per_obs_sharpe ** 2))
    dsr = _norm_cdf((per_obs_sharpe - sr0) * math.sqrt(max(1, n_obs - 1)) / denom)
    return dsr, sr0


# ----------------------------- result -----------------------------
class BacktestResult(BaseModel):
    start_equity: float
    end_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    win_rate: float
    profit_factor: float
    expectancy_usd: float
    num_trades: int
    total_fees: float
    deflated_sharpe: float
    n_trials: int
    bars: int
    sample: str  # "full" | "in-sample" | "out-of-sample"


# ----------------------------- engine -----------------------------
class Backtest:
    def __init__(self, settings: Settings, strategy=None):
        self.s = settings
        self.executor = PaperExecutor(settings)
        self.guard = Guardrail(settings)
        self.strategy = strategy or SwingStrategy(settings)
        self.bars_per_year = self._bars_per_year(settings.strategy.timeframe)
        self.trades: list[ClosedTrade] = []
        self.total_fees = Decimal("0")

    @staticmethod
    def _bars_per_year(tf: str) -> int:
        return {"1h": 8760, "4h": 2190, "1d": 365}.get(tf, 2190)

    @staticmethod
    def _bar_dt(series: dict, symbols: list, t: int):
        """Historical bar timestamp so day-rolls/daily caps work correctly in backtest.
        Returns None if no real timestamp (synthetic feeds) -> caller uses wall-clock."""
        try:
            raw = float(series[symbols[0]]["ts"].iloc[t])
            if raw > 1e11:  # ms since epoch
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc)
        except Exception:
            return None
        return None

    def run(self, series: dict[str, pd.DataFrame], n_trials: int = 1, sample: str = "full") -> BacktestResult:
        symbols = list(series.keys())
        n = min(len(series[s]) for s in symbols) if symbols else 0
        pf = Portfolio(self.s)
        need = getattr(self.strategy, "need", None) or getattr(self.strategy, "_need", 55)
        equity_curve: list[float] = [float(self.s.starting_equity_usd)]

        for t in range(n):
            closes = {sym: _D(float(series[sym]["close"].iloc[t])) for sym in symbols}
            bar_dt = self._bar_dt(series, symbols, t)
            pf.update_risk_flags(closes, now=bar_dt)

            # 1) intrabar stop / TP (pessimistic: stop before TP in the same bar)
            for sym in list(pf.positions.keys()):
                pos = pf.positions[sym]
                bar = series[sym].iloc[t]
                lo, hi = _D(float(bar["low"])), _D(float(bar["high"]))
                hit_stop = pos.stop_price > 0 and lo <= pos.stop_price
                hit_tp = pos.take_profit is not None and hi >= pos.take_profit
                if hit_stop:
                    self._close(pf, sym, pos.stop_price, DecisionSource.KEEPER, "stop-loss", f"bt-{t}")
                elif hit_tp:
                    self._close(pf, sym, pos.take_profit, DecisionSource.KEEPER, "take-profit", f"bt-{t}")
            pf.update_risk_flags(closes, now=bar_dt)
            if pf.killed:
                for sym in list(pf.positions.keys()):
                    self._close(pf, sym, closes.get(sym, Decimal("0")), DecisionSource.KILLSWITCH, "kill-flatten", f"bt-{t}")
                equity_curve.append(float(pf.equity(closes)))
                break

            # 2) strategy: exits on held positions, then entries
            for sym in symbols:
                if sym in pf.positions:
                    window = series[sym].iloc[:t + 1]
                    if len(window) < need:
                        continue
                    _, sig = self.strategy.analyze(sym, window, in_position=True)
                    if sig.action == "exit":
                        self._close(pf, sym, closes[sym], DecisionSource.STRATEGY, sig.rationale, f"bt-{t}")
            for sym in symbols:
                if sym in pf.positions:
                    continue
                window = series[sym].iloc[:t + 1]
                if len(window) < need:
                    continue
                snap, sig = self.strategy.analyze(sym, window, in_position=False)
                if sig.action != "enter":
                    continue
                gate = GateDecision(approve=True, size_factor=1.0)  # deterministic shell; LLM forward-tested apart
                verdict = self.guard.evaluate(sig, gate, snap, pf.risk_context(closes), f"bt-{t}")
                if not verdict.approved or verdict.intent is None:
                    continue
                fill = self.executor.submit(verdict.intent, closes[sym])
                pf.apply_fill(fill, stop_price=verdict.intent.stop_price, take_profit=verdict.intent.take_profit)
                self.total_fees += fill.fee

            equity_curve.append(float(pf.equity(closes)))

        return self._metrics(equity_curve, n_trials, sample)

    # -- close helper --------------------------------------------------------
    def _close(self, pf: Portfolio, sym: str, ref_price: Decimal, source: DecisionSource,
               rationale: str, cycle_id: str) -> None:
        pos = pf.positions.get(sym)
        if pos is None or ref_price <= 0:
            return
        intent = OrderIntent(symbol=sym, side=Side.SELL, qty=pos.qty, stop_price=Decimal("0"),
                             source=source, rationale=rationale, cycle_id=cycle_id)
        fill = self.executor.submit(intent, ref_price)
        closed = pf.apply_fill(fill)
        self.total_fees += fill.fee
        if closed is not None:
            self.trades.append(closed)

    # -- metrics -------------------------------------------------------------
    def _metrics(self, equity_curve: list[float], n_trials: int, sample: str) -> BacktestResult:
        import numpy as np  # local import; numpy is a pandas dep

        eq = np.asarray(equity_curve, dtype=float)
        rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
        n_obs = len(rets)

        start, end = float(eq[0]), float(eq[-1])
        total_ret = (end / start - 1) * 100 if start > 0 else 0.0
        years = max(1, n_obs) / self.bars_per_year
        cagr = ((end / start) ** (1 / years) - 1) * 100 if (start > 0 and years > 0) else 0.0

        # max drawdown
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max
        max_dd = float(abs(dd.min()) * 100) if dd.size else 0.0

        mean = float(rets.mean()) if n_obs else 0.0
        std = float(rets.std(ddof=1)) if n_obs > 1 else 0.0
        per_obs_sharpe = mean / std if std > 0 else 0.0
        sharpe = per_obs_sharpe * math.sqrt(self.bars_per_year)
        downside = rets[rets < 0]
        dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = (mean / dstd * math.sqrt(self.bars_per_year)) if dstd > 0 else 0.0

        wins = [tr for tr in self.trades if tr.pnl > 0]
        losses = [tr for tr in self.trades if tr.pnl <= 0]
        gross_win = sum(float(tr.pnl) for tr in wins)
        gross_loss = -sum(float(tr.pnl) for tr in losses)
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        expectancy = (sum(float(tr.pnl) for tr in self.trades) / len(self.trades)) if self.trades else 0.0

        skew = float(pd.Series(rets).skew()) if n_obs > 2 else 0.0
        kurt = float(pd.Series(rets).kurt()) if n_obs > 2 else 0.0
        dsr, _sr0 = deflated_sharpe(per_obs_sharpe, n_obs, n_trials, skew, kurt)

        return BacktestResult(
            start_equity=start, end_equity=end, total_return_pct=total_ret, cagr_pct=cagr,
            max_drawdown_pct=max_dd, sharpe=sharpe, sortino=sortino, win_rate=win_rate,
            profit_factor=profit_factor, expectancy_usd=expectancy, num_trades=len(self.trades),
            total_fees=float(self.total_fees), deflated_sharpe=dsr, n_trials=n_trials,
            bars=n_obs, sample=sample,
        )
