"""The agent loop — Decide / Guard / Sign orchestrator.

Each cycle:
  1. fetch prices + candles, mark-to-market, update risk flags
  2. KEEPER: stop-loss / take-profit sells (risk-reducing; bypass the gate)
  3. if not halted/killed: STRATEGY -> GLM GATE -> GUARD -> SIGN for entries/exits
  4. re-check risk flags; if killed trips, flatten everything (no auto-resume)
Everything is logged to the hash-chained audit log.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Optional

import structlog

from trading_agent.brain.glm_overlay import GLMOverlay
from trading_agent.config import Settings
from trading_agent.data.market_data import PriceSource
from trading_agent.execution.base import Executor
from trading_agent.execution.paper import PaperExecutor
from trading_agent.keeper.stop_keeper import StopKeeper
from trading_agent.models import (
    DecisionSource,
    EventKind,
    Fill,
    MarketSnapshot,
    Observation,
    OrderIntent,
    Position,
    Side,
    Signal,
    utcnow,
)
from trading_agent.risk.guardrail import Guardrail
from trading_agent.state.audit_log import AuditLog
from trading_agent.state.portfolio import Portfolio
from trading_agent.strategy import build_strategy

log = structlog.get_logger()


def _json(obj) -> dict:
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else dict(obj)


class TradingAgent:
    def __init__(self, settings: Settings, price_source: PriceSource, audit_log: AuditLog,
                 executor: Optional[Executor] = None):
        self.s = settings
        self.prices_src = price_source
        self.log = audit_log
        self.portfolio = Portfolio.from_audit_log(settings, audit_log)
        self.strategy = build_strategy(settings)
        self.brain = GLMOverlay(settings)
        self.guard = Guardrail(settings)
        self.keeper = StopKeeper(settings)
        self.executor: Executor = executor or PaperExecutor(settings)

    # ------------------------------------------------------------------ cycle
    def cycle(self) -> dict:
        cycle_id = f"c-{uuid.uuid4().hex[:10]}"
        symbols = self.s.symbols

        try:
            prices = self.prices_src.prices(symbols)
        except Exception as exc:
            self.log.append(EventKind.ERROR, {"cycle_id": cycle_id, "error": repr(exc)}, cycle_id)
            log.error("price_fetch_failed", cycle_id=cycle_id, error=repr(exc))
            return {"cycle_id": cycle_id, "status": "price_error"}

        # strategy analyze per symbol -> snapshots + signals
        snapshots: dict[str, MarketSnapshot] = {}
        signals: dict[str, Signal] = {}
        for sym in symbols:
            try:
                df = self.prices_src.candles(sym, bars=100)
                in_pos = sym in self.portfolio.positions
                snap, sig = self.strategy.analyze(sym, df, in_position=in_pos)
                snapshots[sym] = snap
                signals[sym] = sig
            except Exception as exc:
                self.log.append(EventKind.ERROR, {"cycle_id": cycle_id, "symbol": sym, "error": repr(exc)}, cycle_id)
                log.warning("analyze_failed", symbol=sym, error=repr(exc))

        self.portfolio.update_risk_flags(prices)

        # build observation (the only thing the brain sees)
        observation = self._observation(prices, list(snapshots.values()))

        # 1) KEEPER — stops/TP (risk-reducing; bypass gate, still logged)
        keeper_intents = self.keeper.check(list(self.portfolio.positions.values()), prices, cycle_id)
        for ki in keeper_intents:
            self._submit(ki, prices.get(ki.symbol, Decimal("0")), cycle_id, log_guard=False)
            log.info("keeper_exit", symbol=ki.symbol, rationale=ki.rationale)
        self.portfolio.update_risk_flags(prices)

        # 2) strategy decisions (entries + trend-break exits)
        fills = 0
        for sym in symbols:
            sig = signals.get(sym)
            if sig is None or sig.action == "hold":
                continue
            self.log.append(EventKind.SIGNAL, {"cycle_id": cycle_id, **_json(sig)}, cycle_id)

            gate = self.brain.gate(observation, sig)
            self.log.append(EventKind.GATE, {"cycle_id": cycle_id, **_json(gate)},
                            cycle_id, llm_raw=self.brain.last_raw)

            verdict = self.guard.evaluate(sig, gate, snapshots[sym],
                                          self.portfolio.risk_context(prices), cycle_id)
            self.log.append(EventKind.GUARD_DECISION, {"cycle_id": cycle_id, **_json(verdict)}, cycle_id)

            if not verdict.approved or verdict.intent is None:
                log.info("guard_rejected", symbol=sym, reason=verdict.reason.value, msg=verdict.message)
                continue

            result = self._submit(verdict.intent, prices.get(sym, Decimal("0")), cycle_id, log_guard=True)
            if result is not None:
                fills += 1
            self.portfolio.update_risk_flags(prices)
            if self.portfolio.killed:
                break

        # 3) kill-switch flatten (no auto-resume)
        if self.portfolio.killed and self.portfolio.positions:
            self.log.append(EventKind.KILLSWITCH, {"cycle_id": cycle_id, "reason": "drawdown_kill"}, cycle_id)
            self._flatten_all(prices, cycle_id)

        self._log_breaches(cycle_id, prices)
        self.log.append(EventKind.OBSERVATION, {"cycle_id": cycle_id, **_json(observation)}, cycle_id)
        self.log.append(EventKind.HEARTBEAT, {"cycle_id": cycle_id, "equity": str(self.portfolio.equity(prices)),
                                              "positions": len(self.portfolio.positions)}, cycle_id)

        summary = {
            "cycle_id": cycle_id,
            "status": "killed" if self.portfolio.killed else ("halted" if self.portfolio.halted else "ok"),
            "equity": str(self.portfolio.equity(prices)),
            "drawdown_pct": str(self.portfolio.drawdown_pct(prices)),
            "positions": len(self.portfolio.positions),
            "fills": fills,
            "keeper_exits": len(keeper_intents),
        }
        log.info("cycle_done", **summary)
        return summary

    def keeper_tick(self) -> dict:
        """Lightweight safety poll: prices + risk flags + stop/TP only (no new entries).
        Runs between bar closes so stops fire even if no decision cycle is due."""
        cycle_id = f"k-{uuid.uuid4().hex[:10]}"
        try:
            prices = self.prices_src.prices(self.s.symbols)
        except Exception as exc:
            self.log.append(EventKind.ERROR, {"cycle_id": cycle_id, "error": repr(exc)}, cycle_id)
            return {"cycle_id": cycle_id, "status": "price_error"}

        self.portfolio.update_risk_flags(prices)
        keeper_intents = self.keeper.check(list(self.portfolio.positions.values()), prices, cycle_id)
        for ki in keeper_intents:
            self._submit(ki, prices.get(ki.symbol, Decimal("0")), cycle_id, log_guard=False)
            log.info("keeper_exit", symbol=ki.symbol, rationale=ki.rationale)
        self.portfolio.update_risk_flags(prices)

        if self.portfolio.killed and self.portfolio.positions:
            self.log.append(EventKind.KILLSWITCH, {"cycle_id": cycle_id, "reason": "drawdown_kill"}, cycle_id)
            self._flatten_all(prices, cycle_id)

        self._log_breaches(cycle_id, prices)
        self.log.append(EventKind.HEARTBEAT,
                        {"cycle_id": cycle_id, "tick": "keeper",
                         "equity": str(self.portfolio.equity(prices)),
                         "positions": len(self.portfolio.positions)}, cycle_id)
        return {"cycle_id": cycle_id, "keeper_exits": len(keeper_intents),
                "equity": str(self.portfolio.equity(prices))}

    # ------------------------------------------------------------------ helpers
    def _observation(self, prices: dict[str, Decimal], markets: list[MarketSnapshot]) -> Observation:
        day_start = self.portfolio.day_start_equity or Decimal("1")
        cap_usd = day_start * Decimal(str(self.s.risk.daily_loss_cap_pct)) / Decimal("100")
        daily_pnl = self.portfolio.daily_pnl(prices)
        risk_budget = max(Decimal("0"), cap_usd + daily_pnl)  # shrinks as losses mount
        return Observation(
            ts=utcnow(),
            equity=self.portfolio.equity(prices),
            free_capital=self.portfolio.cash,
            risk_budget_remaining_today=risk_budget,
            daily_pnl=daily_pnl,
            drawdown_pct=self.portfolio.drawdown_pct(prices),
            positions=list(self.portfolio.positions.values()),
            markets=markets,
            recent_trades=self.portfolio.recent_trades,
            halted=self.portfolio.halted,
            killed=self.portfolio.killed,
            trades_today=self.portfolio.trades_today,
        )

    def _submit(self, intent: OrderIntent, ref_price: Decimal, cycle_id: str,
                log_guard: bool) -> Optional[tuple[Fill, Optional[object]]]:
        if ref_price <= 0:
            return None
        # resolve full-qty sells
        if intent.side == Side.SELL and intent.qty == 0:
            pos = self.portfolio.positions.get(intent.symbol)
            if pos is None:
                return None
            intent = intent.model_copy(update={"qty": pos.qty})

        fill = self.executor.submit(intent, ref_price)
        self.log.append(EventKind.ORDER_REQUEST, {"cycle_id": cycle_id, **_json(intent)}, cycle_id)
        closed = self.portfolio.apply_fill(
            fill,
            stop_price=intent.stop_price if intent.side == Side.BUY else None,
            take_profit=intent.take_profit if intent.side == Side.BUY else None,
        )
        self.log.append(EventKind.FILL, {"cycle_id": cycle_id, **_json(fill)}, cycle_id)
        return fill, closed

    def _flatten_all(self, prices: dict[str, Decimal], cycle_id: str) -> None:
        for sym in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[sym]
            intent = OrderIntent(
                symbol=sym, side=Side.SELL, qty=pos.qty, stop_price=Decimal("0"),
                source=DecisionSource.KILLSWITCH, rationale="kill-switch flatten", cycle_id=cycle_id,
            )
            self._submit(intent, prices.get(sym, Decimal("0")), cycle_id, log_guard=False)

    def _log_breaches(self, cycle_id: str, prices: dict[str, Decimal]) -> None:
        if self.portfolio.halted:
            self.log.append(EventKind.RISK_BREACH,
                            {"cycle_id": cycle_id, "kind": "daily_loss_cap",
                             "daily_pnl": str(self.portfolio.daily_pnl(prices))}, cycle_id)
        if self.portfolio.killed:
            self.log.append(EventKind.RISK_BREACH,
                            {"cycle_id": cycle_id, "kind": "drawdown_kill",
                             "drawdown_pct": str(self.portfolio.drawdown_pct(prices))}, cycle_id)
