"""GLM regime/sentiment overlay — the "AI" in the agent, used as a GATE not an
order-issuer. Calls z.ai's OpenAI-compatible PAYG endpoint (glm-4.6 default).

Design constraints from the verified docs:
  * tool_choice only supports 'auto' (can't force a tool) -> we ask for JSON in
    the prompt and parse defensively.
  * disable deep thinking + low temperature for fast, cheap, deterministic-ish calls.
  * On any parse failure we FAIL OPEN (approve, size 1.0) because the Guard bounds
    risk in hard code; the LLM is an enhancer, not the safety layer. The raw
    response is always logged so failures are visible.
  * With no API key (paper mode) -> neutral stub gate.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from trading_agent.config import Settings
from trading_agent.models import GateDecision, Observation, Signal

SYSTEM_PROMPT = """You are the RISK-GATE overlay of a conservative swing-trading bot on Base (crypto).
A deterministic strategy has just proposed entering a LONG position. Your job is to
gate it: assess whether the market REGIME and any obvious RISK/SENTIMENT make this
entry unwise RIGHT NOW.

You do NOT set the order size or stop — hard code guardrails do that. You only decide
APPROVE/DENY and an optional SIZE_FACTOR (0.0-1.0 to scale risk down).

Deny when: choppy/range regime against a breakout, obvious macro risk event imminent,
or momentum looks exhausted. Approve when the regime supports the breakout.

Reply with ONLY a JSON object, nothing else:
{"approve": <bool>, "size_factor": <0.0-1.0>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}"""


class GLMOverlay:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = None
        if settings.glm_api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=settings.glm_api_key, base_url=settings.glm.base_url)
            except Exception:
                self.client = None

    @property
    def last_raw(self) -> Optional[str]:
        return getattr(self, "_last_raw", None)

    def gate(self, observation: Observation, signal: Signal) -> GateDecision:
        # Only entries need gating; exits/holds pass through.
        if signal.action != "enter":
            return GateDecision(approve=True, size_factor=1.0, rationale="non-entry signal", stub=True)

        if self.client is None:
            return GateDecision(
                approve=True, size_factor=1.0, confidence=0.0,
                rationale="paper-mode stub: no GLM key, neutral gate",
                model="stub", stub=True,
            )

        user_msg = self._build_user_msg(observation, signal)
        try:
            resp = self.client.chat.completions.create(
                model=self.s.glm.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.s.glm.temperature,
                max_tokens=self.s.glm.max_tokens,
                # NOTE: do not enable 'thinking' here — we want fast, cheap decisions.
            )
            content = resp.choices[0].message.content or ""
            self._last_raw = content
            return self._parse(content)
        except Exception as exc:  # network / API error -> fail open, log
            self._last_raw = f"ERROR: {exc!r}"
            return GateDecision(
                approve=True, size_factor=1.0, confidence=0.0,
                rationale=f"GLM call failed ({exc!r}); failing open (guard bounds risk)",
                model=self.s.glm.model, stub=True,
            )

    def _build_user_msg(self, observation: Observation, signal: Signal) -> str:
        # Compact, structured context (the only thing the brain sees).
        pos = [{"sym": p.symbol, "entry": str(p.entry_price), "uPnL%": "n/a"} for p in observation.positions]
        mkt = {
            ms.symbol: {
                "price": str(ms.price), "atr%": f"{(ms.atr / ms.price * 100):.2f}" if ms.price else "0",
                "trend": ms.trend,
            }
            for ms in observation.markets
        }
        ctx = {
            "equity_usd": str(observation.equity),
            "daily_pnl_usd": str(observation.daily_pnl),
            "drawdown_pct": str(observation.drawdown_pct),
            "open_positions": pos,
            "markets": mkt,
            "proposed_entry": {
                "symbol": signal.symbol, "entry": str(signal.entry_price),
                "stop": str(signal.stop_price), "tp": str(signal.take_profit),
                "rationale": signal.rationale,
            },
        }
        return "Proposed entry (deterministic breakout strategy):\n" + json.dumps(ctx, indent=2)

    def _parse(self, content: str) -> GateDecision:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return GateDecision(
                    approve=True, size_factor=1.0, confidence=0.0,
                    rationale=f"unparseable GLM response; failing open", model=self.s.glm.model, stub=True,
                )
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return GateDecision(
                    approve=True, size_factor=1.0, confidence=0.0,
                    rationale="unparseable GLM JSON; failing open", model=self.s.glm.model, stub=True,
                )
        size = float(data.get("size_factor", 1.0))
        size = max(0.0, min(1.5, size))
        return GateDecision(
            approve=bool(data.get("approve", True)),
            size_factor=size,
            confidence=float(data.get("confidence", 0.0)),
            rationale=str(data.get("rationale", ""))[:300],
            model=self.s.glm.model,
            escalated=False,
            stub=False,
        )
