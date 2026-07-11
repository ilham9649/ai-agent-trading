# trading-agent

An on-chain **L2 (Base) swing-trading agent** with a GLM overlay and **hard, deterministic code guardrails**. Paper-mode by default — it never touches real funds until you explicitly switch to live.

> ⚠️ **Real money, eventually.** The honest baseline: ~95% of retail day/algo traders lose money. This project is built to *survive* while you research an edge — bounded by code, paper-first, tiny-live-last. Treat the $500 as tuition.

## Architecture — Decide / Guard / Sign

Three layers, each can veto, none can act alone unsafely:

- **Decide** — a deterministic swing strategy proposes a setup; GLM (z.ai, via OpenAI-compatible API) acts as a **regime/sentiment gate** that approves / denies / sizes it. The LLM never issues orders directly.
- **Guard** — pure deterministic code enforces the risk envelope ($5/trade, position caps, mandatory stop, −3%/day halt, −15% kill). **The LLM cannot touch these.**
- **Sign** — a dedicated hot-wallet signer executes only guard-approved intents. In live mode the hot-wallet key lives in AWS KMS and never leaves it; an on-chain `Pausable` backstop re-clamps size and can halt everything.

Everything is recorded in a **hash-chained append-only audit log** (SQLite) — the single source of truth, replayable.

## Key decisions (verified, mid-2026)

| Area | Choice | Why |
|---|---|---|
| Chain | **Base** | ~$0.02/swap, #1 L2 stablecoin supply, deep majors liquidity, best Python tooling |
| DEX | **CowSwap** | structural MEV protection (batch auction) + gasless; `cow-py` SDK |
| Brain | **GLM 4.6/5.2 via z.ai PAYG** | OpenAI-compatible, function-calling; <$1/mo at swing cadence |
| Signer | **Dedicated hot wallet (AWS KMS)** | key never leaves AWS; main wallet never used |
| Stops | **Keeper** (+ Defender Sentinel fallback) | no native on-chain stops |

## Status

**Paper-mode MVP.** Decide/Guard/Sign loop runs against live Base prices with simulated (CowSwap-modelled) fills. Live CowSwap/1inch + KMS signing and the AWS Terraform module come in a later phase.

## Run (paper mode)

```bash
uv sync                                   # install deps
uv run python scripts/run_paper.py --once # one decision cycle (no keys needed)
uv run python scripts/run_paper.py        # scheduled loop (bar-close + keeper)
```

No API keys are required for paper mode — the GLM overlay returns a neutral gate until `ZAI_API_KEY` is set.

## Tests

```bash
uv run pytest
```

## Safety invariants (enforced in code)

- Risk ≤ **1% equity / trade**; ≤ **25% equity / position**; ≤ **2 concurrent positions**.
- **Mandatory stop** on every position; keeper sells when breached.
- **−3%/day → halt**; **−15% drawdown → kill + manual re-arm**; no auto-resume.
- Hot wallet holds **only what the bot may lose**; profits swept to cold.
- Spot only (1×). No leverage at this size.

See `config/settings.yaml` for the full envelope. See `CLAUDE.md` / memory for the research behind every choice.
