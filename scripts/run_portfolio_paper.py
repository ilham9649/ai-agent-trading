#!/usr/bin/env python3
"""Portfolio PAPER trader: 10-asset equal-weight CRP-bands + hysteresis crash gate.

Runs on LIVE Base/Binance prices, simulates fills (paper), persists state to
data/portfolio_state.json, logs every action to the audit log. Paper-only — no real funds.
  uv run python scripts/run_portfolio_paper.py --once     # one rebalance cycle
  uv run python scripts/run_portfolio_paper.py            # daily scheduled loop
"""
import json
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402

from trading_agent.backtest.feed import binance_series  # noqa: E402
from trading_agent.config import UniverseItem, load_settings  # noqa: E402
from trading_agent.state.audit_log import AuditLog  # noqa: E402

BASKET = ["cbBTC-USDC", "WETH-USDC", "SOL-USDC", "BNB-USDC", "XRP-USDC",
          "ADA-USDC", "AVAX-USDC", "DOGE-USDC", "LINK-USDC", "DOT-USDC"]
HISTORY = 500
BAND = 0.10
EXIT_DD = 0.20
REENTRY_DD = 0.05
FEE = 0.001
SLIP = 0.001
STATE_PATH = Path("data/portfolio_state.json")


def _D(x):
    return Decimal(str(x))


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"holdings": {s: 0.0 for s in BASKET}, "cash": 0.0, "invested": True, "started": False}


def save_state(st):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2))


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    once = "--once" in argv
    settings = load_settings()
    settings.universe = list(settings.universe) + [
        UniverseItem(symbol=s, base=s.split("-")[0], quote="USDC", coingecko_id=s.lower(), decimals=6)
        for s in BASKET if s not in settings.symbols]
    audit = AuditLog("data/portfolio_paper.sqlite")
    cycle_id = f"p-{uuid.uuid4().hex[:8]}"
    st = load_state()

    # fetch live history + current prices
    prices, basket = {}, []
    for sym in BASKET:
        df = binance_series(settings, sym, interval="1d", total=HISTORY)
        px = float(df["close"].iloc[-1])
        prices[sym] = px
        norm = df["close"].to_numpy(float) / df["close"].to_numpy(float)[0]
        basket.append(norm)
    import numpy as np
    idx = np.mean(np.stack(basket), axis=0)
    dd = float((idx[-1] - idx.max()) / idx.max())  # current drawdown from window high

    # initial funding (paper): if not started, deposit $500
    if not st["started"]:
        st["cash"] = 500.0
        st["started"] = True

    # crash gate: exit at dd>20%; re-enter on a NEW 60-DAY HIGH (fires during recovery,
    # not at the old peak — avoids sitting in cash through the whole rebound)
    new60hi = bool(idx[-1] >= float(np.max(idx[-61:-1])))
    if st["invested"] and dd < -EXIT_DD:
        st["invested"] = False
        audit.append("killswitch", {"cycle_id": cycle_id, "reason": f"basket dd {dd:.1%}<=-{EXIT_DD:.0%}"}, cycle_id)
    elif (not st["invested"]) and new60hi:
        st["invested"] = True
        audit.append("heartbeat", {"cycle_id": cycle_id, "note": "re-enter on new 60-day high"}, cycle_id)

    n = len(BASKET)
    target = (1.0 / n) if st["invested"] else 0.0
    equity = st["cash"] + sum(st["holdings"][s] * prices[s] for s in BASKET)

    # rebalance if gate flipped or any weight drifts past band
    weights = {s: (st["holdings"][s] * prices[s] / equity if equity > 0 else 0.0) for s in BASKET}
    need_reb = st["invested"] is False or max(
        (abs(weights[s] - target) for s in BASKET), default=0) > BAND
    trades = []
    if need_reb:
        for s in BASKET:
            desired_value = target * equity
            desired_qty = desired_value / prices[s]
            delta = desired_qty - st["holdings"][s]
            if abs(delta) < 1e-8:
                continue
            if delta > 0:  # buy
                cost = delta * prices[s] * (1 + SLIP)
                st["cash"] -= cost + delta * prices[s] * FEE
            else:  # sell
                proceeds = (-delta) * prices[s] * (1 - SLIP)
                st["cash"] += proceeds - (-delta) * prices[s] * FEE
            st["holdings"][s] = desired_qty
            trades.append({"sym": s, "side": "buy" if delta > 0 else "sell", "qty": round(desired_qty, 6),
                           "price": round(prices[s], 4)})
        audit.append("order_request", {"cycle_id": cycle_id, "trades": trades,
                                       "invested": st["invested"], "basket_dd": round(dd, 4)}, cycle_id)

    new_equity = st["cash"] + sum(st["holdings"][s] * prices[s] for s in BASKET)
    save_state(st)
    audit.append("observation", {"cycle_id": cycle_id, "equity": round(new_equity, 2),
                                 "invested": st["invested"], "basket_dd": round(dd, 4),
                                 "n_trades": len(trades)}, cycle_id)
    print(f"[{cycle_id}] dd={dd:+.1%} invested={st['invested']} equity=${new_equity:.2f} "
          f"trades={len(trades)} target_w={target:.2f}")

    if once:
        return 0
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(lambda: main(["--once"]), "cron", hour=1, minute=7, id="daily_rebalance")
    print("Portfolio paper-trader scheduled (daily ~01:07 UTC). Ctrl-C to stop.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
