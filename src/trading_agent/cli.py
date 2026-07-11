"""CLI entrypoint. Paper-mode default.

  uv run python scripts/run_paper.py --once            # single cycle
  uv run python scripts/run_paper.py --simulated --once # offline, synthetic data
  uv run python scripts/run_paper.py                    # scheduled loop
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import structlog
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from trading_agent.agent import TradingAgent
from trading_agent.config import load_settings
from trading_agent.data.market_data import CoinGeckoPriceSource, SimulatedPriceSource
from trading_agent.state.audit_log import AuditLog


def setup_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(__import__("logging"), level)),
    )


def build_price_source(settings, simulated: bool):
    return SimulatedPriceSource(settings) if simulated else CoinGeckoPriceSource(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="On-chain L2 swing trading agent (paper mode)")
    parser.add_argument("--once", action="store_true", help="run a single cycle then exit")
    parser.add_argument("--simulated", action="store_true", help="use synthetic market data (offline)")
    parser.add_argument("--config", default=None, help="path to settings.yaml")
    parser.add_argument("--db", default=None, help="path to the audit sqlite db")
    args = parser.parse_args(argv)

    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    settings = load_settings(Path(args.config) if args.config else None)
    db_path = args.db or os.environ.get("AGENT_DB", "data/agent.sqlite")

    log = structlog.get_logger()
    if not settings.is_paper:
        log.warning("LIVE mode configured but only the paper executor is implemented; refusing to trade live.")
        log.warning("Set mode: paper in settings.yaml until the CowSwap/KMS executor lands.")
        return 2

    audit = AuditLog(db_path)
    price_src = build_price_source(settings, args.simulated)
    agent = TradingAgent(settings, price_src, audit)

    log.info("agent_start", mode=settings.mode, chain=settings.chain_name,
             symbols=settings.symbols, simulated=args.simulated, db=db_path)

    if args.once:
        print(agent.cycle())
        return 0

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(agent.cycle, CronTrigger.from_crontab(settings.keeper.bar_close_cron),
                  id="bar_close", max_instances=1, coalesce=True)
    sched.add_job(agent.keeper_tick, "interval",
                  minutes=settings.keeper.safety_poll_minutes,
                  id="keeper_poll", max_instances=1, coalesce=True)
    log.info("scheduler_started", bar_close_cron=settings.keeper.bar_close_cron,
             safety_poll_min=settings.keeper.safety_poll_minutes)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("agent_stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
