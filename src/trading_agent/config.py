"""Configuration: typed Settings loaded from config/settings.yaml + env secrets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from trading_agent.models import utcnow


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float
    max_position_notional_pct: float
    max_concurrent_positions: int
    max_deployed_pct: float
    daily_loss_cap_pct: float
    drawdown_alert_pct: float
    drawdown_kill_pct: float
    max_trades_per_day: int
    leverage_cap: int = 1
    # Sizing mode: "risk_pct" (size from risk%/stop-distance) or "fixed_fraction"
    # (deploy a fixed % of equity per position — for long/flat trend where the
    # strategy's exit signal, not a tight stop, is the real risk control).
    sizing_mode: str = "risk_pct"
    fixed_fraction_pct: float = 25.0
    # vol-target (TSMOM) sizing: deploy (target_vol / realized_vol) * equity, clamped.
    target_annual_vol_pct: float = 40.0


class StrategyConfig(BaseModel):
    name: str = "swing"  # swing | trend_long_flat | ema_cross | mean_reversion | momentum | forecast
    timeframe: str
    donchian_period: int
    ema_fast: int
    ema_slow: int
    atr_period: int
    stop_atr_mult: float
    min_stop_pct: float
    take_profit_atr_mult: float
    volume_filter_mult: float


class UniverseItem(BaseModel):
    symbol: str
    base: str
    quote: str
    coingecko_id: str
    decimals: int


class GLMConfig(BaseModel):
    base_url: str
    model: str
    overlay_model: str
    api_key_env: str
    temperature: float
    max_tokens: int


class ExecutionConfig(BaseModel):
    router: str
    slippage_bps: int
    stop_slippage_bps: int
    cowswap_api_url: str
    deadline_seconds: int


class KeeperConfig(BaseModel):
    bar_close_cron: str
    safety_poll_minutes: int


class Settings(BaseModel):
    mode: Literal["paper", "live"]
    starting_equity_usd: float
    chain_name: str
    chain_id: int
    universe: list[UniverseItem]
    risk: RiskConfig
    strategy: StrategyConfig
    glm: GLMConfig
    execution: ExecutionConfig
    keeper: KeeperConfig
    aws_region: str

    # env-derived secrets
    rpc_url: str = ""
    glm_api_key: Optional[str] = None
    hot_wallet_address: Optional[str] = None
    hot_wallet_kms_key_id: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    @property
    def symbols(self) -> list[str]:
        return [u.symbol for u in self.universe]

    def universe_item(self, symbol: str) -> UniverseItem:
        for u in self.universe:
            if u.symbol == symbol:
                return u
        raise KeyError(f"symbol {symbol} not in universe")


def load_settings(yaml_path: Optional[Path] = None, env: Optional[dict] = None) -> Settings:
    """Load settings.yaml and merge env secrets. Env defaults to os.environ."""
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    env = env if env is not None else os.environ
    raw = yaml.safe_load(yaml_path.read_text())

    glm = raw["glm"]
    api_key = env.get(glm["api_key_env"]) or None

    return Settings(
        mode=raw["mode"],
        starting_equity_usd=raw["account"]["starting_equity_usd"],
        chain_name=raw["chain"]["name"],
        chain_id=raw["chain"]["chain_id"],
        universe=raw["universe"],
        risk=raw["risk"],
        strategy=raw["strategy"],
        glm=raw["glm"],
        execution=raw["execution"],
        keeper=raw["keeper"],
        aws_region=raw["deployment"]["aws_region"],
        rpc_url=env.get("BASE_RPC_URL", ""),
        glm_api_key=api_key,
        hot_wallet_address=env.get("HOT_WALLET_ADDRESS") or None,
        hot_wallet_kms_key_id=env.get("HOT_WALLET_KMS_KEY_ID") or None,
        telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=env.get("TELEGRAM_CHAT_ID") or None,
    )


__all__ = [
    "Settings",
    "RiskConfig",
    "StrategyConfig",
    "UniverseItem",
    "GLMConfig",
    "ExecutionConfig",
    "KeeperConfig",
    "load_settings",
    "utcnow",
]
