"""Historical bar feeds for backtesting: Binance public klines (real, years),
CoinGecko OHLCV, or synthetic (offline). Data-only — never places trades."""
from __future__ import annotations

from typing import Optional

import httpx
import pandas as pd

from trading_agent.config import Settings
from trading_agent.data.market_data import CoinGeckoPriceSource, SimulatedPriceSource

_BASE_MAP = {"WETH": "ETH", "cbBTC": "BTC", "ETH": "ETH", "BTC": "BTC",
             "SOL": "SOL", "BNB": "BNB", "XRP": "XRP", "AVAX": "AVAX", "DOGE": "DOGE",
             "ADA": "ADA", "LINK": "LINK", "DOT": "DOT", "LTC": "LTC", "TRX": "TRX",
             "ATOM": "ATOM", "NEAR": "NEAR", "MATIC": "MATIC", "UNI": "UNI", "XLM": "XLM"}
_BINANCE_DATA_URL = "https://data-api.binance.vision/api/v3/klines"  # public market-data mirror (no auth, no geo-block)


def binance_series(settings: Settings, symbol: str, interval: str = "4h",
                   total: int = 2000) -> pd.DataFrame:
    """Real OHLCV history from Binance public data (no key). Maps our symbol to a
    USDT pair (e.g. WETH-USDC -> ETHUSDT, cbBTC-USDC -> BTCUSDT)."""
    u = settings.universe_item(symbol)
    bsym = f"{_BASE_MAP.get(u.base, u.base)}USDT"
    rows: list = []
    end_time: Optional[int] = None
    with httpx.Client(timeout=60.0) as client:
        while len(rows) < total:
            params = {"symbol": bsym, "interval": interval, "limit": min(1000, total - len(rows))}
            if end_time:
                params["endTime"] = end_time
            batch = None
            for attempt in range(4):
                try:
                    r = client.get(_BINANCE_DATA_URL, params=params)
                    r.raise_for_status()
                    batch = r.json()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    import time
                    time.sleep(1.5 * (attempt + 1))
            if not batch:
                break
            rows = batch + rows  # batch is old->new; prepend to keep chronological
            end_time = batch[0][0] - 1
            if len(batch) < 1000:
                break
    cols = ["ts", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "i"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates().reset_index(drop=True)
    return df.tail(total).reset_index(drop=True)


def synthetic_series(settings: Settings, bars: int = 1500, seed: int = 42,
                     symbols: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    src = SimulatedPriceSource(settings, seed=seed)
    syms = symbols or settings.symbols
    return {sym: src.candles(sym, bars=bars) for sym in syms}


def coingecko_series(settings: Settings, bars: int = 500,
                     symbols: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    src = CoinGeckoPriceSource(settings)
    syms = symbols or settings.symbols
    try:
        return {sym: src.candles(sym, bars=bars) for sym in syms}
    finally:
        src.close()


def is_oos_split(series: dict[str, pd.DataFrame], oos_frac: float = 0.3
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Deterministic in-sample / out-of-sample split (same row cut per symbol)."""
    in_sample, oos = {}, {}
    for sym, df in series.items():
        cut = int(len(df) * (1 - oos_frac))
        in_sample[sym] = df.iloc[:cut].reset_index(drop=True)
        oos[sym] = df.iloc[cut:].reset_index(drop=True)
    return in_sample, oos
