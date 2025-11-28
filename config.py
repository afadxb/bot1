"""Configuration constants and runtime config helpers for the EMA+ADX bot."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Dict, Literal

HOST = "127.0.0.1"
PORT = 7497  # 7497 for paper, 7496 for live
CLIENT_ID = 104

SYMBOL = "TSLA"
EXCHANGE = "SMART"
CURRENCY = "USD"

DEFAULT_TIMEFRAME_MINUTES = 180  # 3-hour default for live trading
DEFAULT_INITIAL_CAPITAL = 100_000.0
MIN_HISTORY_BARS = 100  # to warm up indicators

LIVE_TRADING = True  # Set to True to actually transmit orders
ENABLE_WEEKLY_OPTIMIZATION = True
WEEKLY_OPTIMIZATION_DAY = 6  # Sunday (0=Monday ... 6=Sunday)
WEEKLY_OPTIMIZATION_HOUR = 20  # 8pm UTC by default
CONFIG_PATH = "ema_adx_config.json"
ENABLE_MARKET_HOURLY_LOOP = True
MARKET_OPEN_UTC = dt.time(13, 30)  # 9:30 AM ET
MARKET_CLOSE_UTC = dt.time(20, 0)  # 4:00 PM ET
MARKET_DAYS = {0, 1, 2, 3, 4}  # Monday-Friday


@dataclass
class StrategyConfig:
    """Runtime configuration loaded from disk or defaults."""

    symbol: str = SYMBOL
    timeframe_minutes: int = DEFAULT_TIMEFRAME_MINUTES
    ema_fast: int = 21
    ema_slow: int = 50
    adx_length: int = 14
    adx_threshold: float = 15.0
    use_adx: bool = True
    use_sl: bool = False
    sl_percent: float = 0.01
    use_be: bool = True
    be_trigger_percent: float = 0.01
    use_trail: bool = False
    trail_offset: float = 0.005
    trail_percent: float = 0.015
    trade_direction: Literal["Long", "Short", "Both"] = "Both"
    position_size_pct: float = 10.0
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    lookback_days: int = 2200
    risk_params: Dict[str, float] = field(default_factory=dict)


def load_config(path: str = CONFIG_PATH) -> StrategyConfig:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded configuration from {path}.")
        # Backward compatibility for renamed keys
        if "fast_ema" in data and "ema_fast" not in data:
            data["ema_fast"] = data.pop("fast_ema")
        if "slow_ema" in data and "ema_slow" not in data:
            data["ema_slow"] = data.pop("slow_ema")
        return StrategyConfig(**data)
    except FileNotFoundError:
        print("No existing configuration found; using defaults.")
    except Exception as exc:  # pragma: no cover - user file errors
        print(f"Failed to load configuration: {exc}. Using defaults.")
    return StrategyConfig()


def save_config(config: StrategyConfig, path: str = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"Saved configuration to {path}.")
