"""Backtesting and historical data utilities for the EMA+ADX bot."""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Optional

import pandas as pd
from ib_insync import Contract, IB, Stock

from config import CLIENT_ID, CURRENCY, EXCHANGE, HOST, PORT, SYMBOL


def rank_better(metrics: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    """Return True if metrics outrank incumbent based on PnL, drawdown, Sharpe."""
    if metrics["net_pnl"] != incumbent["net_pnl"]:
        return metrics["net_pnl"] > incumbent["net_pnl"]
    if metrics["max_drawdown"] != incumbent["max_drawdown"]:
        return metrics["max_drawdown"] < incumbent["max_drawdown"]
    return metrics["sharpe"] > incumbent["sharpe"]


def fetch_historical_dataframe(timeframe_minutes: int, lookback_days: int = 120) -> pd.DataFrame:
    """Fetch historical bars from IBKR and aggregate to timeframe."""
    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID + 100, timeout=5)
        contract: Contract = Stock(SYMBOL, EXCHANGE, CURRENCY)
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{lookback_days} D",
            barSizeSetting=ib_bar_size_setting(timeframe_minutes),
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        rows = []
        for b in bars:
            ts = b.date if isinstance(b.date, dt.datetime) else dt.datetime.strptime(b.date, "%Y%m%d %H:%M:%S")
            rows.append({"time": ts, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume})
        return pd.DataFrame(rows)
    finally:
        ib.disconnect()


def ib_bar_size_setting(timeframe_minutes: int) -> str:
    """Map timeframe minutes to an IB-compatible barSizeSetting string."""
    if timeframe_minutes % 60 == 0:
        hours = timeframe_minutes // 60
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{timeframe_minutes} mins"


def _compute_adx_series(df: pd.DataFrame, length: int) -> pd.Series:
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    up_move = highs.diff()
    down_move = lows.shift(1) - lows
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        highs - lows,
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr)
    dx = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).abs()) * 100
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx.fillna(0)


def backtest_strategy(df: pd.DataFrame, params) -> Dict[str, float]:
    """Simulate strategy over historical bars using given parameters."""
    if df.empty:
        return {"net_pnl": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": 0}

    df = df.copy()
    df["fast"] = df["close"].ewm(span=params.fast_ema, adjust=False).mean()
    df["slow"] = df["close"].ewm(span=params.slow_ema, adjust=False).mean()
    df["fast_prev"] = df["fast"].shift(1)
    df["slow_prev"] = df["slow"].shift(1)
    df["adx"] = _compute_adx_series(df, params.adx_length)

    position = 0  # +1 long, -1 short
    entry_price = 0.0
    be_triggered = False
    stop_price: Optional[float] = None
    high_since_entry: Optional[float] = None
    low_since_entry: Optional[float] = None
    cash = 0.0
    equity_curve = []
    trades = 0

    for _, row in df.iterrows():
        fast = row.fast
        slow = row.slow
        prev_fast = row.fast_prev
        prev_slow = row.slow_prev
        adx_val = row.adx
        price = row.close
        high = row.high
        low = row.low

        ema_crossover = prev_fast <= prev_slow and fast > slow
        ema_crossunder = prev_fast >= prev_slow and fast < slow
        adx_condition = adx_val > params.adx_threshold if params.use_adx else True

        # Risk management updates
        if position != 0:
            if position > 0:
                high_since_entry = max(high_since_entry or price, high)
                if params.use_be and high_since_entry > entry_price * (1 + params.be_trigger_percent):
                    be_triggered = True
                base_stop = entry_price * (1 - params.sl_percent) if params.use_sl else None
                if be_triggered:
                    base_stop = max(base_stop or 0, entry_price)
                trail_candidate = None
                if params.use_trail and price > entry_price * (1 + params.trail_offset):
                    trail_candidate = price - entry_price * params.trail_percent
                stop_price = max(v for v in [base_stop, trail_candidate] if v is not None) if any(
                    v is not None for v in [base_stop, trail_candidate]
                ) else stop_price
                if stop_price and low <= stop_price:
                    cash += (stop_price - entry_price)
                    position = 0
                    trades += 1
                    stop_price = None
            else:  # short
                low_since_entry = min(low_since_entry or price, low)
                if params.use_be and low_since_entry < entry_price * (1 - params.be_trigger_percent):
                    be_triggered = True
                base_stop = entry_price * (1 + params.sl_percent) if params.use_sl else None
                if be_triggered:
                    base_stop = min(base_stop or float("inf"), entry_price)
                trail_candidate = None
                if params.use_trail and price < entry_price * (1 - params.trail_offset):
                    trail_candidate = price + entry_price * params.trail_percent
                stop_price = min(v for v in [base_stop, trail_candidate] if v is not None) if any(
                    v is not None for v in [base_stop, trail_candidate]
                ) else stop_price
                if stop_price and high >= stop_price:
                    cash += (entry_price - stop_price)
                    position = 0
                    trades += 1
                    stop_price = None

        # Reverse exits
        if position > 0 and ema_crossunder:
            cash += (price - entry_price)
            position = 0
            trades += 1
            stop_price = None
        elif position < 0 and ema_crossover:
            cash += (entry_price - price)
            position = 0
            trades += 1
            stop_price = None

        # Entries
        if position <= 0 and ema_crossover and adx_condition and params.trade_direction in ("Long", "Both"):
            position = 1
            entry_price = price
            be_triggered = False
            high_since_entry = price
            low_since_entry = price
            stop_price = None
        elif position >= 0 and ema_crossunder and adx_condition and params.trade_direction in ("Short", "Both"):
            position = -1
            entry_price = price
            be_triggered = False
            high_since_entry = price
            low_since_entry = price
            stop_price = None

        equity_curve.append(cash + position * price)

    if not equity_curve:
        return {"net_pnl": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": 0}

    equity_series = pd.Series(equity_curve)
    net_pnl = equity_series.iloc[-1]
    peak = equity_series.cummax()
    drawdown = (equity_series - peak).min()
    returns = equity_series.diff().fillna(0)
    sharpe = returns.mean() / returns.std() * math.sqrt(252) if returns.std() > 0 else 0.0
    return {
        "net_pnl": float(net_pnl),
        "max_drawdown": float(drawdown),
        "sharpe": float(sharpe),
        "trades": trades,
    }
