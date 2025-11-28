"""Backtesting and shared strategy logic for the EMA Cross + ADX strategy."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from config import StrategyConfig
from data_cache import get_resampled_bars


def rank_better(metrics: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    """Return True if metrics outrank incumbent based on PnL, drawdown, Sharpe."""
    if metrics["net_pnl"] != incumbent["net_pnl"]:
        return metrics["net_pnl"] > incumbent["net_pnl"]
    if metrics["max_drawdown"] != incumbent["max_drawdown"]:
        return metrics["max_drawdown"] > incumbent["max_drawdown"]
    return metrics["sharpe"] > incumbent["sharpe"]


@dataclass
class StrategyState:
    cash: float
    position_qty: int = 0  # signed
    position_side: int = 0  # 1 long, -1 short, 0 flat
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    high_since_entry: Optional[float] = None
    low_since_entry: Optional[float] = None
    be_triggered: bool = False
    trades: int = 0


def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1 / length if length > 0 else 1
    return series.ewm(alpha=alpha, adjust=False).mean()


def compute_dmi_adx(df: pd.DataFrame, length: int) -> pd.Series:
    """Return ADX series equivalent to Pine's ta.dmi(length, length).adx."""
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)

    up_move = highs.diff()
    down_move = lows.shift(1) - lows

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_components = pd.concat(
        [
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows - closes.shift()).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr = _rma(true_range, length)
    plus_rma = _rma(plus_dm, length)
    minus_rma = _rma(minus_dm, length)

    plus_di = 100 * (plus_rma / atr)
    minus_di = 100 * (minus_rma / atr)

    dx = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).abs()) * 100
    adx = _rma(dx, length)
    return adx.fillna(0)


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def fetch_historical_dataframe(timeframe_minutes: int, lookback_days: int) -> pd.DataFrame:
    """Fetch historical bars from cache (hourly base) and resample as needed."""
    return get_resampled_bars(timeframe_minutes, lookback_days)


def ib_bar_size_setting(timeframe_minutes: int) -> str:
    """Map timeframe minutes to an IB-compatible barSizeSetting string."""
    if timeframe_minutes % 60 == 0:
        hours = timeframe_minutes // 60
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{timeframe_minutes} mins"


def ema_cross_signals(fast: float, slow: float, prev_fast: float, prev_slow: float) -> Dict[str, bool]:
    return {
        "cross_up": prev_fast <= prev_slow and fast > slow,
        "cross_down": prev_fast >= prev_slow and fast < slow,
    }


def update_stop_price(state: StrategyState, bar: pd.Series, config: StrategyConfig) -> Optional[float]:
    if state.position_side == 0 or state.entry_price is None:
        return None

    price = float(bar.close)
    high = float(bar.high)
    low = float(bar.low)

    if state.position_side > 0:
        state.high_since_entry = max(state.high_since_entry or price, high)
        base_stop = state.entry_price * (1 - config.sl_percent) if config.use_sl else None
        if config.use_be and state.high_since_entry >= state.entry_price * (1 + config.be_trigger_percent):
            state.be_triggered = True
        be_stop = state.entry_price if state.be_triggered else None
        trail_stop = None
        if config.use_trail and price >= state.entry_price * (1 + config.trail_offset):
            trail_stop = state.high_since_entry - state.entry_price * config.trail_percent
        candidates = [c for c in [base_stop, be_stop, trail_stop, state.stop_price] if c is not None]
        state.stop_price = max(candidates) if candidates else None
    else:
        state.low_since_entry = min(state.low_since_entry or price, low)
        base_stop = state.entry_price * (1 + config.sl_percent) if config.use_sl else None
        if config.use_be and state.low_since_entry <= state.entry_price * (1 - config.be_trigger_percent):
            state.be_triggered = True
        be_stop = state.entry_price if state.be_triggered else None
        trail_stop = None
        if config.use_trail and price <= state.entry_price * (1 - config.trail_offset):
            trail_stop = state.low_since_entry + state.entry_price * config.trail_percent
        candidates = [c for c in [base_stop, be_stop, trail_stop, state.stop_price] if c is not None]
        state.stop_price = min(candidates) if candidates else None
    return state.stop_price


def _close_position(state: StrategyState, price: float) -> None:
    if state.position_side == 0:
        return
    state.cash += state.position_qty * price
    state.position_qty = 0
    state.position_side = 0
    state.entry_price = None
    state.stop_price = None
    state.high_since_entry = None
    state.low_since_entry = None
    state.be_triggered = False
    state.trades += 1


def _open_position(state: StrategyState, price: float, side: int, config: StrategyConfig, equity: float) -> None:
    invest_value = equity * (config.position_size_pct / 100.0)
    qty = math.floor(invest_value / price)
    if qty <= 0:
        return
    state.position_qty = qty if side > 0 else -qty
    state.position_side = side
    state.entry_price = price
    if side > 0:
        state.cash -= qty * price
        state.high_since_entry = price
        state.low_since_entry = price
    else:
        state.cash += qty * price
        state.high_since_entry = price
        state.low_since_entry = price
    state.stop_price = None
    state.be_triggered = False


def backtest_strategy(df: pd.DataFrame, params: StrategyConfig) -> Dict[str, float]:
    """Simulate strategy over historical bars using TradingView-like semantics."""
    if df is None or df.empty:
        return {"net_pnl": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": 0}

    df = df.copy().reset_index(drop=True)
    df["fast"] = compute_ema(df["close"], params.ema_fast)
    df["slow"] = compute_ema(df["close"], params.ema_slow)
    df["fast_prev"] = df["fast"].shift(1)
    df["slow_prev"] = df["slow"].shift(1)
    df["adx"] = compute_dmi_adx(df, params.adx_length)

    state = StrategyState(cash=params.initial_capital)
    equity_curve = []

    for _, row in df.iterrows():
        price = float(row.close)
        signals = ema_cross_signals(row.fast, row.slow, row.fast_prev, row.slow_prev)
        adx_condition = (row.adx > params.adx_threshold) if params.use_adx else True

        if state.position_side != 0:
            update_stop_price(state, row, params)
            if state.stop_price is not None:
                if (state.position_side > 0 and row.low <= state.stop_price) or (
                    state.position_side < 0 and row.high >= state.stop_price
                ):
                    _close_position(state, state.stop_price)
                    price = float(row.close)  # fall through for potential reverse/entry

        if state.position_side > 0 and signals["cross_down"]:
            _close_position(state, price)
            if params.trade_direction in ("Short", "Both") and adx_condition:
                equity = state.cash
                _open_position(state, price, -1, params, equity)
        elif state.position_side < 0 and signals["cross_up"]:
            _close_position(state, price)
            if params.trade_direction in ("Long", "Both") and adx_condition:
                equity = state.cash
                _open_position(state, price, 1, params, equity)
        elif state.position_side == 0:
            equity = state.cash
            if signals["cross_up"] and adx_condition and params.trade_direction in ("Long", "Both"):
                _open_position(state, price, 1, params, equity)
            elif signals["cross_down"] and adx_condition and params.trade_direction in ("Short", "Both"):
                _open_position(state, price, -1, params, equity)

        equity_curve.append(state.cash + state.position_qty * price)

    if state.position_side != 0 and not df.empty:
        last_price = float(df.iloc[-1].close)
        _close_position(state, last_price)
        equity_curve[-1] = state.cash

    if not equity_curve:
        return {"net_pnl": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": 0}

    equity_series = pd.Series(equity_curve)
    net_pnl = equity_series.iloc[-1] - params.initial_capital
    peak = equity_series.cummax()
    drawdown = (equity_series - peak).min()
    returns = equity_series.pct_change().fillna(0)
    sharpe = returns.mean() / returns.std() * math.sqrt(252) if returns.std() > 0 else 0.0
    return {
        "net_pnl": float(net_pnl),
        "max_drawdown": float(drawdown),
        "sharpe": float(sharpe),
        "trades": state.trades,
    }


def run_basic_test() -> None:
    """Simple test entry point to compare against TradingView defaults."""
    cfg = StrategyConfig()
    df = fetch_historical_dataframe(cfg.timeframe_minutes, cfg.lookback_days)
    metrics = backtest_strategy(df, cfg)
    start_date = df["time"].iloc[0] if not df.empty else None
    end_date = df["time"].iloc[-1] if not df.empty else None
    print(
        f"Symbol={cfg.symbol} TF={cfg.timeframe_minutes}m Start={start_date} End={end_date} "
        f"PnL={metrics['net_pnl']:.2f} DD={metrics['max_drawdown']:.2f} "
        f"Sharpe={metrics['sharpe']:.2f} Trades={metrics['trades']} FinalEq={metrics['net_pnl'] + cfg.initial_capital:.2f}"
    )


if __name__ == "__main__":
    run_basic_test()
