"""
EMA Cross + ADX Filter trading bot for Interactive Brokers (IBKR). 

This script recreates the provided TradingView Pine Script strategy using the
ib_insync library. It is intended for educational / paper-trading use. Review
all parameters before enabling LIVE_TRADING.
"""
from __future__ import annotations

import datetime as dt
import math
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
from ib_insync import (BarDataList, Contract, IB, LimitOrder, Order,
                       RealTimeBar, Stock)

from data_cache import get_resampled_bars
from config import (CLIENT_ID, CONFIG_PATH, CURRENCY, ENABLE_MARKET_HOURLY_LOOP,
                    ENABLE_WEEKLY_OPTIMIZATION, EXCHANGE, HOST, LIVE_TRADING,
                    MARKET_CLOSE_UTC, MARKET_DAYS, MARKET_OPEN_UTC,
                    MIN_HISTORY_BARS, PORT, SYMBOL, WEEKLY_OPTIMIZATION_DAY,
                    WEEKLY_OPTIMIZATION_HOUR, StrategyConfig, load_config,
                    save_config)
from weekly_optimization import print_top_results, run_weekly_optimization_once

LOG_PATH = Path("logs/bot.log")


@dataclass
class Bar:
    """Simple OHLCV container for aggregated timeframe bars."""

    time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IndicatorState:
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    adx: float = 0.0
    prev_fast_ema: float = 0.0
    prev_slow_ema: float = 0.0


class EmaAdxBot:
    def __init__(self, config: StrategyConfig) -> None:
        self.ib = IB()
        self.contract: Contract = Stock(SYMBOL, EXCHANGE, CURRENCY)
        self.bars: List[Bar] = []
        self.current_bar: Optional[Bar] = None
        self.indicators = IndicatorState()
        self.position_side: str = "flat"  # "long", "short", "flat"
        self.entry_price: float = 0.0
        self.stop_order: Optional[Order] = None
        self.be_triggered: bool = False
        self.high_since_entry: Optional[float] = None
        self.low_since_entry: Optional[float] = None
        self.config = config
        self._maintenance_thread: Optional[threading.Thread] = None
        self._last_optimization_date: Optional[dt.date] = None
        self._last_hourly_tick: Optional[dt.datetime] = None
        self.logger = logging.getLogger("ema_adx_bot")

    # -------------------------
    # Connection helpers
    # -------------------------
    def connect(self) -> None:
        """Connect to TWS/IB Gateway with simple retry logic."""
        try:
            self.logger.info(f"Connecting to IBKR at {HOST}:{PORT} (clientId={CLIENT_ID})...")
            self.ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=5)
            if self.ib.isConnected():
                self.logger.info("Connected to IBKR.")
        except Exception as exc:  # pragma: no cover - network errors
            self.logger.warning(f"Initial connection failed: {exc}")
            self.logger.info("Retrying in 5 seconds...")
            self.ib.sleep(5)
            self.ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=5)

    # -------------------------
    # Data handling
    # -------------------------
    def start(self) -> None:
        """Entry point to start subscriptions, scheduler, and event loop."""
        self.connect()
        self.ib.qualifyContracts(self.contract)
        self._load_initial_history()

        if ENABLE_WEEKLY_OPTIMIZATION or ENABLE_MARKET_HOURLY_LOOP:
            self._maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
            self._maintenance_thread.start()

        if ENABLE_MARKET_HOURLY_LOOP:
            self._hourly_thread = threading.Thread(target=self._hourly_market_loop, daemon=True)
            self._hourly_thread.start()

        rt_bars: BarDataList = self.ib.reqRealTimeBars(
            self.contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=True,
            realTimeBarsOptions=[],
        )
        rt_bars.updateEvent += self.on_realtime_bar
        self.logger.info("Subscribed to real-time bars. Running event loop...")
        self.ib.run()

    def _maintenance_loop(self) -> None:
        """Simple loop to coordinate weekly optimization and hourly market heartbeats."""
        while True:
            now = dt.datetime.utcnow()

            if ENABLE_WEEKLY_OPTIMIZATION and (
                now.weekday() == WEEKLY_OPTIMIZATION_DAY
                and now.hour == WEEKLY_OPTIMIZATION_HOUR
                and (self._last_optimization_date is None or self._last_optimization_date != now.date())
            ):
                self.logger.info("Starting weekly optimization...")
                self._last_optimization_date = now.date()
                try:
                    self.run_weekly_optimization()
                except Exception as exc:  # pragma: no cover - runtime safety
                    self.logger.exception(f"Weekly optimization failed: {exc}")

            if ENABLE_MARKET_HOURLY_LOOP and self._should_emit_hourly(now):
                self._last_hourly_tick = now
                status = "connected" if self.ib.isConnected() else "disconnected"
                self.logger.info(
                    f"[HOURLY] Market session heartbeat at {now.isoformat()} UTC. "
                    f"Trading loop is {status}."
                )

            time.sleep(30)

    def _hourly_market_loop(self) -> None:
        """Emit an hourly heartbeat during market hours to confirm the trading loop is active."""
        while True:
            now = dt.datetime.utcnow()
            if self._should_emit_hourly(now):
                self._last_hourly_tick = now
                status = "connected" if self.ib.isConnected() else "disconnected"
                self.logger.info(
                    f"[HOURLY] Market session heartbeat at {now.isoformat()} UTC. "
                    f"Trading loop is {status}."
                )
            time.sleep(30)

    def _market_session_open(self, now: dt.datetime) -> bool:
        return now.weekday() in MARKET_DAYS and MARKET_OPEN_UTC <= now.time() <= MARKET_CLOSE_UTC

    def _should_emit_hourly(self, now: dt.datetime) -> bool:
        """Emit immediately on start, then roughly every hour during market hours (no minute==0 dependency)."""
        if not self._market_session_open(now):
            return False
        if self._last_hourly_tick is None:
            return True
        return (now - self._last_hourly_tick) >= dt.timedelta(hours=1) - dt.timedelta(seconds=5)

    @staticmethod
    def _rt_value(bar: RealTimeBar, field: str) -> float:
        """Access RT bar fields consistently across ib_insync versions."""
        val = getattr(bar, field, None)
        if val is None:
            val = getattr(bar, f"{field}_", None)
        if val is None:
            raise AttributeError(f"RealTimeBar missing expected field '{field}'/'{field}_'")
        return val

    def _load_initial_history(self) -> None:
        """Prefill bars using historical data to warm up indicators."""
        hist_df = get_resampled_bars(self.config.timeframe_minutes, lookback_days=30)
        if hist_df.empty:
            self.logger.warning("No historical data available from cache/IB.")
            return
        for row in hist_df.itertuples():
            bar = Bar(
                time=self._to_naive_utc(row.time.to_pydatetime() if hasattr(row.time, "to_pydatetime") else row.time),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            self.bars.append(bar)
        self.logger.info(f"Loaded {len(self.bars)} historical bars from cache/IB.")
        if self.bars:
            self.current_bar = self.bars[-1]
            self.update_indicators()
            self._refresh_position_state()

    def on_realtime_bar(self, bars: BarDataList, has_new_bar: bool) -> None:
        """Aggregate IB 5-second bars into the configured timeframe bars."""
        if not has_new_bar or not bars:
            return

        latest_bar = bars[-1]

        bar_open = self._rt_value(latest_bar, "open")
        bar_high = self._rt_value(latest_bar, "high")
        bar_low = self._rt_value(latest_bar, "low")
        bar_close = self._rt_value(latest_bar, "close")
        bar_volume = self._rt_value(latest_bar, "volume")

        if isinstance(latest_bar.time, dt.datetime):
            bar_end = self._to_naive_utc(latest_bar.time)
        else:
            bar_end = dt.datetime.utcfromtimestamp(latest_bar.time)

        frame_start = self._floor_time(bar_end, self.config.timeframe_minutes)

        if self.current_bar is None or frame_start > self.current_bar.time:
            if self.current_bar:
                # Close previous bar and act on it
                self.bars.append(self.current_bar)
                self.logger.info(f"New bar closed at {self.current_bar.time}. Close={self.current_bar.close:.2f}")
                self.on_bar_close()
            # start a new bar
            self.current_bar = Bar(
                time=frame_start,
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=bar_close,
                volume=bar_volume,
            )
        else:
            # update current bar
            self.current_bar.high = max(self.current_bar.high, bar_high)
            self.current_bar.low = min(self.current_bar.low, bar_low)
            self.current_bar.close = bar_close
            self.current_bar.volume += bar_volume

    def _floor_time(self, ts: dt.datetime, minutes: int) -> dt.datetime:
        discard = dt.timedelta(minutes=ts.minute % minutes, seconds=ts.second, microseconds=ts.microsecond)
        return ts - discard

    @staticmethod
    def _to_naive_utc(ts: dt.datetime) -> dt.datetime:
        """Normalize datetimes to naive UTC for consistent comparisons."""
        if ts.tzinfo:
            return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return ts.replace(tzinfo=None)

    # -------------------------
    # Indicator calculations
    # -------------------------
    def update_indicators(self) -> None:
        closes = [b.close for b in self.bars]
        if len(closes) < max(self.config.fast_ema, self.config.slow_ema, self.config.adx_length + 1):
            return
        series = pd.Series(closes)
        fast_ema = series.ewm(span=self.config.fast_ema, adjust=False).mean()
        slow_ema = series.ewm(span=self.config.slow_ema, adjust=False).mean()

        prev_fast = self.indicators.fast_ema
        prev_slow = self.indicators.slow_ema
        self.indicators.prev_fast_ema = prev_fast
        self.indicators.prev_slow_ema = prev_slow
        self.indicators.fast_ema = float(fast_ema.iloc[-1])
        self.indicators.slow_ema = float(slow_ema.iloc[-1])
        self.indicators.adx = self._compute_adx()

    def _compute_adx(self) -> float:
        if len(self.bars) < self.config.adx_length + 1:
            return 0.0
        highs = pd.Series([b.high for b in self.bars])
        lows = pd.Series([b.low for b in self.bars])
        closes = pd.Series([b.close for b in self.bars])

        up_move = highs.diff()
        down_move = lows.shift(1) - lows

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat([
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows - closes.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / self.config.adx_length, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / self.config.adx_length, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / self.config.adx_length, adjust=False).mean() / atr)
        dx = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).abs()) * 100
        adx = dx.ewm(alpha=1 / self.config.adx_length, adjust=False).mean()
        return float(adx.iloc[-1]) if not math.isnan(adx.iloc[-1]) else 0.0

    def _compute_atr(self, length: int) -> Optional[float]:
        """
        Compute latest ATR over self.bars for the given length.
        Return None if not enough data.
        """
        if len(self.bars) < length + 1:
            return None
        highs = pd.Series([b.high for b in self.bars])
        lows = pd.Series([b.low for b in self.bars])
        closes = pd.Series([b.close for b in self.bars])
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows - closes.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / length, adjust=False).mean()
        value = atr.iloc[-1]
        if math.isnan(value):
            return None
        return float(value)

    # -------------------------
    # Strategy logic
    # -------------------------
    def on_bar_close(self) -> None:
        if len(self.bars) < MIN_HISTORY_BARS:
            self.logger.info(f"Waiting for more history ({len(self.bars)}/{MIN_HISTORY_BARS})...")
            return
        self.update_indicators()
        fast = self.indicators.fast_ema
        slow = self.indicators.slow_ema
        prev_fast = self.indicators.prev_fast_ema
        prev_slow = self.indicators.prev_slow_ema
        adx_val = self.indicators.adx

        self.logger.info(
            f"Bar closed. Fast EMA={fast:.2f}, Slow EMA={slow:.2f}, PrevFast={prev_fast:.2f}, PrevSlow={prev_slow:.2f}, ADX={adx_val:.2f}"
        )

        ema_crossover = prev_fast <= prev_slow and fast > slow
        ema_crossunder = prev_fast >= prev_slow and fast < slow
        adx_condition = (adx_val > self.config.adx_threshold) if self.config.use_adx else True

        self._refresh_position_state()

        # Reverse exits
        if self.position_side == "long" and ema_crossunder:
            self.logger.info("EMA reverse exit triggered for long.")
            self.close_position()
        elif self.position_side == "short" and ema_crossover:
            self.logger.info("EMA reverse exit triggered for short.")
            self.close_position()

        # Entries
        if (self.config.trade_direction in ("Long", "Both")) and ema_crossover and adx_condition:
            if self.position_side != "long":
                if self.position_side == "short":
                    self.close_position()
                self.enter_position("long")
        if (self.config.trade_direction in ("Short", "Both")) and ema_crossunder and adx_condition:
            if self.position_side != "short":
                if self.position_side == "long":
                    self.close_position()
                self.enter_position("short")

        # Risk management adjustments
        self.manage_risk()

    def enter_position(self, side: str) -> None:
        price = self.bars[-1].close
        qty = self._compute_position_size(price)
        if qty <= 0:
            self.logger.info("Computed quantity is 0; skipping entry.")
            return
        action = "BUY" if side == "long" else "SELL"
        limit_price = price
        if self.config.use_marketable_limits:
            if side == "long":
                limit_price = price + self.config.entry_limit_offset
            else:
                limit_price = price - self.config.entry_limit_offset
        limit_price = self._apply_min_tick(limit_price)
        self.logger.info(f"Placing {side} entry for {qty} shares with limit {limit_price:.2f}.")
        order = LimitOrder(action, qty, limit_price, transmit=LIVE_TRADING)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(1)
        if trade.orderStatus.status in {"Filled", "Submitted"}:
            self.position_side = side
            self.entry_price = price
            self.be_triggered = False
            self.stop_order = None
            self.high_since_entry = price
            self.low_since_entry = price
            self.logger.info(f"Entered {side} at ~{price:.2f}. Live={LIVE_TRADING}")

    def close_position(self) -> None:
        self._refresh_position_state()
        if self.position_side == "flat":
            return
        action = "SELL" if self.position_side == "long" else "BUY"
        qty = abs(self._current_position_size())
        last_close = self.bars[-1].close
        limit_price = last_close
        if self.config.use_marketable_limits:
            if action == "SELL":
                limit_price = last_close - self.config.exit_limit_offset
            else:
                limit_price = last_close + self.config.exit_limit_offset
        limit_price = self._apply_min_tick(limit_price)
        if self.position_side == "long":
            approx_pnl = (limit_price - self.entry_price) * qty
        else:
            approx_pnl = (self.entry_price - limit_price) * qty
        self.logger.info(
            f"Closing {self.position_side} position of {qty} shares at limit {limit_price:.2f} "
            f"(entry {self.entry_price:.2f}), approx PnL={approx_pnl:.2f}"
        )
        order = LimitOrder(action, qty, limit_price, transmit=LIVE_TRADING)
        self.ib.placeOrder(self.contract, order)
        self.stop_order = None
        self.be_triggered = False
        self.position_side = "flat"
        self.high_since_entry = None
        self.low_since_entry = None

    def manage_risk(self) -> None:
        if self.position_side == "flat":
            return
        current_price = self.bars[-1].close
        current_high = self.bars[-1].high
        current_low = self.bars[-1].low
        position_qty = abs(self._current_position_size())
        if position_qty == 0:
            return
        entry = self.entry_price or current_price
        atr_value = self._compute_atr(self.config.atr_length) if self.config.use_atr_risk else None
        atr_ready = atr_value is not None and not math.isnan(atr_value) and atr_value > 0

        stop_price = None
        trail_candidate = None

        if self.position_side == "long":
            self.high_since_entry = max(self.high_since_entry or entry, current_high)
            if self.config.use_atr_risk and atr_ready:
                base_stop = entry - self.config.atr_sl_mult * atr_value if self.config.use_sl else None
                if self.config.use_be and self.high_since_entry > entry + self.config.atr_be_mult * atr_value:
                    self.be_triggered = True
                if self.be_triggered:
                    base_stop = max(base_stop or 0, entry)
                if self.config.use_trail and current_price > entry:
                    trail_candidate = current_price - self.config.atr_trail_mult * atr_value
            else:
                base_stop = entry * (1 - self.config.sl_percent) if self.config.use_sl else None
                if self.config.use_be and self.high_since_entry > entry * (1 + self.config.be_trigger_percent):
                    self.be_triggered = True
                if self.be_triggered:
                    base_stop = max(base_stop or 0, entry)
                if self.config.use_trail and current_price > entry * (1 + self.config.trail_offset):
                    trail_candidate = current_price - entry * self.config.trail_percent
            stop_price = max(v for v in [base_stop, trail_candidate] if v is not None) if any(
                v is not None for v in [base_stop, trail_candidate]
            ) else None
        elif self.position_side == "short":
            self.low_since_entry = min(self.low_since_entry or entry, current_low)
            if self.config.use_atr_risk and atr_ready:
                base_stop = entry + self.config.atr_sl_mult * atr_value if self.config.use_sl else None
                if self.config.use_be and self.low_since_entry < entry - self.config.atr_be_mult * atr_value:
                    self.be_triggered = True
                if self.be_triggered:
                    base_stop = min(base_stop or float("inf"), entry)
                if self.config.use_trail and current_price < entry:
                    trail_candidate = current_price + self.config.atr_trail_mult * atr_value
            else:
                base_stop = entry * (1 + self.config.sl_percent) if self.config.use_sl else None
                if self.config.use_be and self.low_since_entry < entry * (1 - self.config.be_trigger_percent):
                    self.be_triggered = True
                if self.be_triggered:
                    base_stop = min(base_stop or float("inf"), entry)
                if self.config.use_trail and current_price < entry * (1 - self.config.trail_offset):
                    trail_candidate = current_price + entry * self.config.trail_percent
            stop_price = min(v for v in [base_stop, trail_candidate] if v is not None) if any(
                v is not None for v in [base_stop, trail_candidate]
            ) else None

        if stop_price is None:
            return

        stop_price = self._apply_min_tick(stop_price)

        if self.stop_order and abs(self.stop_order.auxPrice - stop_price) / stop_price < 0.001:
            return  # no meaningful change

        # Cancel previous stop if exists
        if self.stop_order:
            self.logger.info("Modifying existing stop order.")
            self.ib.cancelOrder(self.stop_order)

        stop_action = "SELL" if self.position_side == "long" else "BUY"
        stop_order = Order(
            action=stop_action,
            orderType="STP LMT",
            totalQuantity=position_qty,
            auxPrice=stop_price,
            lmtPrice=stop_price,
            transmit=LIVE_TRADING,
        )
        self.stop_order = stop_order
        self.ib.placeOrder(self.contract, stop_order)
        self.logger.info(f"Placed/updated stop at {stop_price:.2f} for {self.position_side} position. Live={LIVE_TRADING}")

    # -------------------------
    # Optimization & backtesting
    # -------------------------
    def run_weekly_optimization(self) -> None:
        """Perform weekly grid search and apply the best configuration."""
        best_config, best_metrics, results = run_weekly_optimization_once(self.config)

        if best_config and best_metrics:
            self.logger.info("Optimization results (top 5):")
            print_top_results(results)

            save_config(best_config)
            if best_config.__dict__ != self.config.__dict__:
                self.logger.info("Best configuration differs from current; applying and sending alert.")
                self._send_push_notification(best_config, best_metrics)
                self.apply_new_config(best_config)
            else:
                self.logger.info("Best configuration matches current settings.")
        else:
            self.logger.info("Optimization did not find a valid configuration.")

    def apply_new_config(self, config: StrategyConfig) -> None:
        """Reload bars and indicators using the optimized configuration."""
        self.config = config
        self.logger.info(
            f"Switching to timeframe {config.timeframe_minutes}m, Fast EMA {config.fast_ema}, "
            f"Slow EMA {config.slow_ema}, ADX {config.adx_length}/{config.adx_threshold}."
        )
        self.bars.clear()
        self.current_bar = None
        self._load_initial_history()

    def _send_push_notification(self, config: StrategyConfig, metrics: Dict[str, float]) -> None:
        """Placeholder for push notification when config changes."""
        self.logger.info(
            "[PUSH] New weekly config -> "
            f"TF={config.timeframe_minutes}m, Fast={config.fast_ema}, Slow={config.slow_ema}, "
            f"ADX={config.adx_length}/{config.adx_threshold}, "
            f"PnL={metrics['net_pnl']:.2f}, DD={metrics['max_drawdown']:.2f}, Sharpe={metrics['sharpe']:.2f}"
        )

    # -------------------------
    # Utilities
    # -------------------------
    def _apply_min_tick(self, price: float) -> float:
        try:
            details = self.ib.reqContractDetails(self.contract)
            min_tick = details[0].minTick if details and details[0].minTick else 0.01
        except Exception:
            min_tick = 0.01
        if min_tick <= 0:
            min_tick = 0.01
        return round(price / min_tick) * min_tick

    def _compute_position_size(self, price: float) -> int:
        account = self.ib.accountSummary()
        net_liq = 0.0
        for row in account:
            if row.tag == "NetLiquidation" and row.currency == CURRENCY:
                net_liq = float(row.value)
                break
        if net_liq <= 0:
            self.logger.warning("Unable to fetch account equity; defaulting quantity to 0.")
            return 0
        value = net_liq * (self.config.position_size_pct / 100)
        qty = int(value // price)
        return max(qty, 0)

    def _refresh_position_state(self) -> None:
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.conId == self.contract.conId or (
                pos.contract.symbol == self.contract.symbol and pos.contract.exchange == self.contract.exchange
            ):
                if pos.position > 0:
                    self.position_side = "long"
                elif pos.position < 0:
                    self.position_side = "short"
                else:
                    self.position_side = "flat"
                self.entry_price = pos.avgCost
                self.high_since_entry = self.entry_price
                self.low_since_entry = self.entry_price
                return
        self.position_side = "flat"

    def _current_position_size(self) -> float:
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.conId == self.contract.conId or (
                pos.contract.symbol == self.contract.symbol and pos.contract.exchange == self.contract.exchange
            ):
                return pos.position
        return 0.0


# -------------------------
# Script entry point 
# -------------------------
def setup_logging(log_path: Path = LOG_PATH) -> None:
    """Configure console + rotating file logging once per process."""
    logger = logging.getLogger()
    if logger.handlers:
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def run_bot() -> None:
    setup_logging()
    config = load_config()
    bot = EmaAdxBot(config)
    try:
        bot.start()
    except KeyboardInterrupt:
        logging.info("Shutting down bot...")
    finally:
        bot.ib.disconnect()


if __name__ == "__main__":
    run_bot()
