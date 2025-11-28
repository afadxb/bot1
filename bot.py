"""
EMA Cross + ADX Filter trading bot for Interactive Brokers (IBKR). 

This script recreates the provided TradingView Pine Script strategy using the
ib_insync library. It is intended for educational / paper-trading use. Review
all parameters before enabling LIVE_TRADING.
"""
from __future__ import annotations

import datetime as dt
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

from backtesting import (
    StrategyState,
    compute_dmi_adx,
    compute_ema,
    ema_cross_signals,
    _close_position as sim_close_position,
    _open_position as sim_open_position,
    update_stop_price,
)
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
        self.state = StrategyState(cash=config.initial_capital)
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

    def _sync_open_stop(self) -> None:
        """Populate self.stop_order from existing open stop/stop-limit orders to prevent duplicates."""
        if self.position_side == "flat":
            self.stop_order = None
            return
        try:
            open_trades = self.ib.openTrades()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"Unable to fetch open orders to sync stops: {exc}")
            return
        for trade in open_trades:
            contract = trade.contract
            order = trade.order
            if contract.conId != self.contract.conId and (
                contract.symbol != self.contract.symbol or contract.exchange != self.contract.exchange
            ):
                continue
            if "STP" in (order.orderType or ""):
                if self.stop_order is None or getattr(order, "orderId", 0) > getattr(self.stop_order, "orderId", -1):
                    self.stop_order = order
        if self.stop_order:
            self.logger.info(
                f"Synced existing stop order (id={getattr(self.stop_order, 'orderId', '?')}) at "
                f"{getattr(self.stop_order, 'auxPrice', 0):.2f}"
            )

    @staticmethod
    def _rt_value(bar: RealTimeBar, field: str) -> float:
        """Access RT bar fields consistently across ib_insync versions."""
        val = getattr(bar, field, None)
        if val is None:
            val = getattr(bar, f"{field}_", None)
        if val is None:
            raise AttributeError(f"RealTimeBar missing expected field '{field}'/'{field}_'")
        return val

    @staticmethod
    def _apply_min_tick(price: float, min_tick: float = 0.05) -> float:
        """Round price to the nearest permissible increment."""
        if min_tick <= 0:
            return price
        return round(price / min_tick) * min_tick

    def _load_initial_history(self) -> None:
        """Prefill bars using historical data to warm up indicators."""
        hist_df = get_resampled_bars(self.config.timeframe_minutes, lookback_days=self.config.lookback_days)
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
        if len(self.bars) < max(self.config.ema_fast, self.config.ema_slow, self.config.adx_length + 1):
            return
        df = pd.DataFrame([b.__dict__ for b in self.bars])
        df["fast"] = compute_ema(df["close"], self.config.ema_fast)
        df["slow"] = compute_ema(df["close"], self.config.ema_slow)
        df["fast_prev"] = df["fast"].shift(1)
        df["slow_prev"] = df["slow"].shift(1)
        df["adx"] = compute_dmi_adx(df, self.config.adx_length)

        if len(df) < 2:
            return
        self.indicators.prev_fast_ema = float(df.iloc[-2].fast)
        self.indicators.prev_slow_ema = float(df.iloc[-2].slow)
        self.indicators.fast_ema = float(df.iloc[-1].fast)
        self.indicators.slow_ema = float(df.iloc[-1].slow)
        self.indicators.adx = float(df.iloc[-1].adx)

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

        signals = ema_cross_signals(fast, slow, prev_fast, prev_slow)
        adx_condition = (adx_val > self.config.adx_threshold) if self.config.use_adx else True

        self._refresh_position_state()

        # Stop logic
        if self.state.position_side != 0:
            bar_row = pd.Series({"close": self.bars[-1].close, "high": self.bars[-1].high, "low": self.bars[-1].low})
            stop_price = update_stop_price(self.state, bar_row, self.config)
            if stop_price is not None:
                if (self.state.position_side > 0 and self.bars[-1].low <= stop_price) or (
                    self.state.position_side < 0 and self.bars[-1].high >= stop_price
                ):
                    self.logger.info(f"Stop hit at {stop_price:.2f}; closing position.")
                    self.close_position(exit_price=stop_price)

        ema_crossover = signals["cross_up"]
        ema_crossunder = signals["cross_down"]

        if self.state.position_side > 0 and ema_crossunder:
            self.logger.info("EMA reverse exit triggered for long.")
            self.close_position()
            if self.config.trade_direction in ("Short", "Both") and adx_condition:
                self.enter_position("short")
        elif self.state.position_side < 0 and ema_crossover:
            self.logger.info("EMA reverse exit triggered for short.")
            self.close_position()
            if self.config.trade_direction in ("Long", "Both") and adx_condition:
                self.enter_position("long")
        elif self.state.position_side == 0:
            if (self.config.trade_direction in ("Long", "Both")) and ema_crossover and adx_condition:
                self.enter_position("long")
            elif (self.config.trade_direction in ("Short", "Both")) and ema_crossunder and adx_condition:
                self.enter_position("short")

        if self.state.stop_price is not None:
            self._place_or_update_stop(self.state.stop_price)

    def enter_position(self, side: str) -> None:
        price = self.bars[-1].close
        qty = self._compute_position_size(price)
        if qty <= 0:
            self.logger.info("Computed quantity is 0; skipping entry.")
            return
        action = "BUY" if side == "long" else "SELL"
        limit_price = price
        self.logger.info(f"Placing {side} entry for {qty} shares with limit {limit_price:.2f}.")
        order = LimitOrder(action, qty, limit_price, transmit=LIVE_TRADING)
        self.ib.placeOrder(self.contract, order)
        sim_open_position(self.state, price, 1 if side == "long" else -1, self.config, self._current_equity(price))
        self.position_side = side
        self.entry_price = price
        self.be_triggered = False
        self.stop_order = None
        self.high_since_entry = price
        self.low_since_entry = price
        self.logger.info(f"Entered {side} at ~{price:.2f}. Live={LIVE_TRADING}")

    def close_position(self, exit_price: Optional[float] = None) -> None:
        self._refresh_position_state()
        if self.position_side == "flat":
            return
        action = "SELL" if self.position_side == "long" else "BUY"
        qty = abs(self._current_position_size())
        limit_price = exit_price or self.bars[-1].close
        self.logger.info(
            f"Closing {self.position_side} position of {qty} shares with limit {limit_price:.2f}."
        )
        order = LimitOrder(action, qty, limit_price, transmit=LIVE_TRADING)
        self.ib.placeOrder(self.contract, order)
        sim_close_position(self.state, limit_price)
        self.stop_order = None
        self.be_triggered = False
        self.position_side = "flat"
        self.state.position_qty = 0
        self.state.position_side = 0
        self.high_since_entry = None
        self.low_since_entry = None

    def manage_risk(self) -> None:
        return  # risk handled in on_bar_close

    def _place_or_update_stop(self, stop_price: float) -> None:
        if self.state.position_side == 0:
            return
        stop_price = self._apply_min_tick(stop_price)
        position_qty = abs(self.state.position_qty)
        if position_qty == 0:
            return
        if self.stop_order and abs(self.stop_order.auxPrice - stop_price) / stop_price < 0.001:
            return
        if self.stop_order:
            self.logger.info("Modifying existing stop order.")
            self.ib.cancelOrder(self.stop_order)
        stop_action = "SELL" if self.state.position_side > 0 else "BUY"
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
        self.logger.info(
            f"Placed/updated stop at {stop_price:.2f} for {self.position_side or 'flat'} position. Live={LIVE_TRADING}"
        )

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
            f"Switching to timeframe {config.timeframe_minutes}m, Fast EMA {config.ema_fast}, "
            f"Slow EMA {config.ema_slow}, ADX {config.adx_length}/{config.adx_threshold}."
        )
        self.bars.clear()
        self.current_bar = None
        self._load_initial_history()

    def _send_push_notification(self, config: StrategyConfig, metrics: Dict[str, float]) -> None:
        """Placeholder for push notification when config changes."""
        self.logger.info(
            "[PUSH] New weekly config -> "
            f"TF={config.timeframe_minutes}m, Fast={config.ema_fast}, Slow={config.ema_slow}, "
            f"ADX={config.adx_length}/{config.adx_threshold}, "
            f"PnL={metrics['net_pnl']:.2f}, DD={metrics['max_drawdown']:.2f}, Sharpe={metrics['sharpe']:.2f}"
        )

    # -------------------------
    # Utilities
    # -------------------------
    def _compute_position_size(self, price: float) -> int:
        equity = self._current_equity(price)
        if equity <= 0:
            self.logger.warning("Unable to fetch account equity; defaulting quantity to 0.")
            return 0
        value = equity * (self.config.position_size_pct / 100)
        qty = int(value // price)
        return max(qty, 0)

    def _current_equity(self, last_price: float) -> float:
        account = self.ib.accountSummary()
        net_liq = 0.0
        for row in account:
            if row.tag == "NetLiquidation" and row.currency == CURRENCY:
                net_liq = float(row.value)
                break
        if self.state.position_qty:
            return net_liq if net_liq else self.state.cash + self.state.position_qty * last_price
        return net_liq if net_liq else self.state.cash

    def _refresh_position_state(self) -> None:
        positions = self.ib.positions()
        active = None
        for pos in positions:
            if pos.contract.conId == self.contract.conId or (
                pos.contract.symbol == self.contract.symbol and pos.contract.exchange == self.contract.exchange
            ):
                active = pos
                break

        last_price = self.bars[-1].close if self.bars else 0
        if active:
            qty = int(active.position)
            self.state.position_qty = qty
            self.state.position_side = 1 if qty > 0 else -1 if qty < 0 else 0
            self.state.entry_price = active.avgCost if qty != 0 else None
            self.state.high_since_entry = self.state.entry_price
            self.state.low_since_entry = self.state.entry_price
            equity = self._current_equity(last_price)
            self.state.cash = equity - qty * last_price
            self.position_side = "long" if qty > 0 else "short" if qty < 0 else "flat"
            self.entry_price = active.avgCost
            self._sync_open_stop()
            return

        self.state.position_qty = 0
        self.state.position_side = 0
        self.state.entry_price = None
        self.position_side = "flat"
        self.state.cash = self._current_equity(last_price)
        self.stop_order = None

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
