# EMA Cross + ADX Strategy Toolkit

This repository mirrors the TradingView strategy **"EMA Cross + ADX Filter & Reverse Exit"** in Python for backtesting, optimization, and live (or paper) trading with Interactive Brokers. The same indicator calculations and trade rules are shared across the backtester, optimizer, and live bot to keep results aligned with TradingView.

## Key Features
- EMA cross entries (fast vs. slow) with optional ADX filter using Pine-style DMI/ADX.
- Reverse-on-cross exits to immediately flip positions when the opposing cross occurs.
- Percent-of-equity position sizing (default 10% of equity per trade).
- Optional stop loss, breakeven trigger, and trailing stop handled bar-by-bar.
- Long/short/both direction control with configurable lookback windows to mirror TradingView history.

## Configuration
Strategy parameters are centralized in [`config.py`](config.py) via `StrategyConfig` and persisted in [`ema_adx_config.json`](ema_adx_config.json). Important fields include:

- `symbol`, `timeframe_minutes` (default 180 for 3h bars)
- `ema_fast`, `ema_slow`, `adx_length`, `adx_threshold`, `use_adx`
- Risk controls: `use_sl`, `sl_percent`, `use_be`, `be_trigger_percent`, `use_trail`, `trail_offset`, `trail_percent`
- `trade_direction` ("Long", "Short", or "Both")
- `position_size_pct` (percent of equity per trade), `initial_capital`
- `lookback_days` (defaults to a long window to match TradingView reports)

Update `ema_adx_config.json` to change live/backtest parameters; `load_config()` reads it everywhere.

## Data Handling
Historical data is fetched from IBKR and cached in `data_cache.sqlite` via [`data_cache.py`](data_cache.py). Use `lookback_days` in your config to ensure the history length matches the TradingView report you are comparing against.

## Backtesting
Run a quick parity test with default Pine parameters:

```bash
python backtesting.py
```

`backtest_strategy` applies the shared EMA/ADX logic, percent-of-equity sizing, and stop/breakeven/trailing rules. The test prints net PnL, drawdown, Sharpe, trades, and the data window.

## Weekly Optimization
Execute a grid search (including Pine defaults) or a single-parameter parity run:

```bash
python weekly_optimization.py          # grid search
python weekly_optimization.py --parity # one-run parity check
```

Results reuse the same backtest engine and history window defined in your config.

## Live/Paper Trading Bot
[`bot.py`](bot.py) connects to IBKR and reuses the shared strategy logic for live or paper trading. Ensure your config and IBKR connection details in `config.py` are correct before enabling `LIVE_TRADING = True`.

## Notes
- Indicator warmup: `MIN_HISTORY_BARS` in `config.py` determines when the bot begins acting on signals after boot.
- Timezones: timestamps are normalized to naive UTC throughout the data pipeline for consistency.
- Always validate against TradingView after parameter changes using the parity test mode before running live.
