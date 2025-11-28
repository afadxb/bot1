"""Streamlit dashboard for running EMA+ADX backtests.

Run with: streamlit run dashboard.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting import backtest_strategy
from config import CURRENCY, EXCHANGE, StrategyConfig, load_config, save_config
from data_cache import fetch_historical_dataframe


def _render_sidebar(default_cfg: StrategyConfig) -> tuple[StrategyConfig, bool, bool]:
    st.sidebar.header("Strategy Parameters")

    symbol = st.sidebar.text_input("Symbol", default_cfg.symbol)
    timeframe_options = [60, 120, 180, 240]
    trade_options = ["Long", "Short", "Both"]
    timeframe_index = timeframe_options.index(default_cfg.timeframe_minutes) if default_cfg.timeframe_minutes in timeframe_options else 0
    trade_index = trade_options.index(default_cfg.trade_direction) if default_cfg.trade_direction in trade_options else 2

    timeframe = st.sidebar.selectbox("Timeframe (minutes)", timeframe_options, index=timeframe_index)
    lookback_days = st.sidebar.number_input("Lookback days", min_value=1, value=default_cfg.lookback_days)
    trade_direction = st.sidebar.selectbox("Trade direction", trade_options, index=trade_index)

    st.sidebar.subheader("EMA Settings")
    ema_fast = st.sidebar.number_input("EMA Fast", min_value=1, value=default_cfg.ema_fast)
    ema_slow = st.sidebar.number_input("EMA Slow", min_value=1, value=default_cfg.ema_slow)

    st.sidebar.subheader("ADX Settings")
    use_adx = st.sidebar.checkbox("Use ADX filter", value=default_cfg.use_adx)
    adx_length = st.sidebar.number_input("ADX Length", min_value=1, value=default_cfg.adx_length)
    adx_threshold = st.sidebar.number_input("ADX Threshold", min_value=0.0, value=float(default_cfg.adx_threshold))

    st.sidebar.subheader("Risk Management")
    use_sl = st.sidebar.checkbox("Use Stop Loss", value=default_cfg.use_sl)
    sl_percent = st.sidebar.number_input("Stop Loss %", min_value=0.0, value=float(default_cfg.sl_percent), format="%.4f")
    use_be = st.sidebar.checkbox("Use Breakeven", value=default_cfg.use_be)
    be_trigger_percent = st.sidebar.number_input(
        "Breakeven Trigger %", min_value=0.0, value=float(default_cfg.be_trigger_percent), format="%.4f"
    )
    use_trail = st.sidebar.checkbox("Use Trailing Stop", value=default_cfg.use_trail)
    trail_offset = st.sidebar.number_input("Trail Offset", min_value=0.0, value=float(default_cfg.trail_offset), format="%.4f")
    trail_percent = st.sidebar.number_input(
        "Trail Percent", min_value=0.0, value=float(default_cfg.trail_percent), format="%.4f"
    )

    st.sidebar.subheader("Position Sizing")
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=0.0, value=float(default_cfg.initial_capital))
    position_size_pct = st.sidebar.number_input(
        "Position Size % of Equity", min_value=0.0, value=float(default_cfg.position_size_pct), format="%.2f"
    )

    persist_checked = st.sidebar.checkbox("Persist to ema_adx_config.json")
    run_clicked = st.sidebar.button("Run backtest")
    save_clicked = st.sidebar.button("Save config")

    cfg = StrategyConfig(
        symbol=symbol,
        timeframe_minutes=int(timeframe),
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        adx_length=int(adx_length),
        adx_threshold=float(adx_threshold),
        use_adx=bool(use_adx),
        use_sl=bool(use_sl),
        sl_percent=float(sl_percent),
        use_be=bool(use_be),
        be_trigger_percent=float(be_trigger_percent),
        use_trail=bool(use_trail),
        trail_offset=float(trail_offset),
        trail_percent=float(trail_percent),
        trade_direction=trade_direction,
        position_size_pct=float(position_size_pct),
        initial_capital=float(initial_capital),
        lookback_days=int(lookback_days),
        risk_params=default_cfg.risk_params.copy() if default_cfg.risk_params else {},
    )

    if save_clicked:
        if persist_checked:
            save_config(cfg)
            st.sidebar.success("Configuration saved to ema_adx_config.json")
        else:
            st.sidebar.info("Enable persistence to save configuration")

    return cfg, run_clicked, persist_checked


def _display_intro(cfg: StrategyConfig) -> None:
    st.header("EMA Cross + ADX Backtest Dashboard")
    st.write(
        "Use the sidebar to adjust parameters and run a backtest using the same logic as the live bot."
    )
    st.subheader("Current Configuration")
    st.json(cfg.__dict__)


def _display_results(metrics: dict, df: pd.DataFrame) -> None:
    st.subheader("Backtest Results")
    start_date = df["time"].iloc[0] if not df.empty else None
    end_date = df["time"].iloc[-1] if not df.empty else None

    net_pnl = metrics.get("net_pnl", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    trades = metrics.get("trades", 0)
    final_equity = net_pnl + metrics.get("initial_capital", 0.0)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Net PnL", f"{net_pnl:,.2f}")
    col2.metric("Max Drawdown", f"{max_dd:,.2f}")
    col3.metric("Sharpe", f"{sharpe:,.2f}")
    col4.metric("Trades", trades)
    col5.metric("Final Equity", f"{final_equity:,.2f}")
    col6.metric("Date Range", f"{start_date} -> {end_date}")

    equity_curve = metrics.get("equity_curve")
    timestamps = metrics.get("timestamps")
    if equity_curve and timestamps:
        equity_series = pd.Series(equity_curve, index=pd.to_datetime(timestamps))
        st.line_chart(equity_series.rename("Equity"))

        drawdown = equity_series - equity_series.cummax()
        st.area_chart(drawdown.rename("Drawdown"))
    else:
        st.info("Equity curve not available from backtest result.")


def main() -> None:
    config = load_config()
    cfg, run_clicked, _ = _render_sidebar(config)

    if not run_clicked:
        _display_intro(cfg)
        return

    st.title("Backtest Output")

    df = fetch_historical_dataframe(
        cfg.timeframe_minutes,
        cfg.lookback_days,
        symbol=cfg.symbol,
        exchange=EXCHANGE,
        currency=CURRENCY,
    )
    if df is None or df.empty:
        st.warning("No historical data available for the requested parameters.")
        return

    with st.spinner("Running backtest..."):
        metrics = backtest_strategy(df, cfg, return_equity=True)
        metrics["initial_capital"] = cfg.initial_capital

    _display_results(metrics, df)


if __name__ == "__main__":
    main()
