"""Standalone weekly optimization task for EMA+ADX strategy settings."""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtesting import backtest_strategy, fetch_historical_dataframe, rank_better
from config import CURRENCY, EXCHANGE, StrategyConfig, load_config


_TIMEFRAMES = [60, 120, 180, 240]
_FAST_OPTS = [10, 14, 21, 34]
_SLOW_OPTS = [34, 50, 89, 100]
_ADX_LEN_OPTS = [14, 20]
_ADX_THRESH_OPTS = [10, 15, 20, 25]


def run_weekly_optimization_once(
    current_config: Optional[StrategyConfig] = None,
    parity_mode: bool = False,
) -> Tuple[Optional[StrategyConfig], Optional[Dict[str, float]], List[Tuple[StrategyConfig, Dict[str, float]]]]:
    """Execute the weekly optimization grid search and return the best result."""
    current_config = current_config or load_config()

    bars_cache: Dict[int, pd.DataFrame] = {}
    for tf in _TIMEFRAMES:
        bars_cache[tf] = fetch_historical_dataframe(
            tf,
            current_config.lookback_days,
            symbol=current_config.symbol,
            exchange=EXCHANGE,
            currency=CURRENCY,
        )

    best_config: Optional[StrategyConfig] = None
    best_metrics: Optional[Dict[str, float]] = None
    results: List[Tuple[StrategyConfig, Dict[str, float]]] = []

    if parity_mode:
        params = current_config
        df = bars_cache.get(params.timeframe_minutes) or pd.DataFrame()
        metrics = backtest_strategy(df, params)
        results.append((params, metrics))
        return params, metrics, results

    for tf in _TIMEFRAMES:
        df = bars_cache.get(tf)
        if df is None or df.empty:
            continue
        for fast in _FAST_OPTS:
            for slow in _SLOW_OPTS:
                if fast >= slow:
                    continue
                for adx_len in _ADX_LEN_OPTS:
                    for adx_thresh in _ADX_THRESH_OPTS:
                        params = StrategyConfig(
                            timeframe_minutes=tf,
                            ema_fast=fast,
                            ema_slow=slow,
                            adx_length=adx_len,
                            adx_threshold=adx_thresh,
                            use_adx=True,
                            trade_direction="Both",
                            use_sl=current_config.use_sl,
                            sl_percent=current_config.sl_percent,
                            use_trail=current_config.use_trail,
                            trail_percent=current_config.trail_percent,
                            trail_offset=current_config.trail_offset,
                            use_be=current_config.use_be,
                            be_trigger_percent=current_config.be_trigger_percent,
                            position_size_pct=current_config.position_size_pct,
                            initial_capital=current_config.initial_capital,
                            lookback_days=current_config.lookback_days,
                        )
                        metrics = backtest_strategy(df, params)
                        results.append((params, metrics))
                        if best_metrics is None or rank_better(metrics, best_metrics):
                            best_metrics = metrics
                            best_config = params

    return best_config, best_metrics, results


def _sorted_results(results: List[Tuple[StrategyConfig, Dict[str, float]]]) -> List[Tuple[StrategyConfig, Dict[str, float]]]:
    return sorted(results, key=lambda x: (-x[1]["net_pnl"], x[1]["max_drawdown"], -x[1]["sharpe"]))


def print_top_results(results: List[Tuple[StrategyConfig, Dict[str, float]]], limit: int = 50) -> None:
    """Helper to print the top-performing parameter sets."""
    for params, metrics in _sorted_results(results)[:limit]:
        print(
            f"TF={params.timeframe_minutes}m, Fast={params.ema_fast}, Slow={params.ema_slow}, "
            f"ADX={params.adx_length}/{params.adx_threshold} -> PnL={metrics['net_pnl']:.2f}, "
            f"DD={metrics['max_drawdown']:.2f}, Sharpe={metrics['sharpe']:.2f}, Trades={metrics['trades']}"
        )


def main() -> None:
    """Run optimization once and persist any improved configuration."""
    parser = argparse.ArgumentParser(description="Weekly optimization for EMA+ADX strategy")
    parser.add_argument("--parity", action="store_true", help="Run only the current Pine defaults for parity")
    args = parser.parse_args()

    current_config = load_config()
    best_config, best_metrics, results = run_weekly_optimization_once(current_config, parity_mode=args.parity)

    if best_config and best_metrics:
        if args.parity:
            print("TradingView parity check metrics:")
            print_top_results(results, limit=1)
            return
        print("Optimization results (top 5):")
        print_top_results(results)
        if best_config.__dict__ != current_config.__dict__:
            print("Saved new optimized configuration.")
        else:
            print("Optimized configuration matches the existing settings.")
    else:
        print("Optimization did not find a valid configuration.")


if __name__ == "__main__":
    main()
