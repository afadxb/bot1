"""Standalone weekly optimization task for EMA+ADX strategy settings."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtesting import backtest_strategy, fetch_historical_dataframe, rank_better
from config import StrategyConfig, load_config, save_config


_TIMEFRAMES = [60, 120, 180, 240]
_FAST_OPTS = [10, 14, 21, 34]
_SLOW_OPTS = [34, 50, 89, 100]
_ADX_LEN_OPTS = [14, 20]
_ADX_THRESH_OPTS = [10, 15, 20, 25]


def run_weekly_optimization_once(
    current_config: Optional[StrategyConfig] = None,
) -> Tuple[Optional[StrategyConfig], Optional[Dict[str, float]], List[Tuple[StrategyConfig, Dict[str, float]]]]:
    """Execute the weekly optimization grid search and return the best result."""
    current_config = current_config or load_config()

    bars_cache: Dict[int, pd.DataFrame] = {}
    for tf in _TIMEFRAMES:
        bars_cache[tf] = fetch_historical_dataframe(tf)

    best_config: Optional[StrategyConfig] = None
    best_metrics: Optional[Dict[str, float]] = None
    results: List[Tuple[StrategyConfig, Dict[str, float]]] = []

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
                            fast_ema=fast,
                            slow_ema=slow,
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
            f"TF={params.timeframe_minutes}m, Fast={params.fast_ema}, Slow={params.slow_ema}, "
            f"ADX={params.adx_length}/{params.adx_threshold} -> PnL={metrics['net_pnl']:.2f}, "
            f"DD={metrics['max_drawdown']:.2f}, Sharpe={metrics['sharpe']:.2f}, Trades={metrics['trades']}"
        )


def main() -> None:
    """Run optimization once and persist any improved configuration."""
    current_config = load_config()
    best_config, best_metrics, results = run_weekly_optimization_once(current_config)

    if best_config and best_metrics:
        print("Optimization results (top 5):")
        print_top_results(results)
        # save_config(best_config)
        if best_config.__dict__ != current_config.__dict__:
            print("Saved new optimized configuration.")
        else:
            print("Optimized configuration matches the existing settings.")
    else:
        print("Optimization did not find a valid configuration.")


if __name__ == "__main__":
    main()
