from __future__ import annotations

"""
Compatibility entrypoint for the refactored backtest package.

This file intentionally keeps the original public API available. Other files can
continue importing from `backtest.py`, and running this file still executes the
same pipeline.
"""

try:
    from .backtest_config import (
        BASE_DIR,
        BACKTEST_DIR,
        MODEL_REPORT_PATH,
        PRICE_TABLE,
        REPORT_DIR,
        BacktestConfig,
        StrategyConfig,
        get_strategy_config,
        parse_args,
        validate_config,
    )
    from .backtest_data import (
        apply_liquidity_filter_to_predictions,
        load_predictions,
        load_prices,
        prepare_price_panel,
        select_rebalance_dates,
    )
    from .backtest_metrics import analyze_by_year, compute_metrics
    from .backtest_portfolio import (
        apply_transaction_costs,
        build_daily_equity_curve,
        compute_turnover,
        normalize_active_lots_exposure,
    )
    from .backtest_runner import (
        main,
        print_cost_sensitivity_table,
        print_summary,
        print_yearly_analysis,
        run_cost_sensitivity,
        run_single_backtest,
        save_outputs,
    )
    from .backtest_signals import (
        apply_macro_regime_filter,
        compute_signal_strength,
        select_positions,
        should_trade_by_signal_history,
        weight_positions,
    )
    from .backtest_trades import (
        build_trade_book,
        expand_trades_to_daily_positions,
        get_calendar,
        get_trade_window,
    )
except ImportError:
    from backtest_config import (
        BASE_DIR,
        BACKTEST_DIR,
        MODEL_REPORT_PATH,
        PRICE_TABLE,
        REPORT_DIR,
        BacktestConfig,
        StrategyConfig,
        get_strategy_config,
        parse_args,
        validate_config,
    )
    from backtest_data import (
        apply_liquidity_filter_to_predictions,
        load_predictions,
        load_prices,
        prepare_price_panel,
        select_rebalance_dates,
    )
    from backtest_metrics import analyze_by_year, compute_metrics
    from backtest_portfolio import (
        apply_transaction_costs,
        build_daily_equity_curve,
        compute_turnover,
        normalize_active_lots_exposure,
    )
    from backtest_runner import (
        main,
        print_cost_sensitivity_table,
        print_summary,
        print_yearly_analysis,
        run_cost_sensitivity,
        run_single_backtest,
        save_outputs,
    )
    from backtest_signals import (
        apply_macro_regime_filter,
        compute_signal_strength,
        select_positions,
        should_trade_by_signal_history,
        weight_positions,
    )
    from backtest_trades import (
        build_trade_book,
        expand_trades_to_daily_positions,
        get_calendar,
        get_trade_window,
    )


if __name__ == "__main__":
    main()
