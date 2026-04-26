from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .backtest_config import BacktestConfig, StrategyConfig
except ImportError:
    from backtest_config import BacktestConfig, StrategyConfig


def compute_metrics(
    equity_curve: pd.DataFrame,
    config: BacktestConfig,
    strategy: StrategyConfig,
    turnover_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calcula as métricas principais do backtest."""
    total_return = (equity_curve["end_capital"].iloc[-1] / config.initial_capital) - 1.0
    avg_daily_return = equity_curve["daily_return"].mean()
    daily_vol = equity_curve["daily_return"].std()

    sharpe = np.nan
    if pd.notna(daily_vol) and daily_vol > 0:
        sharpe = (avg_daily_return / daily_vol) * np.sqrt(252)

    avg_turnover = turnover_df["turnover_ratio"].mean() if not turnover_df.empty else None
    median_turnover = turnover_df["turnover_ratio"].median() if not turnover_df.empty else None
    max_turnover = turnover_df["turnover_ratio"].max() if not turnover_df.empty else None

    return {
        "strategy": config.backtest_strategy,
        "portfolio_mode": config.portfolio_mode,
        "initial_capital": float(config.initial_capital),
        "final_capital": float(equity_curve["end_capital"].iloc[-1]),
        "total_return": float(total_return),
        "avg_daily_return": float(avg_daily_return),
        "daily_volatility": float(daily_vol) if pd.notna(daily_vol) else None,
        "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "max_drawdown": float(equity_curve["drawdown"].min()),
        "positive_day_ratio": float((equity_curve["daily_return"] > 0).mean()),
        "num_days": int(len(equity_curve)),
        "top_n_longs_entry": int(config.top_n),
        "top_n_longs_exit": int(config.top_n_exit),
        "top_n_shorts_entry": int(config.bottom_n),
        "top_n_shorts_exit": int(config.bottom_n_exit),
        "transaction_cost_per_side": float(config.transaction_cost_per_side),
        "slippage": float(config.slippage),
        "holding_days": int(config.holding_days),
        "rebalance_every_n_days": int(strategy.rebalance_every_n_days),
        "normalize_active_lots": bool(strategy.normalize_active_lots),
        "long_avg_daily_return": float(equity_curve["long_contribution"].mean()),
        "short_avg_daily_return": float(equity_curve["short_contribution"].mean()),
        "long_total_contribution": float(equity_curve["long_contribution"].sum()),
        "short_total_contribution": float(equity_curve["short_contribution"].sum()),
        "avg_gross_exposure_effective": float(equity_curve["gross_exposure_effective"].mean()),
        "avg_net_exposure_effective": float(equity_curve["net_exposure_effective"].mean()),
        "avg_turnover": float(avg_turnover) if avg_turnover is not None and pd.notna(avg_turnover) else None,
        "median_turnover": float(median_turnover) if median_turnover is not None and pd.notna(median_turnover) else None,
        "max_turnover": float(max_turnover) if max_turnover is not None and pd.notna(max_turnover) else None,
    }

def analyze_by_year(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Monta um resumo anual para inspeção visual."""
    df = equity_curve.copy()
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year

    yearly = df.groupby("year").agg(
        total_return=("end_capital", lambda x: x.iloc[-1] / x.iloc[0] - 1),
        avg_return=("daily_return", "mean"),
        vol=("daily_return", "std"),
        win_rate=("daily_return", lambda x: (x > 0).mean()),
        long_avg=("long_contribution", "mean"),
        short_avg=("short_contribution", "mean"),
    )
    yearly["sharpe"] = yearly["avg_return"] / yearly["vol"] * np.sqrt(252)
    return yearly
