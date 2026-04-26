from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

try:
    from .backtest_config import BacktestConfig, StrategyConfig
    from .backtest_data import apply_liquidity_filter_to_predictions, select_rebalance_dates
    from .backtest_signals import compute_signal_strength, select_positions
except ImportError:
    from backtest_config import BacktestConfig, StrategyConfig
    from backtest_data import apply_liquidity_filter_to_predictions, select_rebalance_dates
    from backtest_signals import compute_signal_strength, select_positions


def get_calendar(prices: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(prices["trade_date"]).drop_duplicates())

def get_trade_window(
    signal_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    holding_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]] | None:
    """Traduz uma data de sinal em entrada, saída e dias ativos de retorno."""
    try:
        signal_idx = calendar.index(signal_date)
    except ValueError:
        return None

    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 1 + holding_days

    if exit_idx >= len(calendar):
        return None

    entry_date = calendar[entry_idx]
    exit_date = calendar[exit_idx]
    active_return_dates = calendar[entry_idx + 1 : exit_idx + 1]

    if len(active_return_dates) != holding_days:
        return None

    return entry_date, exit_date, active_return_dates

def build_trade_book(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    """Gera o livro de trades do backtest."""
    rebalance_dates = select_rebalance_dates(preds, step=strategy.rebalance_every_n_days)
    calendar = get_calendar(prices)

    historical_signal_strengths: list[float] = []
    historical_ics: list[float] = []
    portfolio_returns: list[float] = []
    trade_rows: list[dict[str, Any]] = []
    trade_id = 0
    prev_long_symbols: set[str] = set()
    prev_short_symbols: set[str] = set()

    for signal_date in rebalance_dates:
        window = get_trade_window(signal_date=signal_date, calendar=calendar, holding_days=config.holding_days)
        if window is None:
            continue

        entry_date, exit_date, active_return_dates = window
        day_df = preds[preds["trade_date"] == signal_date].copy()

        filtered_day_df = apply_liquidity_filter_to_predictions(day_df, config)
        filtered_day_df = filtered_day_df.sort_values("prediction", ascending=False).copy()

        if filtered_day_df.empty:
            prev_long_symbols = set()
            prev_short_symbols = set()
            continue

        current_signal_strength = compute_signal_strength(filtered_day_df)

        positions = select_positions(
            day_df=day_df,
            config=config,
            prev_long_symbols=prev_long_symbols,
            prev_short_symbols=prev_short_symbols,
            historical_signal_strengths=historical_signal_strengths,
            historical_ics=historical_ics,
            portfolio_returns=portfolio_returns,
        )

        # =========================
        # CALCULAR IC DO DIA
        # =========================
        ic_target_col = "target_exec_t1"

        if ic_target_col in filtered_day_df.columns:
            ic = filtered_day_df["prediction"].corr(
                filtered_day_df[ic_target_col],
                method="spearman",
            )
            if pd.notna(ic):
                historical_ics.append(float(ic))

        if pd.notna(current_signal_strength):
            historical_signal_strengths.append(float(current_signal_strength))
            

        if positions.empty:
            prev_long_symbols = set()
            prev_short_symbols = set()
            continue

        prev_long_symbols = set(positions.loc[positions["side"] == "long", "symbol"].astype(str))
        prev_short_symbols = set(positions.loc[positions["side"] == "short", "symbol"].astype(str))

        for _, row in positions.iterrows():
            trade_rows.append(
                {
                    "trade_id": trade_id,
                    "signal_date": pd.to_datetime(signal_date),
                    "entry_date": pd.to_datetime(entry_date),
                    "exit_date": pd.to_datetime(exit_date),
                    "symbol": str(row["symbol"]),
                    "side": str(row["side"]),
                    "weight": float(row["weight"]),
                    "prediction": float(row["prediction"]),
                    "active_return_dates": json.dumps([d.strftime("%Y-%m-%d") for d in active_return_dates]),
                }
            )
            trade_id += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError("Nenhum trade foi gerado no backtest.")
    return trades

def expand_trades_to_daily_positions(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Expande cada trade em posições diárias com contribuição de retorno."""
    price_panel = prices[["symbol", "trade_date", "daily_return"]].copy()
    expanded_rows: list[dict[str, Any]] = []

    for _, trade in trades.iterrows():
        active_dates = json.loads(trade["active_return_dates"])
        for dt in active_dates:
            expanded_rows.append(
                {
                    "trade_id": int(trade["trade_id"]),
                    "trade_date": pd.to_datetime(dt),
                    "symbol": str(trade["symbol"]),
                    "side": str(trade["side"]),
                    "weight": float(trade["weight"]),
                    "signal_date": pd.to_datetime(trade["signal_date"]),
                    "entry_date": pd.to_datetime(trade["entry_date"]),
                    "exit_date": pd.to_datetime(trade["exit_date"]),
                    "prediction": float(trade["prediction"]),
                    "row_type": "market_return",
                }
            )

    daily_positions = pd.DataFrame(expanded_rows)
    if daily_positions.empty:
        raise ValueError("Nenhuma posição diária foi expandida.")

    daily_positions = daily_positions.merge(price_panel, on=["symbol", "trade_date"], how="left")
    daily_positions = daily_positions.dropna(subset=["daily_return"]).copy()

    daily_positions["position_return"] = np.where(
        daily_positions["side"] == "long",
        daily_positions["daily_return"],
        -daily_positions["daily_return"],
    )
    daily_positions["capital_scale"] = 1.0
    daily_positions["position_contribution"] = daily_positions["weight"] * daily_positions["position_return"]
    daily_positions["signed_weight"] = np.where(
        daily_positions["side"] == "long",
        daily_positions["weight"],
        -daily_positions["weight"],
    )
    return daily_positions
