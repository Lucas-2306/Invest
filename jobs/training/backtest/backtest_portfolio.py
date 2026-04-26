from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .backtest_config import BacktestConfig
except ImportError:
    from backtest_config import BacktestConfig


def normalize_active_lots_exposure(daily_positions: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a exposição quando há vários lotes simultâneos ativos."""
    df = daily_positions.copy()

    if "row_type" not in df.columns:
        df["row_type"] = "market_return"

    market_df = df[df["row_type"] == "market_return"].copy()
    active_lots = (
        market_df.groupby("trade_date")["signal_date"]
        .nunique()
        .rename("num_active_lots")
        .reset_index()
    )

    df = df.merge(active_lots, on="trade_date", how="left")
    df["num_active_lots"] = df["num_active_lots"].fillna(0).astype(int)

    is_market = df["row_type"] == "market_return"
    df["capital_scale"] = 1.0

    valid_market = is_market & (df["num_active_lots"] > 0)
    df.loc[valid_market, "capital_scale"] = 1.0 / df.loc[valid_market, "num_active_lots"]

    df["position_contribution"] = np.where(
        is_market,
        df["weight"] * df["position_return"] * df["capital_scale"],
        df["position_contribution"],
    )
    return df

def apply_transaction_costs(
    daily_positions: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Adiciona linhas sintéticas de custo na entrada e na saída dos trades."""
    daily_positions = daily_positions.copy()

    if "row_type" not in daily_positions.columns:
        daily_positions["row_type"] = "market_return"

    active_lots = (
        daily_positions[daily_positions["row_type"] == "market_return"]
        .groupby("trade_date")["signal_date"]
        .nunique()
        .rename("num_active_lots")
        .reset_index()
    )
    active_lots_map = dict(zip(active_lots["trade_date"], active_lots["num_active_lots"]))

    cost_rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        weight = float(trade["weight"])
        entry_date = pd.to_datetime(trade["entry_date"])
        exit_date = pd.to_datetime(trade["exit_date"])

        entry_active_lots = int(active_lots_map.get(entry_date, 1))
        exit_active_lots = int(active_lots_map.get(exit_date, 1))

        entry_scale = 1.0 / entry_active_lots if entry_active_lots > 0 else 1.0
        exit_scale = 1.0 / exit_active_lots if exit_active_lots > 0 else 1.0

        entry_cost = weight * entry_scale * (config.transaction_cost_per_side + config.slippage)
        exit_cost = weight * exit_scale * config.transaction_cost_per_side

        base_row = {
            "trade_id": int(trade["trade_id"]),
            "symbol": str(trade["symbol"]),
            "side": str(trade["side"]),
            "weight": weight,
            "signal_date": pd.to_datetime(trade["signal_date"]),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "prediction": float(trade["prediction"]),
            "daily_return": np.nan,
            "position_return": 0.0,
            "signed_weight": 0.0,
        }

        cost_rows.append(
            {
                **base_row,
                "trade_date": entry_date,
                "capital_scale": entry_scale,
                "position_contribution": -entry_cost,
                "row_type": "entry_cost",
            }
        )
        cost_rows.append(
            {
                **base_row,
                "trade_date": exit_date,
                "capital_scale": exit_scale,
                "position_contribution": -exit_cost,
                "row_type": "exit_cost",
            }
        )

    cost_df = pd.DataFrame(cost_rows)
    out = pd.concat([daily_positions, cost_df], ignore_index=True)
    return out.sort_values(["trade_date", "trade_id", "row_type"]).reset_index(drop=True)

def compute_turnover(trades: pd.DataFrame) -> pd.DataFrame:
    """Mede o giro entre um rebalanceamento e o próximo."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "num_prev_symbols",
                "num_curr_symbols",
                "kept_symbols",
                "entered_symbols",
                "exited_symbols",
                "turnover_ratio",
            ]
        )

    rows: list[dict[str, Any]] = []
    previous_symbols: set[str] = set()

    for signal_date, group in trades.groupby("signal_date"):
        current_symbols = set(group["symbol"].astype(str).tolist())
        kept = len(previous_symbols & current_symbols)
        entered = len(current_symbols - previous_symbols)
        exited = len(previous_symbols - current_symbols)

        denom = max(len(previous_symbols), len(current_symbols), 1)
        turnover_ratio = (entered + exited) / denom

        rows.append(
            {
                "signal_date": pd.to_datetime(signal_date),
                "num_prev_symbols": int(len(previous_symbols)),
                "num_curr_symbols": int(len(current_symbols)),
                "kept_symbols": int(kept),
                "entered_symbols": int(entered),
                "exited_symbols": int(exited),
                "turnover_ratio": float(turnover_ratio),
            }
        )
        previous_symbols = current_symbols

    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)

def build_daily_equity_curve(
    daily_positions: pd.DataFrame,
    initial_capital: float,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Consolida PnL diário, exposição e curva de patrimônio."""
    df = daily_positions.copy()

    if "row_type" not in df.columns:
        df["row_type"] = "market_return"

    pnl_by_day = (
        df.groupby("trade_date", as_index=False)
        .agg(daily_return=("position_contribution", "sum"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    market_df = df[df["row_type"] == "market_return"].copy()
    market_df["gross_weight_raw"] = market_df["weight"]
    market_df["gross_weight_effective"] = market_df["weight"] * market_df["capital_scale"]
    market_df["net_weight_effective"] = market_df["signed_weight"] * market_df["capital_scale"]
    market_df["long_contribution"] = np.where(market_df["side"] == "long", market_df["position_contribution"], 0.0)
    market_df["short_contribution"] = np.where(market_df["side"] == "short", market_df["position_contribution"], 0.0)

    exposure_by_day = (
        market_df.groupby("trade_date", as_index=False)
        .agg(
            num_active_rows=("trade_id", "count"),
            num_active_trades=("trade_id", "nunique"),
            gross_exposure_raw=("gross_weight_raw", "sum"),
            gross_exposure_effective=("gross_weight_effective", "sum"),
            net_exposure_effective=("net_weight_effective", "sum"),
            long_contribution=("long_contribution", "sum"),
            short_contribution=("short_contribution", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    grouped = pnl_by_day.merge(exposure_by_day, on="trade_date", how="left")
    for col in [
        "num_active_rows",
        "num_active_trades",
        "gross_exposure_raw",
        "gross_exposure_effective",
        "net_exposure_effective",
        "long_contribution",
        "short_contribution",
    ]:
        default = 0 if "num_" in col else 0.0
        grouped[col] = grouped[col].fillna(default)

    grouped["num_active_rows"] = grouped["num_active_rows"].astype(int)
    grouped["num_active_trades"] = grouped["num_active_trades"].astype(int)

    rows: list[dict[str, Any]] = []
    capital = initial_capital
    peak_capital = initial_capital
    portfolio_returns: list[float] = []

    for _, row in grouped.iterrows():
        start_capital = capital
        
        raw_daily_return = float(row["daily_return"])

        # =========================
        # PERFORMANCE SCALING
        # =========================
        recent_returns = portfolio_returns[-20:]

        performance_scale = 1.0

        if config.use_performance_scaling:
            recent_returns = portfolio_returns[-config.performance_lookback:]

            if len(recent_returns) >= config.performance_lookback:
                mean_return = float(np.mean(recent_returns))

                if mean_return < config.performance_hard_threshold:
                    performance_scale = config.performance_hard_scale
                elif mean_return < config.performance_soft_threshold:
                    performance_scale = config.performance_soft_scale

        daily_return = raw_daily_return * performance_scale

        long_contribution = float(row["long_contribution"]) * performance_scale
        short_contribution = float(row["short_contribution"]) * performance_scale

        capital = capital * (1.0 + daily_return)
        peak_capital = max(peak_capital, capital)
        drawdown = (capital / peak_capital) - 1.0

        rows.append(
            {
                "trade_date": pd.to_datetime(row["trade_date"]),
                "start_capital": start_capital,
                "daily_return": daily_return,
                "end_capital": capital,
                "drawdown": drawdown,
                "num_active_rows": int(row["num_active_rows"]),
                "num_active_trades": int(row["num_active_trades"]),
                "gross_exposure_raw": float(row["gross_exposure_raw"]),
                "gross_exposure_effective": float(row["gross_exposure_effective"]),
                "net_exposure_effective": float(row["net_exposure_effective"]),
                "long_contribution": long_contribution,
                "short_contribution": short_contribution,
            }
        )

        portfolio_returns.append(float(daily_return))

    equity_curve = pd.DataFrame(rows)
    if equity_curve.empty:
        raise ValueError("Equity curve vazia.")
    return equity_curve
