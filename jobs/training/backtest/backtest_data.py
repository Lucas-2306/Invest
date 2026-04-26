from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.training_db import engine

try:
    from .backtest_config import BacktestConfig, MODEL_REPORT_PATH, PRICE_TABLE
except ImportError:
    from backtest_config import BacktestConfig, MODEL_REPORT_PATH, PRICE_TABLE


def load_predictions(model_report_path: Path = MODEL_REPORT_PATH) -> pd.DataFrame:
    """Carrega as previsões do modelo e garante as colunas mínimas esperadas."""
    if not model_report_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {model_report_path}. Rode antes o train_model.py."
        )

    df = pd.read_csv(model_report_path, parse_dates=["trade_date"])

    required_cols = {"trade_date", "symbol", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em test_predictions.csv: {sorted(missing)}")

    return df.sort_values(["trade_date", "prediction"], ascending=[True, False]).reset_index(drop=True)

def load_prices(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Busca os preços necessários para todo o intervalo do backtest."""
    symbols_sql = ", ".join([f"'{s}'" for s in sorted(set(symbols))])

    query = f"""
        SELECT
            s.symbol,
            p.trade_date,
            p.open_price,
            p.close_price,
            p.adjusted_close_price
        FROM {PRICE_TABLE} p
        JOIN market_data.symbols s
          ON s.id = p.symbol_id
        WHERE s.symbol IN ({symbols_sql})
          AND p.trade_date >= '{start_date.date()}'
          AND p.trade_date <= '{end_date.date()}'
        ORDER BY s.symbol, p.trade_date
    """

    df = pd.read_sql_query(query, engine, parse_dates=["trade_date"])
    if df.empty:
        raise ValueError("Nenhum preço encontrado para o período do backtest.")
    return df

def apply_liquidity_filter_to_predictions(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Remove ativos sem liquidez mínima, quando as colunas existirem."""
    df = df.copy()

    required_cols = {"avg_daily_volume_20d", "avg_daily_traded_value_20d"}
    if not required_cols.issubset(df.columns):
        return df

    valid_volume = (
        df["avg_daily_volume_20d"].notna()
        & (df["avg_daily_volume_20d"] >= config.min_avg_daily_volume_20d)
    )
    valid_traded_value = (
        df["avg_daily_traded_value_20d"].notna()
        & (df["avg_daily_traded_value_20d"] >= config.min_avg_daily_traded_value_20d)
    )

    return df[valid_volume & valid_traded_value].copy()

def prepare_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Padroniza preços e calcula retorno diário por ativo."""
    df = prices.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    for col in ["open_price", "close_price", "adjusted_close_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ref_price"] = df["adjusted_close_price"].fillna(df["close_price"])
    prev_ref = df.groupby("symbol")["ref_price"].shift(1)
    raw_ret = df["ref_price"] / prev_ref

    valid_ret = (
        prev_ref.notna()
        & (prev_ref > 0)
        & raw_ret.notna()
        & np.isfinite(raw_ret)
        & (raw_ret > 0.05)
        & (raw_ret < 20.0)
    )

    df["daily_return"] = np.nan
    df.loc[valid_ret, "daily_return"] = raw_ret.loc[valid_ret] - 1.0
    return df

def select_rebalance_dates(preds: pd.DataFrame, step: int) -> list[pd.Timestamp]:
    """Seleciona datas de rebalanceamento conforme o passo configurado."""
    unique_dates = sorted(preds["trade_date"].drop_duplicates())
    return unique_dates[::step]
