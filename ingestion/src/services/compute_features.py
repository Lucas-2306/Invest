import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from repositories.feature_repository import FeatureRepository
from repositories.price_repository import PriceRepository
from repositories.symbol_repository import SymbolRepository

logger = logging.getLogger(__name__)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()

    numeric_cols = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close_price",
        "volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ref_price"] = df["adjusted_close_price"].fillna(df["close_price"])

    ratio_1d = df["ref_price"] / df["ref_price"].shift(1)
    valid_ratio_1d = (
        ratio_1d.notna()
        & np.isfinite(ratio_1d)
        & (ratio_1d > 0.2)
        & (ratio_1d < 5.0)
    )

    df["return_1d"] = np.nan
    df.loc[valid_ratio_1d, "return_1d"] = ratio_1d.loc[valid_ratio_1d] - 1.0

    df["log_return_1d"] = np.nan
    df.loc[valid_ratio_1d, "log_return_1d"] = np.log(ratio_1d.loc[valid_ratio_1d])

    ratio_5d = df["ref_price"] / df["ref_price"].shift(5)
    valid_ratio_5d = (
        ratio_5d.notna()
        & np.isfinite(ratio_5d)
        & (ratio_5d > 0.1)
        & (ratio_5d < 10.0)
    )

    df["return_5d"] = np.nan
    df.loc[valid_ratio_5d, "return_5d"] = ratio_5d.loc[valid_ratio_5d] - 1.0

    ratio_21d = df["ref_price"] / df["ref_price"].shift(21)
    valid_ratio_21d = (
        ratio_21d.notna()
        & np.isfinite(ratio_21d)
        & (ratio_21d > 0.05)
        & (ratio_21d < 20.0)
    )
    df["return_21d"] = np.nan
    df.loc[valid_ratio_21d, "return_21d"] = ratio_21d.loc[valid_ratio_21d] - 1.0

    df["sma_20"] = df["ref_price"].rolling(20, min_periods=20).mean()
    df["sma_50"] = df["ref_price"].rolling(50, min_periods=50).mean()
    df["price_sma_20_ratio"] = df["ref_price"] / df["sma_20"]

    df["volatility_21d"] = df["return_1d"].rolling(21, min_periods=21).std()

    volume_avg_20d = df["volume"].rolling(20, min_periods=20).mean()
    df["volume_ratio_20d"] = df["volume"] / volume_avg_20d

    valid_close = df["close_price"].notna() & (df["close_price"] > 0)
    df["high_low_ratio"] = np.nan
    df.loc[valid_close, "high_low_ratio"] = (
        (df.loc[valid_close, "high_price"] - df.loc[valid_close, "low_price"])
        / df.loc[valid_close, "close_price"]
    )

    prev_close = df["close_price"].shift(1)
    raw_gap_ratio = df["open_price"] / prev_close
    valid_gap = (
        prev_close.notna()
        & (prev_close > 0)
        & raw_gap_ratio.notna()
        & np.isfinite(raw_gap_ratio)
        & (raw_gap_ratio > 0.2)
        & (raw_gap_ratio < 20.0)
    )
    df["gap"] = np.nan
    df.loc[valid_gap, "gap"] = raw_gap_ratio.loc[valid_gap] - 1.0

    future_ratio_5d = df["ref_price"].shift(-5) / df["ref_price"]
    valid_target_5d = (
        future_ratio_5d.notna()
        & np.isfinite(future_ratio_5d)
        & (future_ratio_5d > 0.05)
        & (future_ratio_5d < 20.0)
    )
    df["target_5d"] = np.nan
    df.loc[valid_target_5d, "target_5d"] = future_ratio_5d.loc[valid_target_5d] - 1.0

    df = df.drop(columns=["ref_price"])
    df = df.dropna(subset=["return_1d"])

    return df


def run(session, overlap_days: int = 60) -> None:
    symbol_repo = SymbolRepository(session)
    price_repo = PriceRepository(session)
    feature_repo = FeatureRepository(session)

    symbol_ids = symbol_repo.list_symbol_ids()
    symbols = sorted(symbol_ids.keys())

    logger.info("Iniciando cálculo de features para %s símbolos", len(symbols))

    processed = 0

    for symbol in symbols:
        try:
            symbol_id = symbol_ids[symbol]
            last_feature_date = feature_repo.get_last_feature_date(symbol_id)

            if last_feature_date is not None:
                start_date = last_feature_date - timedelta(days=overlap_days)
            else:
                start_date = None

            df = price_repo.get_prices_as_dataframe(symbol_id, start_date=start_date)

            if df.empty:
                continue

            df = compute(df)

            if df.empty:
                continue

            feature_repo.upsert_dataframe(symbol_id, df)

            session.commit()
            processed += 1

            if processed % 25 == 0:
                logger.info("Features calculadas para %s/%s símbolos", processed, len(symbols))

        except Exception as exc:
            session.rollback()
            logger.exception("Erro ao calcular features para %s: %s", symbol, exc)

    logger.info(
        "Cálculo de features finalizado. Processados %s/%s símbolos",
        processed,
        len(symbols),
    )