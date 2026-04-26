from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from apps.ingestion.providers.yahoo_provider import YahooProvider

logger = logging.getLogger(__name__)


MARKET_FEATURE_TABLE = "macro_data.market_features"

# Yahoo Finance tickers
IBOV_SYMBOL = "BOVA11"
SP500_SYMBOL = "IVVB11"

# Banco Central SGS
# 11 = Selic diária
# 433 = IPCA mensal
BCB_SELIC_CODE = 11
BCB_IPCA_CODE = 433


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def fetch_yahoo_index_history(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    provider = YahooProvider()

    history = provider.get_daily_history(
        symbol=symbol,
        interval="1d",
        start_date=start_date,
        end_date=end_date,
    )

    if not history:
        return pd.DataFrame(columns=["trade_date", "close"])

    rows = []
    for item in history:
        close_price = item.get("adjusted_close_price") or item.get("close_price")
        rows.append(
            {
                "trade_date": pd.to_datetime(item["trade_date"]),
                "close": close_price,
            }
        )

    df = pd.DataFrame(rows)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["trade_date", "close"])
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    return df


def fetch_bcb_series(
    series_code: int,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"

    params = {
        "formato": "json",
        "dataInicial": start_date.strftime("%d/%m/%Y"),
        "dataFinal": end_date.strftime("%d/%m/%Y"),
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    if not data:
        return pd.DataFrame(columns=["trade_date", "value"])

    df = pd.DataFrame(data)
    df["trade_date"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    df["value"] = (
        df["valor"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df = df[["trade_date", "value"]].dropna()
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    return df


def add_index_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()

    close_col = f"{prefix}_close"
    df = df.rename(columns={"close": close_col})

    df[f"{prefix}_return_1d"] = df[close_col].pct_change(1)
    df[f"{prefix}_return_5d"] = df[close_col].pct_change(5)
    df[f"{prefix}_return_21d"] = df[close_col].pct_change(21)
    df[f"{prefix}_return_63d"] = df[close_col].pct_change(63)

    df[f"{prefix}_vol_21d"] = df[f"{prefix}_return_1d"].rolling(21, min_periods=21).std()
    df[f"{prefix}_vol_63d"] = df[f"{prefix}_return_1d"].rolling(63, min_periods=63).std()

    sma_200 = df[close_col].rolling(200, min_periods=200).mean()
    df[f"{prefix}_above_sma_200"] = np.where(
        sma_200.notna(),
        (df[close_col] > sma_200).astype(float),
        np.nan,
    )

    return df


def prepare_selic_features(selic_df: pd.DataFrame) -> pd.DataFrame:
    df = selic_df.copy()
    df = df.rename(columns={"value": "selic_rate"})

    df["selic_change_21d"] = df["selic_rate"] - df["selic_rate"].shift(21)
    df["selic_change_63d"] = df["selic_rate"] - df["selic_rate"].shift(63)

    return df


def prepare_ipca_features(ipca_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    IPCA é mensal. Para usar em datas diárias, fazemos forward-fill
    para o calendário de pregões.
    """
    monthly = ipca_df.copy()
    monthly = monthly.rename(columns={"value": "ipca_monthly"})

    monthly["ipca_3m"] = monthly["ipca_monthly"].rolling(3, min_periods=3).sum()
    monthly["ipca_6m"] = monthly["ipca_monthly"].rolling(6, min_periods=6).sum()
    monthly["ipca_12m"] = monthly["ipca_monthly"].rolling(12, min_periods=12).sum()

    out = calendar[["trade_date"]].copy().sort_values("trade_date")

    out = pd.merge_asof(
        out,
        monthly.sort_values("trade_date"),
        on="trade_date",
        direction="backward",
    )

    out["ipca_change_3m"] = out["ipca_3m"] - out["ipca_3m"].shift(63)
    out["ipca_change_6m"] = out["ipca_6m"] - out["ipca_6m"].shift(126)

    return out


def build_market_features(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    ibov = fetch_yahoo_index_history(IBOV_SYMBOL, start_date, end_date)
    sp500 = fetch_yahoo_index_history(SP500_SYMBOL, start_date, end_date)

    if ibov.empty:
        raise ValueError("Histórico do Ibovespa vazio.")
    if sp500.empty:
        raise ValueError("Histórico do S&P 500 vazio.")

    ibov_features = add_index_features(ibov, "ibov")
    sp500_features = add_index_features(sp500, "sp500")

    calendar = ibov_features[["trade_date"]].copy()

    selic = fetch_bcb_series(BCB_SELIC_CODE, start_date, end_date)
    ipca = fetch_bcb_series(BCB_IPCA_CODE, start_date, end_date)

    selic_features = prepare_selic_features(selic)
    ipca_features = prepare_ipca_features(ipca, calendar)

    df = calendar.copy()

    df = df.merge(ibov_features, on="trade_date", how="left")
    df = df.merge(sp500_features, on="trade_date", how="left")

    df = pd.merge_asof(
        df.sort_values("trade_date"),
        selic_features.sort_values("trade_date"),
        on="trade_date",
        direction="backward",
    )

    df = df.merge(ipca_features, on="trade_date", how="left")

    numeric_cols = [col for col in df.columns if col != "trade_date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("trade_date").reset_index(drop=True)


def upsert_market_features(session, df: pd.DataFrame) -> None:
    sql = text(f"""
        INSERT INTO {MARKET_FEATURE_TABLE} (
            trade_date,

            ibov_close,
            ibov_return_1d,
            ibov_return_5d,
            ibov_return_21d,
            ibov_return_63d,
            ibov_vol_21d,
            ibov_vol_63d,
            ibov_above_sma_200,

            sp500_close,
            sp500_return_1d,
            sp500_return_5d,
            sp500_return_21d,
            sp500_return_63d,
            sp500_vol_21d,
            sp500_vol_63d,
            sp500_above_sma_200,

            selic_rate,
            selic_change_21d,
            selic_change_63d,

            ipca_monthly,
            ipca_3m,
            ipca_6m,
            ipca_12m,
            ipca_change_3m,
            ipca_change_6m,

            updated_at
        ) VALUES (
            :trade_date,

            :ibov_close,
            :ibov_return_1d,
            :ibov_return_5d,
            :ibov_return_21d,
            :ibov_return_63d,
            :ibov_vol_21d,
            :ibov_vol_63d,
            :ibov_above_sma_200,

            :sp500_close,
            :sp500_return_1d,
            :sp500_return_5d,
            :sp500_return_21d,
            :sp500_return_63d,
            :sp500_vol_21d,
            :sp500_vol_63d,
            :sp500_above_sma_200,

            :selic_rate,
            :selic_change_21d,
            :selic_change_63d,

            :ipca_monthly,
            :ipca_3m,
            :ipca_6m,
            :ipca_12m,
            :ipca_change_3m,
            :ipca_change_6m,

            NOW()
        )
        ON CONFLICT (trade_date)
        DO UPDATE SET
            ibov_close = EXCLUDED.ibov_close,
            ibov_return_1d = EXCLUDED.ibov_return_1d,
            ibov_return_5d = EXCLUDED.ibov_return_5d,
            ibov_return_21d = EXCLUDED.ibov_return_21d,
            ibov_return_63d = EXCLUDED.ibov_return_63d,
            ibov_vol_21d = EXCLUDED.ibov_vol_21d,
            ibov_vol_63d = EXCLUDED.ibov_vol_63d,
            ibov_above_sma_200 = EXCLUDED.ibov_above_sma_200,

            sp500_close = EXCLUDED.sp500_close,
            sp500_return_1d = EXCLUDED.sp500_return_1d,
            sp500_return_5d = EXCLUDED.sp500_return_5d,
            sp500_return_21d = EXCLUDED.sp500_return_21d,
            sp500_return_63d = EXCLUDED.sp500_return_63d,
            sp500_vol_21d = EXCLUDED.sp500_vol_21d,
            sp500_vol_63d = EXCLUDED.sp500_vol_63d,
            sp500_above_sma_200 = EXCLUDED.sp500_above_sma_200,

            selic_rate = EXCLUDED.selic_rate,
            selic_change_21d = EXCLUDED.selic_change_21d,
            selic_change_63d = EXCLUDED.selic_change_63d,

            ipca_monthly = EXCLUDED.ipca_monthly,
            ipca_3m = EXCLUDED.ipca_3m,
            ipca_6m = EXCLUDED.ipca_6m,
            ipca_12m = EXCLUDED.ipca_12m,
            ipca_change_3m = EXCLUDED.ipca_change_3m,
            ipca_change_6m = EXCLUDED.ipca_change_6m,

            updated_at = NOW()
    """)

    records = df.replace({np.nan: None}).to_dict(orient="records")
    for row in records:
        session.execute(sql, row)


def run(
    session,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> None:
    """
    Ingere/calcula features de mercado por data.

    Essas features são iguais para todos os ativos no mesmo dia e devem ser
    usadas como features de regime no dataset de treino.
    """
    resolved_end_date = parse_date(end_date) if end_date else date.today()
    resolved_start_date = (
        parse_date(start_date)
        if start_date
        else resolved_end_date - timedelta(days=3650)
    )

    logger.info(
        "Iniciando ingestão de market features: %s → %s",
        resolved_start_date,
        resolved_end_date,
    )

    try:
        df = build_market_features(
            start_date=resolved_start_date,
            end_date=resolved_end_date,
        )

        if df.empty:
            logger.warning("Nenhuma market feature gerada.")
            return

        upsert_market_features(session, df)
        session.commit()

        logger.info(
            "Market features salvas: %s linhas | %s → %s",
            len(df),
            df["trade_date"].min().date(),
            df["trade_date"].max().date(),
        )

    except Exception as exc:
        session.rollback()
        logger.exception("Erro ao ingerir market features: %s", exc)
        raise