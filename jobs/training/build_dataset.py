from pathlib import Path
import pandas as pd
import numpy as np

from core.training_config import OUTPUT_DIR, settings
from core.training_db import engine

MIN_PRICE = 5.0
MIN_AVG_DAILY_VOLUME_20D = 100000
MIN_AVG_DAILY_TRADED_VALUE_20D = 1000000.0

BASE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    "return_126d",
    "log_return_1d",
    "price_sma_20_ratio",
    "volatility_21d",
    "volatility_63d",
    "volatility_ratio_21_63",
    "return_21d_over_vol_21d",
    "volume_ratio_20d",
    "high_low_ratio",
    "gap",
    "return_1d_vs_sector",
    "return_5d_vs_sector",
    "momentum_21_63",
    "momentum_5_21",
    "avg_daily_volume_20d",
    "avg_daily_traded_value_20d",
    "volume_trend_5_20",
    "traded_value_trend_5_20",

    "market_cap",
    "price_to_earnings",
    "price_to_book",
    "eps",
    "roe",
    "roa",
    "debt_to_equity",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "sma20_to_52w_high",
    "sma20_to_52w_low",
]

CS_FEATURE_COLUMNS = [f"{col}_cs" for col in BASE_FEATURE_COLUMNS]

# colunas mínimas para manter uma linha no dataset
REQUIRED_CORE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    "log_return_1d",
    "price_sma_20_ratio",
    "volatility_21d",
    "volume_ratio_20d",
    "high_low_ratio",
    "gap",
    "return_1d_vs_sector",
    "return_5d_vs_sector",
    "avg_daily_volume_20d",
    "avg_daily_traded_value_20d",
    "target_63d",
]


def load_dataset(start_date: str) -> pd.DataFrame:
    query = f"""
        SELECT
            f.symbol_id,
            s.symbol,
            f.trade_date,
            f.return_1d,
            f.return_5d,
            f.return_21d,
            f.log_return_1d,
            f.sma_20,
            f.sma_50,
            f.price_sma_20_ratio,
            f.volatility_21d,
            f.volume_ratio_20d,
            f.high_low_ratio,
            f.gap,
            f.target_5d,
            f.target_5d_t1,
            f.target_21d,
            f.target_21d_t1,
            f.target_63d,
            f.target_63d_t1,
            f.avg_daily_volume_20d,
            f.avg_daily_traded_value_20d,
            f.return_63d,
            f.return_126d,
            f.volatility_63d,
            f.volatility_ratio_21_63,
            f.return_21d_over_vol_21d,
            f.momentum_21_63,
            f.momentum_5_21,
            f.volume_trend_5_20,
            f.traded_value_trend_5_20,
            c.sector,

            lf.reference_date AS fundamentals_reference_date,
            lf.market_cap,
            lf.price_to_earnings,
            lf.price_to_book,
            lf.eps,
            lf.roe,
            lf.roa,
            lf.debt_to_equity,
            lf.dividend_yield,
            lf.beta,
            lf.fifty_two_week_high,
            lf.fifty_two_week_low

        FROM market_data.features f
        JOIN market_data.symbols s
          ON s.id = f.symbol_id
        JOIN market_data.companies c
          ON c.id = s.company_id

        LEFT JOIN LATERAL (
            SELECT
                fs.reference_date,
                fs.market_cap,
                fs.price_to_earnings,
                fs.price_to_book,
                fs.eps,
                fs.roe,
                fs.roa,
                fs.debt_to_equity,
                fs.dividend_yield,
                fs.beta,
                fs.fifty_two_week_high,
                fs.fifty_two_week_low
            FROM market_data.fundamentals_snapshot fs
            WHERE fs.symbol_id = f.symbol_id
              AND fs.reference_date <= f.trade_date
            ORDER BY fs.reference_date DESC
            LIMIT 1
        ) lf ON TRUE

        WHERE f.trade_date >= '{start_date}'
        ORDER BY f.trade_date, s.symbol
    """

    df = pd.read_sql_query(query, engine, parse_dates=["trade_date", "fundamentals_reference_date"])
    return df


def normalize_sector(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sector"] = (
        df["sector"]
        .fillna("Desconhecido")
        .astype(str)
        .str.strip()
        .replace("", "Desconhecido")
    )

    return df


def add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sector_return_1d_mean"] = df.groupby(["trade_date", "sector"])["return_1d"].transform("mean")
    df["return_1d_vs_sector"] = df["return_1d"] - df["sector_return_1d_mean"]

    df["sector_return_5d_mean"] = df.groupby(["trade_date", "sector"])["return_5d"].transform("mean")
    df["return_5d_vs_sector"] = df["return_5d"] - df["sector_return_5d_mean"]

    df = df.drop(columns=["sector_return_1d_mean", "sector_return_5d_mean"])

    return df


def add_fundamental_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_high = df["fifty_two_week_high"].notna() & (df["fifty_two_week_high"] > 0)
    valid_low = df["fifty_two_week_low"].notna() & (df["fifty_two_week_low"] > 0)
    valid_sma20 = df["sma_20"].notna() & (df["sma_20"] > 0)

    df["sma20_to_52w_high"] = np.nan
    df.loc[valid_sma20 & valid_high, "sma20_to_52w_high"] = (
        df.loc[valid_sma20 & valid_high, "sma_20"]
        / df.loc[valid_sma20 & valid_high, "fifty_two_week_high"]
    )

    df["sma20_to_52w_low"] = np.nan
    df.loc[valid_sma20 & valid_low, "sma20_to_52w_low"] = (
        df.loc[valid_sma20 & valid_low, "sma_20"]
        / df.loc[valid_sma20 & valid_low, "fifty_two_week_low"]
    )

    df["sma20_to_52w_high"] = df["sma20_to_52w_high"].where(
        df["sma20_to_52w_high"].between(0.01, 100)
    )
    df["sma20_to_52w_low"] = df["sma20_to_52w_low"].where(
        df["sma20_to_52w_low"].between(0.01, 100)
    )

    return df


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cs_columns = BASE_FEATURE_COLUMNS.copy()

    for col in cs_columns:
        df[f"{col}_cs"] = df.groupby("trade_date")[col].rank(pct=True)

    return df


def apply_liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_volume = (
        df["avg_daily_volume_20d"].notna()
        & (df["avg_daily_volume_20d"] >= MIN_AVG_DAILY_VOLUME_20D)
    )
    valid_traded_value = (
        df["avg_daily_traded_value_20d"].notna()
        & (df["avg_daily_traded_value_20d"] >= MIN_AVG_DAILY_TRADED_VALUE_20D)
    )

    df = df[valid_volume & valid_traded_value].copy()
    return df


def clip_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_5d"] = df["target_5d"].clip(-0.2, 0.2)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = normalize_sector(df)
    df = add_sector_relative_features(df)
    df = add_fundamental_relative_features(df)
    df = add_cross_sectional_features(df)
    df = clip_target(df)
    df = apply_liquidity_filter(df)

    # exige apenas o núcleo técnico + target
    df = df.dropna(subset=REQUIRED_CORE_COLUMNS)

    df = pd.get_dummies(
        df,
        columns=["sector"],
        prefix="sector",
        dtype=int,
    )

    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    df["symbol_id"] = df["symbol_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)

    return df


def save_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "model_dataset.csv"
    df.to_csv(csv_path, index=False)

    parquet_path = output_dir / "model_dataset.parquet"
    df.to_parquet(parquet_path, index=False)

    print(f"Parquet salvo em: {parquet_path}")
    print(f"CSV salvo em: {csv_path}")


def print_summary(df: pd.DataFrame) -> None:
    print("\nResumo do dataset")
    print("-----------------")
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {len(df.columns)}")
    print(f"Data mínima: {df['trade_date'].min()}")
    print(f"Data máxima: {df['trade_date'].max()}")
    print(f"Símbolos únicos: {df['symbol'].nunique()}")

    print("\nResumo do target_5d (clipado)")
    print(df["target_5d"].describe())

    print("\nPreenchimento de colunas fundamentais")
    print(
        df[
            [
                "market_cap",
                "price_to_earnings",
                "price_to_book",
                "eps",
                "roe",
                "roa",
                "debt_to_equity",
                "dividend_yield",
                "beta",
                "fifty_two_week_high",
                "fifty_two_week_low",
                "sma20_to_52w_high",
                "sma20_to_52w_low",
            ]
        ].notna().mean().sort_values()
    )

    if "fundamentals_reference_date" in df.columns:
        lag_days = (df["trade_date"] - df["fundamentals_reference_date"]).dt.days
        print("\nLag dos fundamentals usados (dias)")
        print(lag_days.describe())


def main() -> None:
    raw_df = load_dataset(start_date=settings.dataset_start_date)
    dataset_df = build_features(raw_df)
    save_dataset(dataset_df, OUTPUT_DIR)
    print_summary(dataset_df)


if __name__ == "__main__":
    main()