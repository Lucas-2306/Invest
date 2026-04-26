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

MARKET_FEATURE_COLUMNS = [
    "ibov_return_1d",
    "ibov_return_5d",
    "ibov_return_21d",
    "ibov_return_63d",
    "ibov_vol_21d",
    "ibov_vol_63d",
    "ibov_above_sma_200",

    "sp500_return_1d",
    "sp500_return_5d",
    "sp500_return_21d",
    "sp500_return_63d",
    "sp500_vol_21d",
    "sp500_vol_63d",
    "sp500_above_sma_200",

    "selic_rate",
    "selic_change_21d",
    "selic_change_63d",

    "ipca_monthly",
    "ipca_3m",
    "ipca_6m",
    "ipca_12m",
    "ipca_change_3m",
    "ipca_change_6m",
]

MACRO_INTERACTION_COLUMNS = [
    "return_21d_x_ibov_return_21d",
    "return_63d_x_ibov_return_63d",
    "return_126d_x_ibov_return_63d",
    "momentum_21_63_x_ibov_return_63d",
    "momentum_5_21_x_ibov_return_21d",

    "return_21d_x_sp500_return_21d",
    "return_63d_x_sp500_return_63d",
    "return_126d_x_sp500_return_63d",

    "volatility_21d_x_ibov_vol_21d",
    "volatility_63d_x_ibov_vol_63d",
    "volatility_21d_x_sp500_vol_21d",
    "volatility_63d_x_sp500_vol_63d",

    "return_21d_x_ibov_above_sma_200",
    "return_63d_x_ibov_above_sma_200",
    "momentum_21_63_x_ibov_above_sma_200",
    "volatility_21d_x_ibov_above_sma_200",

    "return_21d_x_selic_change_21d",
    "return_63d_x_selic_change_63d",
    "return_21d_x_ipca_12m",
]

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


def load_dataset(start_date: str, end_date: str | None = None) -> pd.DataFrame:
    where_end = f"AND f.trade_date < '{end_date}'" if end_date else ""

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
            lf.fifty_two_week_low,

            mf.ibov_close,
            mf.ibov_return_1d,
            mf.ibov_return_5d,
            mf.ibov_return_21d,
            mf.ibov_return_63d,
            mf.ibov_vol_21d,
            mf.ibov_vol_63d,
            mf.ibov_above_sma_200,

            mf.sp500_close,
            mf.sp500_return_1d,
            mf.sp500_return_5d,
            mf.sp500_return_21d,
            mf.sp500_return_63d,
            mf.sp500_vol_21d,
            mf.sp500_vol_63d,
            mf.sp500_above_sma_200,

            mf.selic_rate,
            mf.selic_change_21d,
            mf.selic_change_63d,

            mf.ipca_monthly,
            mf.ipca_3m,
            mf.ipca_6m,
            mf.ipca_12m,
            mf.ipca_change_3m,
            mf.ipca_change_6m

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

        LEFT JOIN macro_data.market_features mf
          ON mf.trade_date = f.trade_date

        WHERE f.trade_date >= '{start_date}'
        {where_end}
        ORDER BY f.trade_date, s.symbol
    """

    return pd.read_sql_query(
        query,
        engine,
        parse_dates=["trade_date", "fundamentals_reference_date"],
    )


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

    for col in ["fifty_two_week_high", "fifty_two_week_low", "sma_20"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_high = df["fifty_two_week_high"].notna() & (df["fifty_two_week_high"] > 0)
    valid_low = df["fifty_two_week_low"].notna() & (df["fifty_two_week_low"] > 0)
    valid_sma20 = df["sma_20"].notna() & (df["sma_20"] > 0)

    df["sma20_to_52w_high"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df["sma20_to_52w_low"] = pd.Series(np.nan, index=df.index, dtype="float64")

    df.loc[valid_sma20 & valid_high, "sma20_to_52w_high"] = (
        df.loc[valid_sma20 & valid_high, "sma_20"]
        / df.loc[valid_sma20 & valid_high, "fifty_two_week_high"]
    )

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


def add_macro_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed_cols = [
        "return_21d",
        "return_63d",
        "return_126d",
        "momentum_21_63",
        "momentum_5_21",
        "volatility_21d",
        "volatility_63d",
        "ibov_return_21d",
        "ibov_return_63d",
        "ibov_vol_21d",
        "ibov_vol_63d",
        "ibov_above_sma_200",
        "sp500_return_21d",
        "sp500_return_63d",
        "sp500_vol_21d",
        "sp500_vol_63d",
        "selic_change_21d",
        "selic_change_63d",
        "ipca_12m",
    ]

    for col in needed_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = {
        "return_21d_x_ibov_return_21d": ("return_21d", "ibov_return_21d"),
        "return_63d_x_ibov_return_63d": ("return_63d", "ibov_return_63d"),
        "return_126d_x_ibov_return_63d": ("return_126d", "ibov_return_63d"),
        "momentum_21_63_x_ibov_return_63d": ("momentum_21_63", "ibov_return_63d"),
        "momentum_5_21_x_ibov_return_21d": ("momentum_5_21", "ibov_return_21d"),

        "return_21d_x_sp500_return_21d": ("return_21d", "sp500_return_21d"),
        "return_63d_x_sp500_return_63d": ("return_63d", "sp500_return_63d"),
        "return_126d_x_sp500_return_63d": ("return_126d", "sp500_return_63d"),

        "volatility_21d_x_ibov_vol_21d": ("volatility_21d", "ibov_vol_21d"),
        "volatility_63d_x_ibov_vol_63d": ("volatility_63d", "ibov_vol_63d"),
        "volatility_21d_x_sp500_vol_21d": ("volatility_21d", "sp500_vol_21d"),
        "volatility_63d_x_sp500_vol_63d": ("volatility_63d", "sp500_vol_63d"),

        "return_21d_x_ibov_above_sma_200": ("return_21d", "ibov_above_sma_200"),
        "return_63d_x_ibov_above_sma_200": ("return_63d", "ibov_above_sma_200"),
        "momentum_21_63_x_ibov_above_sma_200": ("momentum_21_63", "ibov_above_sma_200"),
        "volatility_21d_x_ibov_above_sma_200": ("volatility_21d", "ibov_above_sma_200"),

        "return_21d_x_selic_change_21d": ("return_21d", "selic_change_21d"),
        "return_63d_x_selic_change_63d": ("return_63d", "selic_change_63d"),
        "return_21d_x_ipca_12m": ("return_21d", "ipca_12m"),
    }

    for new_col, (left_col, right_col) in pairs.items():
        if left_col in df.columns and right_col in df.columns:
            df[new_col] = df[left_col] * df[right_col]
        else:
            df[new_col] = np.nan

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = normalize_sector(df)
    df = add_sector_relative_features(df)
    df = add_fundamental_relative_features(df)
    df = add_macro_interaction_features(df)

    df = apply_liquidity_filter(df)

    df = add_cross_sectional_features(df)
    df = clip_target(df)

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

    parquet_path = output_dir / "model_dataset.parquet"
    df.to_parquet(parquet_path, index=False)

    print(f"Parquet salvo em: {parquet_path}")


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

    market_cols = [col for col in MARKET_FEATURE_COLUMNS if col in df.columns]
    if market_cols:
        print("\nPreenchimento de colunas de mercado")
        print(df[market_cols].notna().mean().sort_values())


def main() -> None:
    start_year = pd.Timestamp(settings.dataset_start_date).year
    end_year = pd.Timestamp.today().year

    parts = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        print(f"\nGerando dataset parcial: {start_date} → {end_date}")

        raw_df = load_dataset(start_date=start_date, end_date=end_date)

        if raw_df.empty:
            print(f"Sem dados para {year}")
            continue

        part_df = build_features(raw_df)

        if part_df.empty:
            print(f"Parte {year} vazia após filtros")
            continue

        parts.append(part_df)
        print(f"Parte {year}: {len(part_df)} linhas")

    if not parts:
        raise ValueError("Nenhuma parte do dataset foi gerada.")

    dataset_df = pd.concat(parts, ignore_index=True)
    dataset_df = dataset_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    save_dataset(dataset_df, OUTPUT_DIR)
    print_summary(dataset_df)


if __name__ == "__main__":
    main()