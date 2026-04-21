from pathlib import Path
import pandas as pd

from core.training_config import OUTPUT_DIR, settings
from core.training_db import engine


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "log_return_1d",
    "sma_20",
    "sma_50",
    "price_sma_20_ratio",
    "volatility_21d",
    "volume_ratio_20d",
    "high_low_ratio",
    "gap",
    "target_5d",
    "return_1d_vs_sector",
    "return_5d_vs_sector",
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
            c.sector
        FROM market_data.features f
        JOIN market_data.symbols s
          ON s.id = f.symbol_id
        JOIN market_data.companies c
          ON c.id = s.company_id
        WHERE f.trade_date >= '{start_date}'
        ORDER BY f.trade_date, s.symbol
    """

    df = pd.read_sql_query(query, engine, parse_dates=["trade_date"])
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


def clip_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 🔥 remove outliers extremos
    df["target_5d"] = df["target_5d"].clip(-0.2, 0.2)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = normalize_sector(df)
    df = add_sector_relative_features(df)

    # 🔥 NOVO
    df = clip_target(df)

    df = df.dropna(subset=FEATURE_COLUMNS)

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


def main() -> None:
    raw_df = load_dataset(start_date=settings.dataset_start_date)
    dataset_df = build_features(raw_df)
    save_dataset(dataset_df, OUTPUT_DIR)
    print_summary(dataset_df)


if __name__ == "__main__":
    main()