from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.training_config import OUTPUT_DIR


BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "artifacts" / "reports"

DATASET_PATH = OUTPUT_DIR / "model_dataset.parquet"
CSV_FALLBACK_PATH = OUTPUT_DIR / "model_dataset.csv"

TARGET_COLUMNS = ["target_5d", "target_21d", "target_63d"]

DROP_COLUMNS = [
    "symbol_id",
    "symbol",
    "trade_date",
    "target_5d",
    "target_5d_t1",
    "target_21d",
    "target_21d_t1",
    "target_63d",
    "target_63d_t1",
]


def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        df = pd.read_parquet(DATASET_PATH)
        print(f"Dataset carregado de: {DATASET_PATH}")
        return df

    if CSV_FALLBACK_PATH.exists():
        df = pd.read_csv(CSV_FALLBACK_PATH, parse_dates=["trade_date"])
        print(f"Dataset carregado de: {CSV_FALLBACK_PATH}")
        return df

    raise FileNotFoundError("Nenhum dataset encontrado. Rode antes o build_dataset.py.")


def mean_daily_spearman(df: pd.DataFrame, feature_col: str, target_col: str) -> float | None:
    daily_values = []

    for _, group in df.groupby("trade_date"):
        temp = group[[feature_col, target_col]].dropna()

        if len(temp) < 5:
            continue

        if temp[feature_col].nunique() <= 1:
            continue

        if temp[target_col].nunique() <= 1:
            continue

        corr = spearmanr(temp[feature_col], temp[target_col]).correlation
        if corr is not None and not np.isnan(corr):
            daily_values.append(corr)

    if not daily_values:
        return None

    return float(np.mean(daily_values))


def check_constant_features(df):
    constant_features = []

    for col in df.columns:
        if col in DROP_COLUMNS:
            continue

        n_unique = df[col].nunique()

        if n_unique <= 1:
            constant_features.append(col)

    print("\nFeatures completamente constantes:")
    for f in constant_features:
        print(f)


def build_feature_ic_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "return_1d",
        "return_5d",
        "return_21d",
        "return_63d",
        "return_126d",
        "price_sma_20_ratio",
        "volatility_21d",
        "volatility_63d",
        "volatility_ratio_21_63",
        "return_21d_over_vol_21d",
        "volume_ratio_20d",
        "avg_daily_volume_20d",
        "avg_daily_traded_value_20d",
        "volume_trend_5_20",
        "traded_value_trend_5_20",
        "high_low_ratio",
        "gap",
        "return_1d_vs_sector",
        "return_5d_vs_sector",
        "momentum_21_63",
        "momentum_5_21",
    ]

    rows = []

    for feature in feature_cols:
        row = {"feature": feature}

        for target in TARGET_COLUMNS:
            if target not in df.columns:
                row[f"{target}_ic"] = None
                continue

            ic = mean_daily_spearman(df, feature, target)
            row[f"{target}_ic"] = ic

        rows.append(row)

    out = pd.DataFrame(rows)

    for target in TARGET_COLUMNS:
        col = f"{target}_ic"
        if col in out.columns:
            out[f"{target}_abs_ic"] = out[col].abs()

    return out


def check_constant_by_day(df):
    bad_features = {}

    for col in df.columns:
        if col in DROP_COLUMNS:
            continue

        count_constant_days = 0

        for _, group in df.groupby("trade_date"):
            if group[col].nunique() <= 1:
                count_constant_days += 1

        if count_constant_days > 0:
            bad_features[col] = count_constant_days

    print("\nFeatures constantes por dia:")
    for k, v in bad_features.items():
        print(f"{k}: {v} dias")


def save_outputs(ic_table: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "feature_ic_analysis.csv"
    ic_table.to_csv(output_path, index=False)

    print(f"Análise de IC por feature salva em: {output_path}")


def print_top_features(ic_table: pd.DataFrame, target: str, top_n: int = 20) -> None:
    col = f"{target}_ic"
    abs_col = f"{target}_abs_ic"

    if col not in ic_table.columns:
        return

    valid = ic_table.dropna(subset=[col]).copy()
    if valid.empty:
        print(f"\nSem dados para {target}")
        return

    top_pos = valid.sort_values(col, ascending=False).head(top_n)[["feature", col]]
    top_neg = valid.sort_values(col, ascending=True).head(top_n)[["feature", col]]
    top_abs = valid.sort_values(abs_col, ascending=False).head(top_n)[["feature", col]]

    print(f"\nTop {top_n} features positivas para {target}")
    print(top_pos.to_string(index=False))

    print(f"\nTop {top_n} features negativas para {target}")
    print(top_neg.to_string(index=False))

    print(f"\nTop {top_n} features por |IC| para {target}")
    print(top_abs.to_string(index=False))


def main() -> None:
    df = load_dataset()

    check_constant_features(df)
    check_constant_by_day(df)

    ic_table = build_feature_ic_table(df)
    save_outputs(ic_table)

    for target in TARGET_COLUMNS:
        print_top_features(ic_table, target=target, top_n=20)


if __name__ == "__main__":
    main()