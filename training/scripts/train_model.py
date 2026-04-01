from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import OUTPUT_DIR


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

DATASET_PATH = OUTPUT_DIR / "model_dataset.parquet"
CSV_FALLBACK_PATH = OUTPUT_DIR / "model_dataset.csv"

TARGET_COLUMN = "target_5d"

DROP_COLUMNS = [
    "symbol_id",
    "symbol",
    "trade_date",
    "target_5d",
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


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    unique_dates = sorted(df["trade_date"].unique())
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]

    train_df = df[df["trade_date"] < split_date].copy()
    test_df = df[df["trade_date"] >= split_date].copy()

    print(f"Split temporal em: {split_date}")
    print(f"Treino: {train_df['trade_date'].min()} → {train_df['trade_date'].max()}")
    print(f"Teste:  {test_df['trade_date'].min()} → {test_df['trade_date'].max()}")

    return train_df, test_df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


def transform_with_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_imputed


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        force_col_wise=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_rmse": float(mean_squared_error(y_train, train_pred) ** 0.5),
        "test_rmse": float(mean_squared_error(y_test, test_pred) ** 0.5),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "train_spearman_corr": float(spearmanr(y_train, train_pred).correlation),
        "test_spearman_corr": float(spearmanr(y_test, test_pred).correlation),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
    }

    print("\nMétricas")
    print("--------")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def save_predictions(
    model: LGBMRegressor,
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pred = model.predict(X_test)

    preds = test_df[["trade_date", "symbol", "target_5d"]].copy()
    preds["prediction"] = pred
    preds = preds.sort_values(["trade_date", "prediction"], ascending=[True, False])

    output_path = REPORT_DIR / "test_predictions.csv"
    preds.to_csv(output_path, index=False)

    print(f"Predições salvas em: {output_path}")


def save_artifacts(
    model: LGBMRegressor,
    imputer: SimpleImputer,
    feature_names: list[str],
    metrics: dict,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "lightgbm_regression.joblib"
    imputer_path = MODEL_DIR / "imputer.joblib"
    features_path = MODEL_DIR / "feature_names.json"
    metrics_path = REPORT_DIR / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {model_path}")
    print(f"Imputer salvo em: {imputer_path}")
    print(f"Features salvas em: {features_path}")
    print(f"Métricas salvas em: {metrics_path}")


def save_feature_importance(model: LGBMRegressor, feature_names: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    output_path = REPORT_DIR / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)

    print(f"Importâncias salvas em: {output_path}")


def main() -> None:
    df = load_dataset()

    train_df, test_df = temporal_split(df)

    X_train_raw, y_train = build_xy(train_df)
    X_test_raw, y_test = build_xy(test_df)

    feature_names = X_train_raw.columns.tolist()

    print("\nResumo do treino")
    print("----------------")
    print(f"X_train: {X_train_raw.shape}")
    print(f"X_test: {X_test_raw.shape}")

    print("\nResumo do target treino")
    print(y_train.describe())

    print("\nResumo do target teste")
    print(y_test.describe())

    imputer = fit_imputer(X_train_raw)
    X_train = transform_with_imputer(imputer, X_train_raw)
    X_test = transform_with_imputer(imputer, X_test_raw)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    save_artifacts(model, imputer, feature_names, metrics)
    save_predictions(model, test_df, X_test)
    save_feature_importance(model, feature_names)


if __name__ == "__main__":
    main()