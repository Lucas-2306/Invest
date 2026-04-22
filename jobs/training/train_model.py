
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

from core.training_config import OUTPUT_DIR
from jobs.training.feature_config import MODEL_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "artifacts" / "reports"

DATASET_PATH = OUTPUT_DIR / "model_dataset.parquet"
CSV_FALLBACK_PATH = OUTPUT_DIR / "model_dataset.csv"


@dataclass
class TrainConfig:
    """Parâmetros configuráveis do pipeline de treino."""

    target_column: str = "target_63d"
    rank_label_buckets: int = 100
    train_ratio: float = 0.8
    split_date: str | None = None
    test_end_date: str | None = None

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42

    model_filename: str = "lightgbm_ranker.joblib"
    imputer_filename: str = "imputer.joblib"
    feature_names_filename: str = "feature_names.json"
    metrics_filename: str = "metrics.json"
    predictions_filename: str = "test_predictions.csv"
    feature_importance_filename: str = "feature_importance.csv"

    verbose: bool = True

    @property
    def target_t1_column(self) -> str:
        return f"{self.target_column}_t1"


def log(message: str, enabled: bool = True) -> None:
    """Imprime mensagens apenas quando o modo verboso estiver ativo."""
    if enabled:
        print(message)


def load_dataset(verbose: bool = True) -> pd.DataFrame:
    """Carrega o dataset principal, preferindo parquet e usando CSV como fallback."""
    if DATASET_PATH.exists():
        df = pd.read_parquet(DATASET_PATH)
        log(f"Dataset carregado de: {DATASET_PATH}", verbose)
        return df

    if CSV_FALLBACK_PATH.exists():
        df = pd.read_csv(CSV_FALLBACK_PATH, parse_dates=["trade_date"])
        log(f"Dataset carregado de: {CSV_FALLBACK_PATH}", verbose)
        return df

    raise FileNotFoundError("Nenhum dataset encontrado. Rode antes o build_dataset.py.")


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    split_date: str | pd.Timestamp | None = None,
    test_end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Faz o split temporal do dataset.

    Se split_date for informado, ele é usado como data de corte.
    Caso contrário, a data é inferida a partir do train_ratio.
    """
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset vazio. Não é possível fazer split temporal.")

    resolved_split_date: pd.Timestamp

    if split_date is not None:
        resolved_split_date = pd.Timestamp(split_date)
        resolved_test_end_date = pd.Timestamp(test_end_date) if test_end_date is not None else None

        train_df = df[df["trade_date"] < resolved_split_date].copy()

        if resolved_test_end_date is None:
            test_df = df[df["trade_date"] >= resolved_split_date].copy()
        else:
            test_df = df[
                (df["trade_date"] >= resolved_split_date)
                & (df["trade_date"] < resolved_test_end_date)
            ].copy()
    else:
        unique_dates = sorted(df["trade_date"].unique())

        if len(unique_dates) == 0:
            raise ValueError("Dataset vazio. Não é possível fazer split temporal.")

        split_idx = int(len(unique_dates) * train_ratio)
        if split_idx >= len(unique_dates):
            split_idx = len(unique_dates) - 1

        resolved_split_date = pd.Timestamp(unique_dates[split_idx])
        train_df = df[df["trade_date"] < resolved_split_date].copy()
        test_df = df[df["trade_date"] >= resolved_split_date].copy()

    if train_df.empty:
        raise ValueError("Treino vazio após split temporal.")

    if test_df.empty:
        raise ValueError("Teste vazio após split temporal.")

    return train_df, test_df, resolved_split_date


def filter_rows_with_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Remove linhas sem target válido."""
    return df[df[target_column].notna()].copy()


def build_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e target com base nas colunas disponíveis no dataset."""
    available_features = [col for col in MODEL_FEATURE_COLUMNS if col in df.columns]
    X = df[available_features].copy()
    y = df[target_column].copy()
    return X, y


def drop_all_null_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove colunas que são 100% nulas no treino.

    O mesmo subconjunto de colunas é aplicado ao teste para manter consistência.
    """
    valid_columns = X_train.columns[X_train.notna().any()].tolist()
    X_train_filtered = X_train[valid_columns].copy()
    X_test_filtered = X_test[valid_columns].copy()

    return X_train_filtered, X_test_filtered, valid_columns


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    """Ajusta imputação por mediana com base apenas no treino."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


def transform_with_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica o imputer e retorna DataFrame preservando colunas e índice."""
    transformed = imputer.transform(X)
    return pd.DataFrame(transformed, columns=X.columns, index=X.index)


def build_rank_targets(df: pd.DataFrame, target_column: str, rank_label_buckets: int) -> pd.Series:
    """
    Converte o target contínuo em labels de ranking por data.

    Cada data é ranqueada em percentis e convertida em buckets inteiros.
    """
    pct_rank = df.groupby("trade_date")[target_column].rank(method="average", pct=True)
    labels = np.floor((pct_rank - 1e-12) * rank_label_buckets)
    labels = labels.clip(lower=0, upper=rank_label_buckets - 1)

    return pd.Series(labels, index=df.index, name=f"{target_column}_rank")


def build_groups(df: pd.DataFrame) -> list[int]:
    """Monta os grupos por data exigidos pelo LGBMRanker."""
    groups = df.groupby("trade_date").size().tolist()
    if not groups:
        raise ValueError("Nenhum grupo encontrado para ranking.")
    return groups


def train_model(
    X_train: pd.DataFrame,
    y_train_rank: pd.Series,
    train_groups: list[int],
    config: TrainConfig,
) -> LGBMRanker:
    """Treina o modelo de ranking com os hiperparâmetros configurados."""
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        force_col_wise=True,
        label_gain=list(range(config.rank_label_buckets)),
    )
    model.fit(X_train, y_train_rank, group=train_groups)
    return model


def mean_daily_spearman(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
) -> float | None:
    """Calcula a média do Spearman cross-sectional diário."""
    daily_values = []

    for _, group in df.groupby("trade_date"):
        if len(group) < 2:
            continue

        corr = spearmanr(group[target_col], group[pred_col]).correlation
        if corr is not None and not np.isnan(corr):
            daily_values.append(corr)

    if not daily_values:
        return None

    return float(np.mean(daily_values))


def evaluate_model(
    model: LGBMRanker,
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: TrainConfig,
) -> dict:
    """Gera previsões de treino/teste e calcula métricas principais."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_eval_df = train_df[["trade_date"]].copy()
    train_eval_df["target"] = y_train.values
    train_eval_df["prediction"] = train_pred

    test_eval_df = test_df[["trade_date"]].copy()
    test_eval_df["target"] = y_test.values
    test_eval_df["prediction"] = test_pred

    train_spearman = spearmanr(y_train, train_pred).correlation
    test_spearman = spearmanr(y_test, test_pred).correlation

    return {
        "target_column": config.target_column,
        "train_spearman_corr": float(train_spearman) if train_spearman is not None and not np.isnan(train_spearman) else None,
        "test_spearman_corr": float(test_spearman) if test_spearman is not None and not np.isnan(test_spearman) else None,
        "train_daily_ic_mean": mean_daily_spearman(train_eval_df, "target", "prediction"),
        "test_daily_ic_mean": mean_daily_spearman(test_eval_df, "target", "prediction"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
        "ranking_label_buckets": int(config.rank_label_buckets),
        "n_estimators": int(config.n_estimators),
        "max_depth": int(config.max_depth),
        "learning_rate": float(config.learning_rate),
        "random_state": int(config.random_state),
    }


def save_predictions(
    model: LGBMRanker,
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    config: TrainConfig,
) -> Path:
    """Salva as previsões do conjunto de teste para uso posterior no backtest."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pred = model.predict(X_test)

    required_cols = [
        "trade_date",
        "symbol",
        config.target_column,
        config.target_t1_column,
        "avg_daily_volume_20d",
        "avg_daily_traded_value_20d",
    ]
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes para salvar predições: {missing_cols}")

    preds = test_df[required_cols].copy()
    preds = preds.rename(
        columns={
            config.target_column: "target_model",
            config.target_t1_column: "target_exec_t1",
        }
    )
    preds["prediction"] = pred
    preds = preds.sort_values(["trade_date", "prediction"], ascending=[True, False])

    output_path = REPORT_DIR / config.predictions_filename
    preds.to_csv(output_path, index=False)
    return output_path


def save_artifacts(
    model: LGBMRanker,
    imputer: SimpleImputer,
    feature_names: list[str],
    metrics: dict,
    config: TrainConfig,
) -> dict[str, Path]:
    """Salva modelo, imputer, lista de features e métricas."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / config.model_filename
    imputer_path = MODEL_DIR / config.imputer_filename
    features_path = MODEL_DIR / config.feature_names_filename
    metrics_path = REPORT_DIR / config.metrics_filename

    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)

    with open(features_path, "w", encoding="utf-8") as file:
        json.dump(feature_names, file, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    return {
        "model_path": model_path,
        "imputer_path": imputer_path,
        "features_path": features_path,
        "metrics_path": metrics_path,
    }


def save_feature_importance(model: LGBMRanker, feature_names: list[str], config: TrainConfig) -> Path:
    """Salva a importância das features produzida pelo modelo."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    output_path = REPORT_DIR / config.feature_importance_filename
    importance_df.to_csv(output_path, index=False)
    return output_path


def build_execution_summary(
    config: TrainConfig,
    split_date: pd.Timestamp,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dropped_columns: list[str],
    train_groups: list[int],
    test_groups: list[int],
    metrics: dict,
) -> list[str]:
    """Monta um resumo compacto da execução para logs no terminal."""
    lines = [
        f"Split temporal em: {split_date}",
        f"Treino: {train_df['trade_date'].min()} → {train_df['trade_date'].max()}",
        f"Teste:  {test_df['trade_date'].min()} → {test_df['trade_date'].max()}",
        f"Target: {config.target_column}",
        f"X_train: {len(train_df)} linhas | X_test: {len(test_df)} linhas",
        f"Features usadas: {metrics['num_features']}",
        f"Colunas removidas por serem 100% nulas no treino: {len(dropped_columns)}",
        f"Grupos treino: {len(train_groups)} | média: {np.mean(train_groups):.2f}",
        f"Grupos teste: {len(test_groups)} | média: {np.mean(test_groups):.2f}",
        f"Train Spearman: {metrics['train_spearman_corr']}",
        f"Test Spearman: {metrics['test_spearman_corr']}",
        f"Train daily IC mean: {metrics['train_daily_ic_mean']}",
        f"Test daily IC mean: {metrics['test_daily_ic_mean']}",
    ]

    if config.test_end_date is not None:
        lines.insert(3, f"Fim do teste configurado em: {pd.Timestamp(config.test_end_date)}")

    return lines


def run_training_pipeline(config: TrainConfig) -> dict:
    """Executa todo o pipeline de treino e retorna os principais objetos/artefatos."""
    df = load_dataset(verbose=config.verbose)

    train_df, test_df, split_date = temporal_split(
        df=df,
        train_ratio=config.train_ratio,
        split_date=config.split_date,
        test_end_date=config.test_end_date,
    )

    train_df = filter_rows_with_target(train_df, config.target_column)
    test_df = filter_rows_with_target(test_df, config.target_column)

    X_train_raw, y_train = build_xy(train_df, config.target_column)
    X_test_raw, y_test = build_xy(test_df, config.target_column)

    original_columns = list(X_train_raw.columns)
    X_train_raw, X_test_raw, feature_names = drop_all_null_columns(X_train_raw, X_test_raw)
    dropped_columns = [col for col in original_columns if col not in feature_names]

    y_train_rank = build_rank_targets(train_df, config.target_column, config.rank_label_buckets).astype(int)
    y_test_rank = build_rank_targets(test_df, config.target_column, config.rank_label_buckets).astype(int)

    train_groups = build_groups(train_df)
    test_groups = build_groups(test_df)

    imputer = fit_imputer(X_train_raw)
    X_train = transform_with_imputer(imputer, X_train_raw)
    X_test = transform_with_imputer(imputer, X_test_raw)

    model = train_model(X_train, y_train_rank, train_groups, config)
    metrics = evaluate_model(model, train_df, X_train, y_train, test_df, X_test, y_test, config)

    artifact_paths = save_artifacts(model, imputer, feature_names, metrics, config)
    predictions_path = save_predictions(model, test_df, X_test, config)
    importance_path = save_feature_importance(model, feature_names, config)

    if config.verbose:
        print("\nResumo do treino")
        print("----------------")
        for line in build_execution_summary(
            config=config,
            split_date=split_date,
            train_df=train_df,
            test_df=test_df,
            dropped_columns=dropped_columns,
            train_groups=train_groups,
            test_groups=test_groups,
            metrics=metrics,
        ):
            print(line)

        print("\nArquivos salvos")
        print("--------------")
        print(f"Modelo: {artifact_paths['model_path']}")
        print(f"Imputer: {artifact_paths['imputer_path']}")
        print(f"Features: {artifact_paths['features_path']}")
        print(f"Métricas: {artifact_paths['metrics_path']}")
        print(f"Predições: {predictions_path}")
        print(f"Importâncias: {importance_path}")

    return {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_rank": y_train_rank,
        "y_test_rank": y_test_rank,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "artifact_paths": {
            **artifact_paths,
            "predictions_path": predictions_path,
            "importance_path": importance_path,
        },
    }


def parse_args() -> TrainConfig:
    """Lê argumentos de linha de comando e constrói a configuração do treino."""
    parser = argparse.ArgumentParser(
        description="Treina o modelo LightGBM Ranker com split temporal e salva artefatos."
    )

    parser.add_argument("--target-column", default="target_63d")
    parser.add_argument("--rank-label-buckets", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-date", type=str, default=None)
    parser.add_argument("--test-end-date", type=str, default=None)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--model-filename", default="lightgbm_ranker.joblib")
    parser.add_argument("--imputer-filename", default="imputer.joblib")
    parser.add_argument("--feature-names-filename", default="feature_names.json")
    parser.add_argument("--metrics-filename", default="metrics.json")
    parser.add_argument("--predictions-filename", default="test_predictions.csv")
    parser.add_argument("--feature-importance-filename", default="feature_importance.csv")

    parser.add_argument("--quiet", action="store_true", help="Reduz saída no terminal.")

    args = parser.parse_args()

    return TrainConfig(
        target_column=args.target_column,
        rank_label_buckets=args.rank_label_buckets,
        train_ratio=args.train_ratio,
        split_date=args.split_date,
        test_end_date=args.test_end_date,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        model_filename=args.model_filename,
        imputer_filename=args.imputer_filename,
        feature_names_filename=args.feature_names_filename,
        metrics_filename=args.metrics_filename,
        predictions_filename=args.predictions_filename,
        feature_importance_filename=args.feature_importance_filename,
        verbose=not args.quiet,
    )


def main() -> None:
    """Ponto de entrada do script."""
    config = parse_args()
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
