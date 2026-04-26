from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORT_DIR = ARTIFACTS_DIR / "reports"
BACKTEST_DIR = REPORT_DIR / "backtest"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"

TRAIN_METRICS_PATH = REPORT_DIR / "metrics.json"
BACKTEST_METRICS_PATH = BACKTEST_DIR / "backtest_metrics.json"
WALKFORWARD_RESULTS_PATH = EXPERIMENTS_DIR / "walkforward_results.csv"


@dataclass(frozen=True)
class WalkforwardConfig:
    split_dates: list[str]
    test_end_dates: list[str | None]

    target_column: str = "target_63d"
    rank_label_buckets: int = 100
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    random_state: int = 42

    portfolio_mode: str = "long_short"
    backtest_strategy: str = "block"
    top_n: int = 12
    bottom_n: int = 10
    top_n_exit: int = 60
    bottom_n_exit: int = 40
    long_quantile: float = 0.9
    short_quantile: float = 0.1
    transaction_cost_per_side: float = 0.001
    slippage: float = 0.001
    holding_days: int = 63
    min_side_weight: float = 0.2
    max_side_weight: float = 0.8
    long_min_signal_strength: float = 0.5
    short_min_signal_strength: float = 0.4
    target_signal_strength: float = 2.5

    use_dynamic_signal_filter: bool = False
    use_macro_regime_scaling: bool = False
    use_ic_exposure_scaling: bool = True

    use_performance_scaling: bool = False
    performance_lookback: int = 20
    performance_soft_threshold: float = 0.0
    performance_hard_threshold: float = -0.001
    performance_soft_scale: float = 0.5
    performance_hard_scale: float = 0.0


def ensure_dirs() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_command(config: WalkforwardConfig, split_date: str, test_end_date: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "jobs.training.train_model",
        "--target-column", config.target_column,
        "--rank-label-buckets", str(config.rank_label_buckets),
        "--split-date", split_date,
        "--n-estimators", str(config.n_estimators),
        "--max-depth", str(config.max_depth),
        "--learning-rate", str(config.learning_rate),
        "--random-state", str(config.random_state),
        "--quiet",
    ]

    if test_end_date is not None:
        cmd.extend(["--test-end-date", test_end_date])

    return cmd


def backtest_command(config: WalkforwardConfig) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "jobs.training.backtest",
        "--portfolio-mode", config.portfolio_mode,
        "--backtest-strategy", config.backtest_strategy,
        "--top-n", str(config.top_n),
        "--bottom-n", str(config.bottom_n),
        "--top-n-exit", str(config.top_n_exit),
        "--bottom-n-exit", str(config.bottom_n_exit),
        "--long-quantile", str(config.long_quantile),
        "--short-quantile", str(config.short_quantile),
        "--transaction-cost-per-side", str(config.transaction_cost_per_side),
        "--slippage", str(config.slippage),
        "--holding-days", str(config.holding_days),
        "--min-side-weight", str(config.min_side_weight),
        "--max-side-weight", str(config.max_side_weight),
        "--long-min-signal-strength", str(config.long_min_signal_strength),
        "--short-min-signal-strength", str(config.short_min_signal_strength),
        "--target-signal-strength", str(config.target_signal_strength),
        "--skip-yearly-analysis",
        "--skip-cost-sensitivity",
    ]

    if config.use_dynamic_signal_filter:
        cmd.append("--dynamic-signal-filter")

    if config.use_macro_regime_scaling:
        cmd.append("--macro-regime-scaling")

    if not config.use_ic_exposure_scaling:
        cmd.append("--no-ic-exposure-scaling")

    if config.use_performance_scaling:
        cmd.append("--performance-scaling")

    cmd.extend([
        "--performance-lookback", str(config.performance_lookback),
        "--performance-soft-threshold", str(config.performance_soft_threshold),
        "--performance-hard-threshold", str(config.performance_hard_threshold),
        "--performance-soft-scale", str(config.performance_soft_scale),
        "--performance-hard-scale", str(config.performance_hard_scale),
    ])

    return cmd


def build_row(
    split_date: str,
    test_end_date: str | None,
    config: WalkforwardConfig,
    train_metrics: dict[str, Any],
    backtest_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split_date": split_date,
        "test_end_date": test_end_date,

        "target_column": config.target_column,
        "rank_label_buckets": config.rank_label_buckets,
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "random_state": config.random_state,

        "portfolio_mode": config.portfolio_mode,
        "backtest_strategy": config.backtest_strategy,
        "top_n": config.top_n,
        "bottom_n": config.bottom_n,
        "top_n_exit": config.top_n_exit,
        "bottom_n_exit": config.bottom_n_exit,
        "long_quantile": config.long_quantile,
        "short_quantile": config.short_quantile,
        "transaction_cost_per_side": config.transaction_cost_per_side,
        "slippage": config.slippage,
        "holding_days": config.holding_days,
        "min_side_weight": config.min_side_weight,
        "max_side_weight": config.max_side_weight,
        "long_min_signal_strength": config.long_min_signal_strength,
        "short_min_signal_strength": config.short_min_signal_strength,
        "target_signal_strength": config.target_signal_strength,

        "train_spearman_corr": train_metrics.get("train_spearman_corr"),
        "test_spearman_corr": train_metrics.get("test_spearman_corr"),
        "train_daily_ic_mean": train_metrics.get("train_daily_ic_mean"),
        "test_daily_ic_mean": train_metrics.get("test_daily_ic_mean"),
        "train_rows": train_metrics.get("train_rows"),
        "test_rows": train_metrics.get("test_rows"),

        "final_capital": backtest_metrics.get("final_capital"),
        "total_return": backtest_metrics.get("total_return"),
        "annualized_sharpe": backtest_metrics.get("annualized_sharpe"),
        "max_drawdown": backtest_metrics.get("max_drawdown"),
        "avg_daily_return": backtest_metrics.get("avg_daily_return"),
        "daily_volatility": backtest_metrics.get("daily_volatility"),
        "positive_day_ratio": backtest_metrics.get("positive_day_ratio"),
        "avg_gross_exposure_effective": backtest_metrics.get("avg_gross_exposure_effective"),
        "avg_net_exposure_effective": backtest_metrics.get("avg_net_exposure_effective"),
        "avg_turnover": backtest_metrics.get("avg_turnover"),
        "median_turnover": backtest_metrics.get("median_turnover"),
        "max_turnover": backtest_metrics.get("max_turnover"),
        "long_avg_daily_return": backtest_metrics.get("long_avg_daily_return"),
        "short_avg_daily_return": backtest_metrics.get("short_avg_daily_return"),
        "long_total_contribution": backtest_metrics.get("long_total_contribution"),
        "short_total_contribution": backtest_metrics.get("short_total_contribution"),
        "num_days": backtest_metrics.get("num_days"),
    }


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    show_cols = [
        "split_date",
        "test_end_date",
        "annualized_sharpe",
        "total_return",
        "max_drawdown",
        "test_daily_ic_mean",
        "avg_gross_exposure_effective",
        "avg_turnover",
        "num_days",
    ]
    available_cols = [c for c in show_cols if c in df.columns]

    print("\nResumo walk-forward")
    print("-------------------")
    print(df[available_cols].to_string(index=False))

    print("\nMédias")
    print("------")
    summary = {
        "mean_sharpe": df["annualized_sharpe"].mean(),
        "median_sharpe": df["annualized_sharpe"].median(),
        "mean_return": df["total_return"].mean(),
        "median_return": df["total_return"].median(),
        "mean_test_ic": df["test_daily_ic_mean"].mean(),
        "mean_max_drawdown": df["max_drawdown"].mean(),
        "mean_exposure": df["avg_gross_exposure_effective"].mean(),
        "mean_turnover": df["avg_turnover"].mean(),
    }
    for k, v in summary.items():
        print(f"{k}: {v}")


def main() -> None:
    ensure_dirs()

    # Cada split treina até split_date e testa daí em diante.
    # Se quiser limitar a janela de teste, preencha test_end_dates.
    config = WalkforwardConfig(
        split_dates=[
            "2019-01-01",
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",
            "2023-01-01",
            "2023-11-03",
        ],
        test_end_dates = [
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",
            "2023-01-01",
            "2024-01-01",
            None,
        ],
        target_column="target_63d",
        rank_label_buckets=100,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,

        portfolio_mode="long_short",
        backtest_strategy="block",
        top_n=10,
        bottom_n=6,
        top_n_exit=20,
        bottom_n_exit=14,
        long_quantile=0.9,
        short_quantile=0.05,
        transaction_cost_per_side=0.001,
        slippage=0.001,
        holding_days=63,
        min_side_weight=0.2,
        max_side_weight=0.8,
        long_min_signal_strength=0.5,
        short_min_signal_strength=0.4,
        target_signal_strength=2.5,

        use_dynamic_signal_filter=False,
        use_macro_regime_scaling=False,
        use_ic_exposure_scaling=False,

        use_performance_scaling = False,
        performance_lookback = 20,
        performance_soft_threshold = 0.0,
        performance_hard_threshold = -0.001,
        performance_soft_scale = 0.7,
        performance_hard_scale = 0.2,
    )

    if len(config.split_dates) != len(config.test_end_dates):
        raise ValueError("split_dates e test_end_dates devem ter o mesmo tamanho.")

    rows: list[dict[str, Any]] = []

    for i, (split_date, test_end_date) in enumerate(zip(config.split_dates, config.test_end_dates), start=1):
        print(f"\n===== WALK-FORWARD {i}/{len(config.split_dates)} =====")
        print(f"split_date={split_date} | test_end_date={test_end_date}")

        run_command(train_command(config, split_date, test_end_date))
        train_metrics = load_json(TRAIN_METRICS_PATH)

        run_command(backtest_command(config))
        backtest_metrics = load_json(BACKTEST_METRICS_PATH)

        row = build_row(
            split_date=split_date,
            test_end_date=test_end_date,
            config=config,
            train_metrics=train_metrics,
            backtest_metrics=backtest_metrics,
        )
        rows.append(row)

        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(WALKFORWARD_RESULTS_PATH, index=False)

        print(
            f"OK | split={split_date} | "
            f"Sharpe={backtest_metrics.get('annualized_sharpe')} | "
            f"Return={backtest_metrics.get('total_return')} | "
            f"IC={train_metrics.get('test_daily_ic_mean')}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(WALKFORWARD_RESULTS_PATH, index=False)

    print(f"\nArquivo salvo em: {WALKFORWARD_RESULTS_PATH}")
    print_summary_table(df)


if __name__ == "__main__":
    main()