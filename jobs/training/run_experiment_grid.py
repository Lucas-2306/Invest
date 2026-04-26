from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORT_DIR = ARTIFACTS_DIR / "reports"
BACKTEST_DIR = REPORT_DIR / "backtest"
MODEL_DIR = ARTIFACTS_DIR / "models"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"


@dataclass(frozen=True)
class ExperimentPaths:
    train_metrics_path: Path
    backtest_metrics_path: Path
    summary_csv_path: Path
    summary_jsonl_path: Path


def ensure_dirs() -> ExperimentPaths:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        train_metrics_path=REPORT_DIR / "metrics.json",
        backtest_metrics_path=BACKTEST_DIR / "backtest_metrics.json",
        summary_csv_path=EXPERIMENTS_DIR / "grid_results.csv",
        summary_jsonl_path=EXPERIMENTS_DIR / "grid_results.jsonl",
    )


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(value: Any) -> str:
    text = str(value)
    return (
        text.replace("/", "-")
        .replace(" ", "_")
        .replace(".", "p")
        .replace(":", "-")
    )


def build_run_name(
    train_params: dict[str, Any],
    backtest_params: dict[str, Any],
) -> str:
    parts = [
        f"target_{safe_name(train_params['target_column'])}",
        f"split_{safe_name(train_params['split_date'])}",
        f"testend_{safe_name(train_params.get('test_end_date'))}",
        f"nest_{safe_name(train_params['n_estimators'])}",
        f"depth_{safe_name(train_params['max_depth'])}",
        f"lr_{safe_name(train_params['learning_rate'])}",
        f"mode_{safe_name(backtest_params['portfolio_mode'])}",
        f"top_{safe_name(backtest_params['top_n'])}",
        f"exit_{safe_name(backtest_params['top_n_exit'])}",
        f"lq_{safe_name(backtest_params['long_quantile'])}",
        f"sq_{safe_name(backtest_params['short_quantile'])}",
        f"lonminsig_{safe_name(backtest_params['long_min_signal_strength'])}",
        f"srtminsig_{safe_name(backtest_params['short_min_signal_strength'])}",
        f"targsig_{safe_name(backtest_params['target_signal_strength'])}",
        f"minside_{safe_name(backtest_params['min_side_weight'])}",
        f"maxside_{safe_name(backtest_params['max_side_weight'])}",
        f"cost_{safe_name(backtest_params['transaction_cost_per_side'])}",
        f"slip_{safe_name(backtest_params['slippage'])}",
        f"hold_{safe_name(backtest_params['holding_days'])}",
    ]
    return "__".join(parts)


def dict_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def train_command(params: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "jobs.training.train_model",
        "--target-column", str(params["target_column"]),
        "--rank-label-buckets", str(params["rank_label_buckets"]),
        "--split-date", str(params["split_date"]),
        "--n-estimators", str(params["n_estimators"]),
        "--max-depth", str(params["max_depth"]),
        "--learning-rate", str(params["learning_rate"]),
        "--random-state", str(params["random_state"]),
        "--quiet",
    ]

    if params.get("test_end_date") is not None:
        cmd.extend(["--test-end-date", str(params["test_end_date"])])

    return cmd


def backtest_command(params: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "jobs.training.backtest",
        "--portfolio-mode", str(params["portfolio_mode"]),
        "--backtest-strategy", str(params["backtest_strategy"]),
        "--top-n", str(params["top_n"]),
        "--bottom-n", str(params["bottom_n"]),
        "--top-n-exit", str(params["top_n_exit"]),
        "--bottom-n-exit", str(params["bottom_n_exit"]),
        "--long-quantile", str(params["long_quantile"]),
        "--short-quantile", str(params["short_quantile"]),
        "--transaction-cost-per-side", str(params["transaction_cost_per_side"]),
        "--slippage", str(params["slippage"]),
        "--holding-days", str(params["holding_days"]),
        "--min-side-weight", str(params["min_side_weight"]),
        "--max-side-weight", str(params["max_side_weight"]),
        "--long-min-signal-strength", str(params["long_min_signal_strength"]),
        "--short-min-signal-strength", str(params["short_min_signal_strength"]),
        "--target-signal-strength", str(params["target_signal_strength"]),
        "--skip-yearly-analysis",
        "--skip-cost-sensitivity",
        
    ]

    if params.get("use_dynamic_signal_filter"):
        cmd.append("--dynamic-signal-filter")

    if params.get("use_macro_regime_scaling"):
        cmd.append("--macro-regime-scaling")

    if not params.get("use_ic_exposure_scaling"):
        cmd.append("--no-ic-exposure-scaling")

    return cmd


def flatten_result(
    run_name: str,
    train_params: dict[str, Any],
    backtest_params: dict[str, Any],
    train_metrics: dict[str, Any],
    backtest_metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"run_name": run_name}

    for k, v in train_params.items():
        row[f"train__{k}"] = v

    for k, v in backtest_params.items():
        row[f"backtest__{k}"] = v

    for k, v in train_metrics.items():
        row[f"train_metric__{k}"] = v

    for k, v in backtest_metrics.items():
        row[f"backtest_metric__{k}"] = v

    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=[
            "backtest_metric__annualized_sharpe",
            "backtest_metric__total_return",
            "train_metric__test_daily_ic_mean",
        ],
        ascending=False,
        na_position="last",
    )
    df.to_csv(path, index=False)


def main() -> None:
    paths = ensure_dirs()

    train_grid = {
        "target_column": ["target_63d"],
        "rank_label_buckets": [100],
        "split_date": ["2023-11-03"],
        "test_end_date": [None],
        "n_estimators": [300],
        "max_depth": [4],
        "learning_rate": [0.05],
        "random_state": [42],
    }

    backtest_grid = {
        "portfolio_mode": ["long_short"],
        "backtest_strategy": ["block"],

        # long não importa no short_only, mas mantém preenchido
        "top_n": [8, 10, 12],
        "bottom_n": [4, 6, 8],
        "top_n_exit": [20, 40, 60],
        "bottom_n_exit": [10, 14, 20],

        # =========================
        # ENTRY FILTER
        # =========================
        "long_quantile": [0.9],
        "short_quantile": [0.05],

        # =========================
        # COST MODEL
        # =========================
        "transaction_cost_per_side": [0.001],
        "slippage": [0.001],

        # =========================
        # HOLDING
        # =========================
        "holding_days": [63],

        # =========================
        # SIDE BALANCE
        # irrelevante no short_only, mantém fixo
        # =========================
        "min_side_weight": [0.2],
        "max_side_weight": [0.8],

        # =========================
        # SIGNAL FILTER
        # =========================
        "long_min_signal_strength": [0.5],
        "short_min_signal_strength": [0.4],
        "target_signal_strength": [2.5],

        # =========================
        # FLAGS
        # =========================
        "use_dynamic_signal_filter": [False],
        "use_macro_regime_scaling": [False],
        "use_ic_exposure_scaling": [True],

        "use_performance_scaling": [False],
        "performance_lookback": [20],
        "performance_hard_threshold": [-0.001],
        "performance_soft_threshold": [0.0],
        "performance_soft_scale": [0.5],
        "performance_hard_scale": [0.0],
    }
    train_combos = dict_product(train_grid)
    backtest_combos = dict_product(backtest_grid)

    all_rows: list[dict[str, Any]] = []
    run_counter = 0
    total_runs = len(train_combos) * len(backtest_combos)

    for train_params in train_combos:
        run_command(train_command(train_params))
        train_metrics = load_json(paths.train_metrics_path)

        for backtest_params in backtest_combos:
            run_counter += 1
            print(f"\n===== RUN {run_counter}/{total_runs} =====")

            run_name = build_run_name(train_params, backtest_params)

            try:
                run_command(backtest_command(backtest_params))
                backtest_metrics = load_json(paths.backtest_metrics_path)

                row = flatten_result(
                    run_name=run_name,
                    train_params=train_params,
                    backtest_params=backtest_params,
                    train_metrics=train_metrics,
                    backtest_metrics=backtest_metrics,
                )
                all_rows.append(row)
                append_jsonl(paths.summary_jsonl_path, row)

                save_csv(paths.summary_csv_path, all_rows)

                print(
                    f"OK | Sharpe={backtest_metrics.get('annualized_sharpe')} | "
                    f"Return={backtest_metrics.get('total_return')} | "
                    f"IC={train_metrics.get('test_daily_ic_mean')}"
                )

            except Exception as exc:
                error_row = {
                    "run_name": run_name,
                    "status": "error",
                    "error": str(exc),
                    **{f"train__{k}": v for k, v in train_params.items()},
                    **{f"backtest__{k}": v for k, v in backtest_params.items()},
                }
                all_rows.append(error_row)
                append_jsonl(paths.summary_jsonl_path, error_row)
                save_csv(paths.summary_csv_path, all_rows)
                print(f"ERRO em {run_name}: {exc}")

    print("\nConcluído.")
    print(f"CSV consolidado: {paths.summary_csv_path}")
    print(f"JSONL consolidado: {paths.summary_jsonl_path}")


if __name__ == "__main__":
    main()