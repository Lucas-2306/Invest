from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")
MODEL_REPORT_PATH = REPORT_DIR / "test_predictions.csv"
BACKTEST_DIR = REPORT_DIR / "backtest"

INITIAL_CAPITAL = 100000.0
TOP_N = 5
BOTTOM_N = 5
TRANSACTION_COST_PER_SIDE = 0.001
REBALANCE_EVERY_N_DAYS = 5


def load_predictions() -> pd.DataFrame:
    if not MODEL_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {MODEL_REPORT_PATH}. Rode antes o train_model.py."
        )

    df = pd.read_csv(MODEL_REPORT_PATH, parse_dates=["trade_date"])

    required_cols = {
        "trade_date",
        "symbol",
        "target_5d",
        "prediction",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em test_predictions.csv: {sorted(missing)}")

    return df


def select_rebalance_dates(preds: pd.DataFrame, step: int) -> list[pd.Timestamp]:
    unique_dates = sorted(preds["trade_date"].drop_duplicates())
    return unique_dates[::step]


def select_positions(day_df: pd.DataFrame, top_n: int, bottom_n: int) -> pd.DataFrame:
    day_df = day_df.sort_values("prediction", ascending=False).copy()

    longs = day_df.head(top_n).copy()
    longs["side"] = "long"
    longs["weight"] = 1.0 / (top_n + bottom_n) if (top_n + bottom_n) > 0 else 0.0
    longs["position_return"] = longs["target_5d"]

    shorts = day_df.tail(bottom_n).copy()
    shorts["side"] = "short"
    shorts["weight"] = 1.0 / (top_n + bottom_n) if (top_n + bottom_n) > 0 else 0.0
    shorts["position_return"] = -shorts["target_5d"]

    positions = pd.concat([longs, shorts], ignore_index=True)
    return positions


def compute_period_return(
    positions: pd.DataFrame,
    transaction_cost_per_side: float,
) -> float:
    if positions.empty:
        return 0.0

    gross_return = float((positions["weight"] * positions["position_return"]).sum())

    total_positions = len(positions)
    average_weight = positions["weight"].mean()
    cost = total_positions * transaction_cost_per_side * average_weight * 2.0

    net_return = gross_return - cost
    return float(net_return)


def run_backtest(
    preds: pd.DataFrame,
    initial_capital: float,
    top_n: int,
    bottom_n: int,
    transaction_cost_per_side: float,
    rebalance_every_n_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    preds = preds.sort_values(["trade_date", "prediction"], ascending=[True, False]).copy()

    rebalance_dates = select_rebalance_dates(preds, step=rebalance_every_n_days)

    daily_rows: list[dict] = []
    trade_rows: list[dict] = []

    capital = initial_capital
    peak_capital = initial_capital

    for trade_date in rebalance_dates:
        day_df = preds[preds["trade_date"] == trade_date].copy()
        positions = select_positions(day_df, top_n=top_n, bottom_n=bottom_n)

        period_return = compute_period_return(
            positions=positions,
            transaction_cost_per_side=transaction_cost_per_side,
        )

        start_capital = capital
        capital = capital * (1.0 + period_return)
        peak_capital = max(peak_capital, capital)
        drawdown = (capital / peak_capital) - 1.0

        daily_rows.append(
            {
                "trade_date": trade_date,
                "start_capital": start_capital,
                "period_return_5d": period_return,
                "end_capital": capital,
                "drawdown": drawdown,
                "num_positions": len(positions),
            }
        )

        if not positions.empty:
            positions = positions.copy()
            positions["trade_date"] = trade_date
            positions["start_capital"] = start_capital
            positions["period_return_5d"] = period_return
            trade_rows.extend(positions.to_dict(orient="records"))

    equity_curve = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)

    if equity_curve.empty:
        raise ValueError("Backtest vazio. Não há dados no período de teste.")

    total_return = (equity_curve["end_capital"].iloc[-1] / initial_capital) - 1.0
    avg_period_return = equity_curve["period_return_5d"].mean()
    period_vol = equity_curve["period_return_5d"].std()

    sharpe = np.nan
    if period_vol and period_vol > 0:
        sharpe = (avg_period_return / period_vol) * np.sqrt(252 / 5)

    max_drawdown = equity_curve["drawdown"].min()
    positive_periods = (equity_curve["period_return_5d"] > 0).mean()

    metrics = {
        "initial_capital": float(initial_capital),
        "final_capital": float(equity_curve["end_capital"].iloc[-1]),
        "total_return": float(total_return),
        "avg_period_return_5d": float(avg_period_return),
        "period_volatility_5d": float(period_vol) if pd.notna(period_vol) else None,
        "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "max_drawdown": float(max_drawdown),
        "positive_period_ratio": float(positive_periods),
        "num_periods": int(len(equity_curve)),
        "top_n_longs": int(top_n),
        "top_n_shorts": int(bottom_n),
        "transaction_cost_per_side": float(transaction_cost_per_side),
        "rebalance_every_n_days": int(rebalance_every_n_days),
    }

    return equity_curve, trades, metrics


def save_outputs(equity_curve: pd.DataFrame, trades: pd.DataFrame, metrics: dict) -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    equity_path = BACKTEST_DIR / "equity_curve.csv"
    trades_path = BACKTEST_DIR / "trades.csv"
    metrics_path = BACKTEST_DIR / "backtest_metrics.json"

    equity_curve.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Curva de patrimônio salva em: {equity_path}")
    print(f"Trades salvos em: {trades_path}")
    print(f"Métricas salvas em: {metrics_path}")


def print_summary(metrics: dict, equity_curve: pd.DataFrame) -> None:
    print("\nResumo do backtest")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nÚltimos períodos da curva")
    print("-------------------------")
    print(equity_curve.tail(10).to_string(index=False))


def main() -> None:
    preds = load_predictions()

    equity_curve, trades, metrics = run_backtest(
        preds=preds,
        initial_capital=INITIAL_CAPITAL,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        transaction_cost_per_side=TRANSACTION_COST_PER_SIDE,
        rebalance_every_n_days=REBALANCE_EVERY_N_DAYS,
    )

    save_outputs(equity_curve, trades, metrics)
    print_summary(metrics, equity_curve)


if __name__ == "__main__":
    main()