from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd

try:
    from .backtest_config import BACKTEST_DIR, BacktestConfig, StrategyConfig, get_strategy_config, parse_args
    from .backtest_data import load_predictions, load_prices, prepare_price_panel
    from .backtest_metrics import analyze_by_year, compute_metrics
    from .backtest_portfolio import (
        apply_transaction_costs,
        build_daily_equity_curve,
        compute_turnover,
        normalize_active_lots_exposure,
    )
    from .backtest_trades import build_trade_book, expand_trades_to_daily_positions
except ImportError:
    from backtest_config import BACKTEST_DIR, BacktestConfig, StrategyConfig, get_strategy_config, parse_args
    from backtest_data import load_predictions, load_prices, prepare_price_panel
    from backtest_metrics import analyze_by_year, compute_metrics
    from backtest_portfolio import (
        apply_transaction_costs,
        build_daily_equity_curve,
        compute_turnover,
        normalize_active_lots_exposure,
    )
    from backtest_trades import build_trade_book, expand_trades_to_daily_positions


def run_single_backtest(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig,
    strategy: StrategyConfig,
) -> dict[str, Any]:
    """Executa o pipeline completo e retorna todos os artefatos em memória."""
    trades = build_trade_book(preds=preds, prices=prices, config=config, strategy=strategy)
    daily_positions = expand_trades_to_daily_positions(trades, prices)

    if strategy.normalize_active_lots:
        daily_positions = normalize_active_lots_exposure(daily_positions)

    daily_positions = apply_transaction_costs(daily_positions, trades, config)
    turnover_df = compute_turnover(trades)
    equity_curve = build_daily_equity_curve(
        daily_positions=daily_positions,
        initial_capital=config.initial_capital,
        config=config,
    )
    metrics = compute_metrics(equity_curve, config, strategy, turnover_df)

    return {
        "trades": trades,
        "daily_positions": daily_positions,
        "turnover_df": turnover_df,
        "equity_curve": equity_curve,
        "metrics": metrics,
    }

def run_cost_sensitivity(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    base_config: BacktestConfig,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    """Executa versões alternativas mudando apenas o custo por lado."""
    rows: list[dict[str, Any]] = []

    for cost in base_config.cost_sensitivity_values:
        scenario_config = BacktestConfig(**{**asdict(base_config), "transaction_cost_per_side": float(cost)})
        result = run_single_backtest(preds=preds, prices=prices, config=scenario_config, strategy=strategy)
        rows.append(
            {
                "transaction_cost_per_side": float(cost),
                "annualized_sharpe": result["metrics"]["annualized_sharpe"],
                "total_return": result["metrics"]["total_return"],
            }
        )

    return pd.DataFrame(rows)

def save_outputs(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    daily_positions: pd.DataFrame,
    turnover_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    """Persiste os principais artefatos do backtest em disco."""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    equity_path = BACKTEST_DIR / "equity_curve.csv"
    trades_path = BACKTEST_DIR / "trade_book.csv"
    daily_positions_path = BACKTEST_DIR / "daily_positions.csv"
    turnover_path = BACKTEST_DIR / "turnover.csv"
    metrics_path = BACKTEST_DIR / "backtest_metrics.json"

    equity_curve.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)
    daily_positions.to_csv(daily_positions_path, index=False)
    turnover_df.to_csv(turnover_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Arquivos salvos em artifacts/reports/backtest/")

def print_summary(metrics: dict[str, Any], equity_curve: pd.DataFrame) -> None:
    """Imprime apenas o resumo realmente útil no terminal."""
    print("\nResumo do backtest")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nÚltimos dias da curva")
    print("---------------------")
    print(equity_curve.tail(10).to_string(index=False))

def print_yearly_analysis(yearly: pd.DataFrame) -> None:
    """Imprime a tabela anual quando habilitada."""
    print("\nPerformance por ano")
    print("-------------------")
    print(yearly)

def print_cost_sensitivity_table(cost_df: pd.DataFrame) -> None:
    """Imprime a análise de sensibilidade a custo de forma compacta."""
    if cost_df.empty:
        return

    print("\nSensibilidade a custo")
    print("---------------------")
    print(cost_df.to_string(index=False))

def main() -> None:
    config = parse_args()
    strategy = get_strategy_config(config)

    preds = load_predictions()
    start_date = preds["trade_date"].min()
    end_date = preds["trade_date"].max() + pd.Timedelta(days=config.holding_days * 2)

    prices = load_prices(
        symbols=preds["symbol"].unique().tolist(),
        start_date=start_date,
        end_date=end_date,
    )
    prices = prepare_price_panel(prices)

    result = run_single_backtest(preds=preds, prices=prices, config=config, strategy=strategy)

    save_outputs(
        equity_curve=result["equity_curve"],
        trades=result["trades"],
        daily_positions=result["daily_positions"],
        turnover_df=result["turnover_df"],
        metrics=result["metrics"],
    )
    print_summary(result["metrics"], result["equity_curve"])

    if config.run_yearly_analysis:
        yearly = analyze_by_year(result["equity_curve"])
        print_yearly_analysis(yearly)

    if config.run_cost_sensitivity:
        cost_df = run_cost_sensitivity(preds=preds, prices=prices, base_config=config, strategy=strategy)
        print_cost_sensitivity_table(cost_df)


if __name__ == "__main__":
    main()
