from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.training_db import engine


BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "artifacts" / "reports"
BACKTEST_DIR = REPORT_DIR / "backtest"
MODEL_REPORT_PATH = REPORT_DIR / "test_predictions.csv"

# Ajuste esta constante se o nome real da sua tabela de preços for diferente.
PRICE_TABLE = "market_data.daily_prices"

INITIAL_CAPITAL = 100000.0
TOP_N = 5
BOTTOM_N = 5
TRANSACTION_COST_PER_SIDE = 0.001
SLIPPAGE = 0.001
HOLDING_DAYS = 5
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
        "prediction",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em test_predictions.csv: {sorted(missing)}")

    df = df.sort_values(["trade_date", "prediction"], ascending=[True, False]).reset_index(drop=True)
    return df


def load_prices(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    symbols_sql = ", ".join([f"'{s}'" for s in sorted(set(symbols))])

    query = f"""
        SELECT
            s.symbol,
            p.trade_date,
            p.open_price,
            p.close_price,
            p.adjusted_close_price
        FROM market_data.daily_prices p
        JOIN market_data.symbols s
          ON s.id = p.symbol_id
        WHERE s.symbol IN ({symbols_sql})
          AND p.trade_date >= '{start_date.date()}'
          AND p.trade_date <= '{end_date.date()}'
        ORDER BY s.symbol, p.trade_date
    """

    df = pd.read_sql_query(query, engine, parse_dates=["trade_date"])

    if df.empty:
        raise ValueError("Nenhum preço encontrado para o período do backtest.")

    return df


def prepare_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    for col in ["open_price", "close_price", "adjusted_close_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ref_price"] = df["adjusted_close_price"].fillna(df["close_price"])

    prev_ref = df.groupby("symbol")["ref_price"].shift(1)
    raw_ret = df["ref_price"] / prev_ref

    valid_ret = (
        prev_ref.notna()
        & (prev_ref > 0)
        & raw_ret.notna()
        & np.isfinite(raw_ret)
        & (raw_ret > 0.05)
        & (raw_ret < 20.0)
    )

    df["daily_return"] = np.nan
    df.loc[valid_ret, "daily_return"] = raw_ret.loc[valid_ret] - 1.0

    return df


def select_rebalance_dates(preds: pd.DataFrame, step: int) -> list[pd.Timestamp]:
    unique_dates = sorted(preds["trade_date"].drop_duplicates())
    return unique_dates[::step]


def select_positions(day_df: pd.DataFrame, top_n: int, bottom_n: int, mode: str = "long_short") -> pd.DataFrame:
    day_df = day_df.sort_values("prediction", ascending=False).copy()

    if mode == "long_only":
        longs = day_df.head(top_n).copy()
        shorts = pd.DataFrame(columns=day_df.columns)
    elif mode == "short_only":
        longs = pd.DataFrame(columns=day_df.columns)
        shorts = day_df.tail(bottom_n).copy()
    else:
        longs = day_df.head(top_n).copy()
        shorts = day_df.tail(bottom_n).copy()

    if not longs.empty:
        longs["side"] = "long"
    if not shorts.empty:
        shorts["side"] = "short"

    positions = pd.concat([longs, shorts], ignore_index=True)

    total_positions = len(positions)
    if total_positions == 0:
        positions["weight"] = 0.0
    else:
        positions["weight"] = 1.0 / total_positions

    return positions


def get_calendar(prices: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(prices["trade_date"]).drop_duplicates())


def get_trade_window(
    signal_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    holding_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]] | None:
    try:
        signal_idx = calendar.index(signal_date)
    except ValueError:
        return None

    # sinal em t, entrada no close de t+1, saída no close de t+6
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 1 + holding_days

    if exit_idx >= len(calendar):
        return None

    entry_date = calendar[entry_idx]
    exit_date = calendar[exit_idx]

    # retornos que capturam close-to-close de t+1 até t+6:
    # datas de pnl: t+2, t+3, t+4, t+5, t+6
    active_return_dates = calendar[entry_idx + 1 : exit_idx + 1]

    if len(active_return_dates) != holding_days:
        return None

    return entry_date, exit_date, active_return_dates


def build_trade_book(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    rebalance_every_n_days: int,
    holding_days: int,
    mode: str = "long_short",
) -> pd.DataFrame:
    rebalance_dates = select_rebalance_dates(preds, step=rebalance_every_n_days)
    calendar = get_calendar(prices)

    trade_rows: list[dict] = []
    trade_id = 0

    for signal_date in rebalance_dates:
        window = get_trade_window(
            signal_date=signal_date,
            calendar=calendar,
            holding_days=holding_days,
        )
        if window is None:
            continue

        entry_date, exit_date, active_return_dates = window

        day_df = preds[preds["trade_date"] == signal_date].copy()
        positions = select_positions(day_df, top_n=top_n, bottom_n=bottom_n, mode=mode)

        if positions.empty:
            continue

        for _, row in positions.iterrows():
            trade_rows.append(
                {
                    "trade_id": trade_id,
                    "signal_date": pd.to_datetime(signal_date),
                    "entry_date": pd.to_datetime(entry_date),
                    "exit_date": pd.to_datetime(exit_date),
                    "symbol": str(row["symbol"]),
                    "side": str(row["side"]),
                    "weight": float(row["weight"]),
                    "prediction": float(row["prediction"]),
                    "active_return_dates": json.dumps([d.strftime("%Y-%m-%d") for d in active_return_dates]),
                }
            )
            trade_id += 1

    trades = pd.DataFrame(trade_rows)

    if trades.empty:
        raise ValueError("Nenhum trade foi gerado no backtest.")

    return trades


def expand_trades_to_daily_positions(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    price_panel = prices[["symbol", "trade_date", "daily_return"]].copy()

    expanded_rows: list[dict] = []

    for _, trade in trades.iterrows():
        active_dates = json.loads(trade["active_return_dates"])

        for dt in active_dates:
            expanded_rows.append(
                {
                    "trade_id": int(trade["trade_id"]),
                    "trade_date": pd.to_datetime(dt),
                    "symbol": str(trade["symbol"]),
                    "side": str(trade["side"]),
                    "weight": float(trade["weight"]),
                    "signal_date": pd.to_datetime(trade["signal_date"]),
                    "entry_date": pd.to_datetime(trade["entry_date"]),
                    "exit_date": pd.to_datetime(trade["exit_date"]),
                    "prediction": float(trade["prediction"]),
                }
            )

    daily_positions = pd.DataFrame(expanded_rows)

    if daily_positions.empty:
        raise ValueError("Nenhuma posição diária foi expandida.")

    daily_positions = daily_positions.merge(
        price_panel,
        on=["symbol", "trade_date"],
        how="left",
    )

    daily_positions = daily_positions.dropna(subset=["daily_return"]).copy()

    daily_positions["position_return"] = np.where(
        daily_positions["side"] == "long",
        daily_positions["daily_return"],
        -daily_positions["daily_return"],
    )

    daily_positions["position_contribution"] = (
        daily_positions["weight"] * daily_positions["position_return"]
    )

    return daily_positions


def apply_transaction_costs(daily_positions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    daily_positions = daily_positions.copy()

    cost_rows: list[dict] = []

    for _, trade in trades.iterrows():
        weight = float(trade["weight"])

        entry_cost = weight * (TRANSACTION_COST_PER_SIDE + SLIPPAGE)
        exit_cost = weight * TRANSACTION_COST_PER_SIDE

        cost_rows.append(
            {
                "trade_id": int(trade["trade_id"]),
                "trade_date": pd.to_datetime(trade["entry_date"]),
                "symbol": str(trade["symbol"]),
                "side": str(trade["side"]),
                "weight": weight,
                "signal_date": pd.to_datetime(trade["signal_date"]),
                "entry_date": pd.to_datetime(trade["entry_date"]),
                "exit_date": pd.to_datetime(trade["exit_date"]),
                "prediction": float(trade["prediction"]),
                "daily_return": np.nan,
                "position_return": 0.0,
                "position_contribution": -entry_cost,
                "row_type": "entry_cost",
            }
        )

        cost_rows.append(
            {
                "trade_id": int(trade["trade_id"]),
                "trade_date": pd.to_datetime(trade["exit_date"]),
                "symbol": str(trade["symbol"]),
                "side": str(trade["side"]),
                "weight": weight,
                "signal_date": pd.to_datetime(trade["signal_date"]),
                "entry_date": pd.to_datetime(trade["entry_date"]),
                "exit_date": pd.to_datetime(trade["exit_date"]),
                "prediction": float(trade["prediction"]),
                "daily_return": np.nan,
                "position_return": 0.0,
                "position_contribution": -exit_cost,
                "row_type": "exit_cost",
            }
        )

    if "row_type" not in daily_positions.columns:
        daily_positions["row_type"] = "market_return"

    cost_df = pd.DataFrame(cost_rows)
    out = pd.concat([daily_positions, cost_df], ignore_index=True)
    out = out.sort_values(["trade_date", "trade_id", "row_type"]).reset_index(drop=True)

    return out


def build_daily_equity_curve(daily_positions: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    df = daily_positions.copy()

    if "row_type" not in df.columns:
        df["row_type"] = "market_return"

    # 1) Retorno total do dia: inclui mercado + custos
    pnl_by_day = (
        df.groupby("trade_date", as_index=False)
        .agg(
            daily_return=("position_contribution", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # 2) Exposição e contagem: apenas posições de mercado
    market_df = df[df["row_type"] == "market_return"].copy()

    exposure_by_day = (
        market_df.groupby("trade_date", as_index=False)
        .agg(
            num_active_rows=("trade_id", "count"),
            num_active_trades=("trade_id", "nunique"),
            gross_exposure=("weight", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # 3) Junta as duas visões
    grouped = pnl_by_day.merge(
        exposure_by_day,
        on="trade_date",
        how="left",
    )

    grouped["num_active_rows"] = grouped["num_active_rows"].fillna(0).astype(int)
    grouped["num_active_trades"] = grouped["num_active_trades"].fillna(0).astype(int)
    grouped["gross_exposure"] = grouped["gross_exposure"].fillna(0.0)

    rows: list[dict] = []

    capital = initial_capital
    peak_capital = initial_capital

    for _, row in grouped.iterrows():
        start_capital = capital
        daily_return = float(row["daily_return"])
        capital = capital * (1.0 + daily_return)
        peak_capital = max(peak_capital, capital)
        drawdown = (capital / peak_capital) - 1.0

        rows.append(
            {
                "trade_date": pd.to_datetime(row["trade_date"]),
                "start_capital": start_capital,
                "daily_return": daily_return,
                "end_capital": capital,
                "drawdown": drawdown,
                "num_active_rows": int(row["num_active_rows"]),
                "num_active_trades": int(row["num_active_trades"]),
                "gross_exposure": float(row["gross_exposure"]),
            }
        )

    equity_curve = pd.DataFrame(rows)

    if equity_curve.empty:
        raise ValueError("Equity curve vazia.")

    return equity_curve


def compute_metrics(equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    total_return = (equity_curve["end_capital"].iloc[-1] / initial_capital) - 1.0
    avg_daily_return = equity_curve["daily_return"].mean()
    daily_vol = equity_curve["daily_return"].std()

    sharpe = np.nan
    if pd.notna(daily_vol) and daily_vol > 0:
        sharpe = (avg_daily_return / daily_vol) * np.sqrt(252)

    max_drawdown = equity_curve["drawdown"].min()
    positive_days = (equity_curve["daily_return"] > 0).mean()

    metrics = {
        "initial_capital": float(initial_capital),
        "final_capital": float(equity_curve["end_capital"].iloc[-1]),
        "total_return": float(total_return),
        "avg_daily_return": float(avg_daily_return),
        "daily_volatility": float(daily_vol) if pd.notna(daily_vol) else None,
        "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "max_drawdown": float(max_drawdown),
        "positive_day_ratio": float(positive_days),
        "num_days": int(len(equity_curve)),
        "top_n_longs": int(TOP_N),
        "top_n_shorts": int(BOTTOM_N),
        "transaction_cost_per_side": float(TRANSACTION_COST_PER_SIDE),
        "slippage": float(SLIPPAGE),
        "holding_days": int(HOLDING_DAYS),
        "rebalance_every_n_days": int(REBALANCE_EVERY_N_DAYS),
    }

    return metrics


def analyze_by_year(equity_curve: pd.DataFrame) -> None:
    df = equity_curve.copy()
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year

    yearly = df.groupby("year").agg(
        total_return=("end_capital", lambda x: x.iloc[-1] / x.iloc[0] - 1),
        avg_return=("daily_return", "mean"),
        vol=("daily_return", "std"),
        win_rate=("daily_return", lambda x: (x > 0).mean()),
    )

    yearly["sharpe"] = yearly["avg_return"] / yearly["vol"] * np.sqrt(252)

    print("\nPerformance por ano")
    print("-------------------")
    print(yearly)


def run_cost_sensitivity(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    costs = [0.001, 0.002, 0.003]

    print("\nSensibilidade a custo")
    print("---------------------")

    global TRANSACTION_COST_PER_SIDE

    original_cost = TRANSACTION_COST_PER_SIDE

    for cost in costs:
        TRANSACTION_COST_PER_SIDE = cost

        trades = build_trade_book(
            preds=preds,
            prices=prices,
            top_n=TOP_N,
            bottom_n=BOTTOM_N,
            rebalance_every_n_days=REBALANCE_EVERY_N_DAYS,
            holding_days=HOLDING_DAYS,
            mode="long_short",
        )
        daily_positions = expand_trades_to_daily_positions(trades, prices)
        daily_positions = apply_transaction_costs(daily_positions, trades)
        equity_curve = build_daily_equity_curve(daily_positions, INITIAL_CAPITAL)
        metrics = compute_metrics(equity_curve, INITIAL_CAPITAL)

        print(f"\nCusto: {cost}")
        print(f"Sharpe: {metrics['annualized_sharpe']}")
        print(f"Return: {metrics['total_return']}")

    TRANSACTION_COST_PER_SIDE = original_cost


def save_outputs(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    daily_positions: pd.DataFrame,
    metrics: dict,
) -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    equity_path = BACKTEST_DIR / "equity_curve.csv"
    trades_path = BACKTEST_DIR / "trade_book.csv"
    daily_positions_path = BACKTEST_DIR / "daily_positions.csv"
    metrics_path = BACKTEST_DIR / "backtest_metrics.json"

    equity_curve.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)
    daily_positions.to_csv(daily_positions_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Curva de patrimônio salva em: {equity_path}")
    print(f"Trade book salvo em: {trades_path}")
    print(f"Posições diárias salvas em: {daily_positions_path}")
    print(f"Métricas salvas em: {metrics_path}")


def print_summary(metrics: dict, equity_curve: pd.DataFrame) -> None:
    print("\nResumo do backtest")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nÚltimos dias da curva")
    print("---------------------")
    print(equity_curve.tail(10).to_string(index=False))


def main() -> None:
    preds = load_predictions()

    start_date = preds["trade_date"].min()
    end_date = preds["trade_date"].max() + pd.Timedelta(days=20)

    prices = load_prices(
        symbols=preds["symbol"].unique().tolist(),
        start_date=start_date,
        end_date=end_date,
    )
    prices = prepare_price_panel(prices)

    trades = build_trade_book(
        preds=preds,
        prices=prices,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        rebalance_every_n_days=REBALANCE_EVERY_N_DAYS,
        holding_days=HOLDING_DAYS,
        mode="long_short",
    )

    daily_positions = expand_trades_to_daily_positions(trades, prices)
    daily_positions = apply_transaction_costs(daily_positions, trades)

    equity_curve = build_daily_equity_curve(
        daily_positions=daily_positions,
        initial_capital=INITIAL_CAPITAL,
    )

    metrics = compute_metrics(equity_curve, INITIAL_CAPITAL)

    save_outputs(equity_curve, trades, daily_positions, metrics)
    print_summary(metrics, equity_curve)
    analyze_by_year(equity_curve)
    run_cost_sensitivity(preds, prices)


if __name__ == "__main__":
    main()