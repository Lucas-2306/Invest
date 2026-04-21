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

PRICE_TABLE = "market_data.daily_prices"

INITIAL_CAPITAL = 100000.0
TOP_N = 5
BOTTOM_N = 5
TRANSACTION_COST_PER_SIDE = 0.001
SLIPPAGE = 0.001
HOLDING_DAYS = 5

# Estratégia de execução:
# - "block_5d": rebalance a cada 5 dias, um lote por vez
# - "staggered": rebalance diário, holding de 5 dias, lotes simultâneos com exposição controlada
BACKTEST_STRATEGY = "staggered"

# Modo de portfólio:
# - "long_short"
# - "long_only"
# - "short_only"
PORTFOLIO_MODE = "short_only"


def get_strategy_config() -> dict:
    if BACKTEST_STRATEGY == "block_5d":
        return {
            "rebalance_every_n_days": 5,
            "normalize_active_lots": False,
        }

    if BACKTEST_STRATEGY == "staggered":
        return {
            "rebalance_every_n_days": 1,
            "normalize_active_lots": True,
        }

    raise ValueError(f"Estratégia desconhecida: {BACKTEST_STRATEGY}")


def load_predictions() -> pd.DataFrame:
    if not MODEL_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {MODEL_REPORT_PATH}. Rode antes o train_model.py."
        )

    df = pd.read_csv(MODEL_REPORT_PATH, parse_dates=["trade_date"])

    required_cols = {"trade_date", "symbol", "prediction"}
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
        FROM {PRICE_TABLE} p
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

    dfs = []

    if mode in ["long_short", "long_only"]:
        longs = day_df.head(top_n).copy()
        if not longs.empty:
            longs["side"] = "long"
            dfs.append(longs)

    if mode in ["long_short", "short_only"]:
        shorts = day_df.tail(bottom_n).copy()
        if not shorts.empty:
            shorts["side"] = "short"
            dfs.append(shorts)

    if not dfs:
        return pd.DataFrame(columns=day_df.columns)

    positions = pd.concat(dfs, ignore_index=True)

    total_positions = len(positions)
    positions["weight"] = 1.0 / total_positions if total_positions > 0 else 0.0

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

    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 1 + holding_days

    if exit_idx >= len(calendar):
        return None

    entry_date = calendar[entry_idx]
    exit_date = calendar[exit_idx]
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
        window = get_trade_window(signal_date=signal_date, calendar=calendar, holding_days=holding_days)
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
                    "row_type": "market_return",
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

    daily_positions["capital_scale"] = 1.0
    daily_positions["position_contribution"] = (
        daily_positions["weight"] * daily_positions["position_return"]
    )

    daily_positions["signed_weight"] = np.where(
        daily_positions["side"] == "long",
        daily_positions["weight"],
        -daily_positions["weight"],
    )

    return daily_positions


def normalize_active_lots_exposure(daily_positions: pd.DataFrame) -> pd.DataFrame:
    df = daily_positions.copy()

    if "row_type" not in df.columns:
        df["row_type"] = "market_return"

    market_df = df[df["row_type"] == "market_return"].copy()

    active_lots = (
        market_df.groupby("trade_date")["signal_date"]
        .nunique()
        .rename("num_active_lots")
        .reset_index()
    )

    df = df.merge(active_lots, on="trade_date", how="left")
    df["num_active_lots"] = df["num_active_lots"].fillna(0).astype(int)

    is_market = df["row_type"] == "market_return"
    df["capital_scale"] = 1.0

    valid_market = is_market & (df["num_active_lots"] > 0)
    df.loc[valid_market, "capital_scale"] = 1.0 / df.loc[valid_market, "num_active_lots"]

    df["position_contribution"] = np.where(
        is_market,
        df["weight"] * df["position_return"] * df["capital_scale"],
        df["position_contribution"],
    )

    return df


def apply_transaction_costs(daily_positions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    daily_positions = daily_positions.copy()

    if "row_type" not in daily_positions.columns:
        daily_positions["row_type"] = "market_return"

    active_lots = (
        daily_positions[daily_positions["row_type"] == "market_return"]
        .groupby("trade_date")["signal_date"]
        .nunique()
        .rename("num_active_lots")
        .reset_index()
    )

    active_lots_map = dict(zip(active_lots["trade_date"], active_lots["num_active_lots"]))

    cost_rows: list[dict] = []

    for _, trade in trades.iterrows():
        weight = float(trade["weight"])
        entry_date = pd.to_datetime(trade["entry_date"])
        exit_date = pd.to_datetime(trade["exit_date"])

        entry_active_lots = int(active_lots_map.get(entry_date, 1))
        exit_active_lots = int(active_lots_map.get(exit_date, 1))

        entry_scale = 1.0 / entry_active_lots if entry_active_lots > 0 else 1.0
        exit_scale = 1.0 / exit_active_lots if exit_active_lots > 0 else 1.0

        entry_cost = weight * entry_scale * (TRANSACTION_COST_PER_SIDE + SLIPPAGE)
        exit_cost = weight * exit_scale * TRANSACTION_COST_PER_SIDE

        base_row = {
            "trade_id": int(trade["trade_id"]),
            "symbol": str(trade["symbol"]),
            "side": str(trade["side"]),
            "weight": weight,
            "signal_date": pd.to_datetime(trade["signal_date"]),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "prediction": float(trade["prediction"]),
            "daily_return": np.nan,
            "position_return": 0.0,
            "signed_weight": 0.0,
        }

        cost_rows.append(
            {
                **base_row,
                "trade_date": entry_date,
                "capital_scale": entry_scale,
                "position_contribution": -entry_cost,
                "row_type": "entry_cost",
            }
        )

        cost_rows.append(
            {
                **base_row,
                "trade_date": exit_date,
                "capital_scale": exit_scale,
                "position_contribution": -exit_cost,
                "row_type": "exit_cost",
            }
        )

    cost_df = pd.DataFrame(cost_rows)
    out = pd.concat([daily_positions, cost_df], ignore_index=True)
    out = out.sort_values(["trade_date", "trade_id", "row_type"]).reset_index(drop=True)

    return out


def compute_turnover(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "num_prev_symbols",
                "num_curr_symbols",
                "kept_symbols",
                "entered_symbols",
                "exited_symbols",
                "turnover_ratio",
            ]
        )

    rows: list[dict] = []
    previous_symbols: set[str] = set()

    for signal_date, group in trades.groupby("signal_date"):
        current_symbols = set(group["symbol"].astype(str).tolist())

        kept = len(previous_symbols & current_symbols)
        entered = len(current_symbols - previous_symbols)
        exited = len(previous_symbols - current_symbols)

        denom = max(len(previous_symbols), len(current_symbols), 1)
        turnover_ratio = (entered + exited) / denom

        rows.append(
            {
                "signal_date": pd.to_datetime(signal_date),
                "num_prev_symbols": int(len(previous_symbols)),
                "num_curr_symbols": int(len(current_symbols)),
                "kept_symbols": int(kept),
                "entered_symbols": int(entered),
                "exited_symbols": int(exited),
                "turnover_ratio": float(turnover_ratio),
            }
        )

        previous_symbols = current_symbols

    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def build_daily_equity_curve(daily_positions: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    df = daily_positions.copy()

    if "row_type" not in df.columns:
        df["row_type"] = "market_return"

    pnl_by_day = (
        df.groupby("trade_date", as_index=False)
        .agg(
            daily_return=("position_contribution", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    market_df = df[df["row_type"] == "market_return"].copy()

    market_df["gross_weight_raw"] = market_df["weight"]
    market_df["gross_weight_effective"] = market_df["weight"] * market_df["capital_scale"]
    market_df["net_weight_effective"] = market_df["signed_weight"] * market_df["capital_scale"]

    market_df["long_contribution"] = np.where(
        market_df["side"] == "long",
        market_df["position_contribution"],
        0.0,
    )
    market_df["short_contribution"] = np.where(
        market_df["side"] == "short",
        market_df["position_contribution"],
        0.0,
    )

    exposure_by_day = (
        market_df.groupby("trade_date", as_index=False)
        .agg(
            num_active_rows=("trade_id", "count"),
            num_active_trades=("trade_id", "nunique"),
            gross_exposure_raw=("gross_weight_raw", "sum"),
            gross_exposure_effective=("gross_weight_effective", "sum"),
            net_exposure_effective=("net_weight_effective", "sum"),
            long_contribution=("long_contribution", "sum"),
            short_contribution=("short_contribution", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    grouped = pnl_by_day.merge(exposure_by_day, on="trade_date", how="left")

    grouped["num_active_rows"] = grouped["num_active_rows"].fillna(0).astype(int)
    grouped["num_active_trades"] = grouped["num_active_trades"].fillna(0).astype(int)
    grouped["gross_exposure_raw"] = grouped["gross_exposure_raw"].fillna(0.0)
    grouped["gross_exposure_effective"] = grouped["gross_exposure_effective"].fillna(0.0)
    grouped["net_exposure_effective"] = grouped["net_exposure_effective"].fillna(0.0)
    grouped["long_contribution"] = grouped["long_contribution"].fillna(0.0)
    grouped["short_contribution"] = grouped["short_contribution"].fillna(0.0)

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
                "gross_exposure_raw": float(row["gross_exposure_raw"]),
                "gross_exposure_effective": float(row["gross_exposure_effective"]),
                "net_exposure_effective": float(row["net_exposure_effective"]),
                "long_contribution": float(row["long_contribution"]),
                "short_contribution": float(row["short_contribution"]),
            }
        )

    equity_curve = pd.DataFrame(rows)

    if equity_curve.empty:
        raise ValueError("Equity curve vazia.")

    return equity_curve


def compute_metrics(
    equity_curve: pd.DataFrame,
    initial_capital: float,
    strategy: dict,
    turnover_df: pd.DataFrame,
) -> dict:
    total_return = (equity_curve["end_capital"].iloc[-1] / initial_capital) - 1.0
    avg_daily_return = equity_curve["daily_return"].mean()
    daily_vol = equity_curve["daily_return"].std()

    sharpe = np.nan
    if pd.notna(daily_vol) and daily_vol > 0:
        sharpe = (avg_daily_return / daily_vol) * np.sqrt(252)

    max_drawdown = equity_curve["drawdown"].min()
    positive_days = (equity_curve["daily_return"] > 0).mean()

    long_avg_daily = equity_curve["long_contribution"].mean()
    short_avg_daily = equity_curve["short_contribution"].mean()

    long_total_contribution = equity_curve["long_contribution"].sum()
    short_total_contribution = equity_curve["short_contribution"].sum()

    avg_gross_exposure_effective = equity_curve["gross_exposure_effective"].mean()
    avg_net_exposure_effective = equity_curve["net_exposure_effective"].mean()

    avg_turnover = turnover_df["turnover_ratio"].mean() if not turnover_df.empty else None
    median_turnover = turnover_df["turnover_ratio"].median() if not turnover_df.empty else None
    max_turnover = turnover_df["turnover_ratio"].max() if not turnover_df.empty else None

    metrics = {
        "strategy": BACKTEST_STRATEGY,
        "portfolio_mode": PORTFOLIO_MODE,
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
        "rebalance_every_n_days": int(strategy["rebalance_every_n_days"]),
        "normalize_active_lots": bool(strategy["normalize_active_lots"]),
        "long_avg_daily_return": float(long_avg_daily),
        "short_avg_daily_return": float(short_avg_daily),
        "long_total_contribution": float(long_total_contribution),
        "short_total_contribution": float(short_total_contribution),
        "avg_gross_exposure_effective": float(avg_gross_exposure_effective),
        "avg_net_exposure_effective": float(avg_net_exposure_effective),
        "avg_turnover": float(avg_turnover) if avg_turnover is not None and pd.notna(avg_turnover) else None,
        "median_turnover": float(median_turnover) if median_turnover is not None and pd.notna(median_turnover) else None,
        "max_turnover": float(max_turnover) if max_turnover is not None and pd.notna(max_turnover) else None,
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
        long_avg=("long_contribution", "mean"),
        short_avg=("short_contribution", "mean"),
    )

    yearly["sharpe"] = yearly["avg_return"] / yearly["vol"] * np.sqrt(252)

    print("\nPerformance por ano")
    print("-------------------")
    print(yearly)


def run_cost_sensitivity(preds: pd.DataFrame, prices: pd.DataFrame, strategy: dict) -> None:
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
            rebalance_every_n_days=strategy["rebalance_every_n_days"],
            holding_days=HOLDING_DAYS,
            mode=PORTFOLIO_MODE,
        )

        daily_positions = expand_trades_to_daily_positions(trades, prices)

        if strategy["normalize_active_lots"]:
            daily_positions = normalize_active_lots_exposure(daily_positions)

        daily_positions = apply_transaction_costs(daily_positions, trades)
        turnover_df = compute_turnover(trades)
        equity_curve = build_daily_equity_curve(daily_positions, INITIAL_CAPITAL)
        metrics = compute_metrics(equity_curve, INITIAL_CAPITAL, strategy, turnover_df)

        print(f"\nCusto: {cost}")
        print(f"Sharpe: {metrics['annualized_sharpe']}")
        print(f"Return: {metrics['total_return']}")

    TRANSACTION_COST_PER_SIDE = original_cost


def save_outputs(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    daily_positions: pd.DataFrame,
    turnover_df: pd.DataFrame,
    metrics: dict,
) -> None:
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

    print(f"Curva de patrimônio salva em: {equity_path}")
    print(f"Trade book salvo em: {trades_path}")
    print(f"Posições diárias salvas em: {daily_positions_path}")
    print(f"Turnover salvo em: {turnover_path}")
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
    strategy = get_strategy_config()
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
        rebalance_every_n_days=strategy["rebalance_every_n_days"],
        holding_days=HOLDING_DAYS,
        mode=PORTFOLIO_MODE,
    )

    daily_positions = expand_trades_to_daily_positions(trades, prices)

    if strategy["normalize_active_lots"]:
        daily_positions = normalize_active_lots_exposure(daily_positions)

    daily_positions = apply_transaction_costs(daily_positions, trades)
    turnover_df = compute_turnover(trades)

    equity_curve = build_daily_equity_curve(
        daily_positions=daily_positions,
        initial_capital=INITIAL_CAPITAL,
    )

    metrics = compute_metrics(equity_curve, INITIAL_CAPITAL, strategy, turnover_df)

    save_outputs(equity_curve, trades, daily_positions, turnover_df, metrics)
    print_summary(metrics, equity_curve)
    analyze_by_year(equity_curve)
    run_cost_sensitivity(preds, prices, strategy)


if __name__ == "__main__":
    main()