from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.training_db import engine


BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "artifacts" / "reports"
BACKTEST_DIR = REPORT_DIR / "backtest"
MODEL_REPORT_PATH = REPORT_DIR / "test_predictions.csv"
PRICE_TABLE = "market_data.daily_prices"


@dataclass(frozen=True)
class BacktestConfig:
    """Parâmetros centrais do backtest.

    A ideia é concentrar aqui as variáveis antes espalhadas pelo módulo,
    facilitando manutenção, testes e execução via CLI.
    """

    initial_capital: float = 100000.0
    top_n: int = 10
    bottom_n: int = 10
    top_n_exit: int = 20
    bottom_n_exit: int = 20
    long_quantile: float = 0.9
    short_quantile: float = 0.1
    transaction_cost_per_side: float = 0.001
    slippage: float = 0.001
    holding_days: int = 63
    backtest_strategy: str = "block"  # block | staggered
    portfolio_mode: str = "long_short"  # long_short | long_only | short_only
    min_avg_daily_volume_20d: float = 100000
    min_avg_daily_traded_value_20d: float = 1000000.0
    min_side_weight: float = 0.2
    max_side_weight: float = 0.8
    min_signal_strength: float = 0.5
    target_signal_strength: float = 2.5
    run_yearly_analysis: bool = True
    run_cost_sensitivity: bool = True
    cost_sensitivity_values: tuple[float, ...] = (0.001, 0.002, 0.003)


@dataclass(frozen=True)
class StrategyConfig:
    rebalance_every_n_days: int
    normalize_active_lots: bool


def parse_args() -> BacktestConfig:
    """Lê argumentos de linha de comando e devolve a configuração final."""
    parser = argparse.ArgumentParser(
        description="Executa o backtest a partir de test_predictions.csv."
    )

    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bottom-n", type=int, default=10)
    parser.add_argument("--top-n-exit", type=int, default=20)
    parser.add_argument("--bottom-n-exit", type=int, default=20)
    parser.add_argument("--long-quantile", type=float, default=0.9)
    parser.add_argument("--short-quantile", type=float, default=0.1)
    parser.add_argument("--transaction-cost-per-side", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--holding-days", type=int, default=63)
    parser.add_argument(
        "--backtest-strategy",
        choices=["block", "staggered"],
        default="block",
    )
    parser.add_argument(
        "--portfolio-mode",
        choices=["long_short", "long_only", "short_only"],
        default="long_short",
    )
    parser.add_argument("--min-avg-daily-volume-20d", type=float, default=100000)
    parser.add_argument("--min-avg-daily-traded-value-20d", type=float, default=1000000.0)
    parser.add_argument("--min-side-weight", type=float, default=0.2)
    parser.add_argument("--max-side-weight", type=float, default=0.8)
    parser.add_argument("--min-signal-strength", type=float, default=0.5)
    parser.add_argument("--target-signal-strength", type=float, default=2.5)
    parser.add_argument(
        "--skip-yearly-analysis",
        action="store_true",
        help="Não imprime o detalhamento anual.",
    )
    parser.add_argument(
        "--skip-cost-sensitivity",
        action="store_true",
        help="Não executa o bloco de sensibilidade a custos.",
    )
    parser.add_argument(
        "--cost-sensitivity-values",
        type=float,
        nargs="*",
        default=[0.001, 0.002, 0.003],
        help="Lista de custos usada na análise de sensibilidade.",
    )

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        top_n_exit=args.top_n_exit,
        bottom_n_exit=args.bottom_n_exit,
        long_quantile=args.long_quantile,
        short_quantile=args.short_quantile,
        transaction_cost_per_side=args.transaction_cost_per_side,
        slippage=args.slippage,
        holding_days=args.holding_days,
        backtest_strategy=args.backtest_strategy,
        portfolio_mode=args.portfolio_mode,
        min_avg_daily_volume_20d=args.min_avg_daily_volume_20d,
        min_avg_daily_traded_value_20d=args.min_avg_daily_traded_value_20d,
        min_side_weight=args.min_side_weight,
        max_side_weight=args.max_side_weight,
        min_signal_strength=args.min_signal_strength,
        target_signal_strength=args.target_signal_strength,
        run_yearly_analysis=not args.skip_yearly_analysis,
        run_cost_sensitivity=not args.skip_cost_sensitivity,
        cost_sensitivity_values=tuple(args.cost_sensitivity_values),
    )

    validate_config(config)
    return config



def validate_config(config: BacktestConfig) -> None:
    """Valida combinações inválidas para falhar cedo e de forma clara."""
    if config.top_n <= 0:
        raise ValueError("top_n deve ser maior que zero.")
    if config.bottom_n <= 0:
        raise ValueError("bottom_n deve ser maior que zero.")
    if config.top_n_exit < config.top_n:
        raise ValueError("top_n_exit deve ser maior ou igual a top_n.")
    if config.bottom_n_exit < config.bottom_n:
        raise ValueError("bottom_n_exit deve ser maior ou igual a bottom_n.")
    if not 0 < config.long_quantile <= 1:
        raise ValueError("long_quantile deve estar entre 0 e 1.")
    if not 0 <= config.short_quantile < 1:
        raise ValueError("short_quantile deve estar entre 0 e 1.")
    if config.min_side_weight < 0 or config.max_side_weight > 1:
        raise ValueError("min_side_weight e max_side_weight devem estar entre 0 e 1.")
    if config.min_side_weight > config.max_side_weight:
        raise ValueError("min_side_weight não pode ser maior que max_side_weight.")
    if config.holding_days <= 0:
        raise ValueError("holding_days deve ser maior que zero.")



def get_strategy_config(config: BacktestConfig) -> StrategyConfig:
    """Traduz o modo de execução em parâmetros operacionais do backtest."""
    if config.backtest_strategy == "block":
        return StrategyConfig(
            rebalance_every_n_days=config.holding_days,
            normalize_active_lots=False,
        )

    if config.backtest_strategy == "staggered":
        return StrategyConfig(
            rebalance_every_n_days=1,
            normalize_active_lots=True,
        )

    raise ValueError(f"Estratégia desconhecida: {config.backtest_strategy}")



def load_predictions(model_report_path: Path = MODEL_REPORT_PATH) -> pd.DataFrame:
    """Carrega as previsões do modelo e garante as colunas mínimas esperadas."""
    if not model_report_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {model_report_path}. Rode antes o train_model.py."
        )

    df = pd.read_csv(model_report_path, parse_dates=["trade_date"])

    required_cols = {"trade_date", "symbol", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em test_predictions.csv: {sorted(missing)}")

    return df.sort_values(["trade_date", "prediction"], ascending=[True, False]).reset_index(drop=True)



def load_prices(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Busca os preços necessários para todo o intervalo do backtest."""
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



def apply_liquidity_filter_to_predictions(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Remove ativos sem liquidez mínima, quando as colunas existirem."""
    df = df.copy()

    required_cols = {"avg_daily_volume_20d", "avg_daily_traded_value_20d"}
    if not required_cols.issubset(df.columns):
        return df

    valid_volume = (
        df["avg_daily_volume_20d"].notna()
        & (df["avg_daily_volume_20d"] >= config.min_avg_daily_volume_20d)
    )
    valid_traded_value = (
        df["avg_daily_traded_value_20d"].notna()
        & (df["avg_daily_traded_value_20d"] >= config.min_avg_daily_traded_value_20d)
    )

    return df[valid_volume & valid_traded_value].copy()



def prepare_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Padroniza preços e calcula retorno diário por ativo."""
    df = prices.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)

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
    """Seleciona datas de rebalanceamento conforme o passo configurado."""
    unique_dates = sorted(preds["trade_date"].drop_duplicates())
    return unique_dates[::step]



def compute_signal_strength(day_df: pd.DataFrame) -> float:
    """Resume a qualidade relativa do sinal daquele dia em um único escalar."""
    top_mean = day_df.head(10)["prediction"].mean()
    median = day_df["prediction"].median()
    max_abs_pred = abs(day_df["prediction"]).max()
    std_norm = day_df["prediction"].std()

    if pd.isna(max_abs_pred) or max_abs_pred == 0:
        return float("nan")

    spread_norm = (top_mean - median) / max_abs_pred
    return float(0.5 * spread_norm + 0.5 * std_norm)



def weight_positions(
    positions: pd.DataFrame,
    day_df: pd.DataFrame,
    config: BacktestConfig,
    signal_strength: float,
) -> pd.DataFrame:
    """Calcula pesos intra-side e pesos agregados entre long e short."""
    longs = positions[positions["side"] == "long"].copy()
    shorts = positions[positions["side"] == "short"].copy()

    has_longs = not longs.empty
    has_shorts = not shorts.empty
    weighted_dfs: list[pd.DataFrame] = []

    if has_longs:
        longs["score"] = longs["prediction"] - longs["prediction"].min() + 1e-6
        longs["intra_weight"] = longs["score"] / longs["score"].sum()

    if has_shorts:
        shorts["score"] = shorts["prediction"].max() - shorts["prediction"] + 1e-6
        shorts["intra_weight"] = shorts["score"] / shorts["score"].sum()

    if has_longs and has_shorts:
        long_strength = float(longs["score"].mean())
        short_strength = float(shorts["score"].mean())
        strength_sum = long_strength + short_strength

        if strength_sum > 0:
            long_side_weight = long_strength / strength_sum
            short_side_weight = short_strength / strength_sum
        else:
            long_side_weight = 0.5
            short_side_weight = 0.5

        long_side_weight = min(max(long_side_weight, config.min_side_weight), config.max_side_weight)
        short_side_weight = 1.0 - long_side_weight

        longs["weight"] = longs["intra_weight"] * long_side_weight
        shorts["weight"] = shorts["intra_weight"] * short_side_weight
        weighted_dfs.extend([longs, shorts])
    elif has_longs:
        longs["weight"] = longs["intra_weight"]
        weighted_dfs.append(longs)
    elif has_shorts:
        shorts["weight"] = shorts["intra_weight"]
        weighted_dfs.append(shorts)

    positions = pd.concat(weighted_dfs, ignore_index=True)

    total_weight = positions["weight"].sum()
    if total_weight > 0:
        positions["weight"] = positions["weight"] / total_weight

    exposure_scale = min(1.0, signal_strength / config.target_signal_strength)
    positions["weight"] = positions["weight"] * exposure_scale
    return positions



def select_positions(
    day_df: pd.DataFrame,
    config: BacktestConfig,
    prev_long_symbols: set[str] | None = None,
    prev_short_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """Seleciona o portfólio do dia, aplicando buffer de saída e pesos."""
    filtered_df = apply_liquidity_filter_to_predictions(day_df, config)
    filtered_df = filtered_df.sort_values("prediction", ascending=False).copy()

    if filtered_df.empty:
        return pd.DataFrame(columns=list(filtered_df.columns) + ["side", "weight"])

    prev_long_symbols = prev_long_symbols or set()
    prev_short_symbols = prev_short_symbols or set()
    selected_parts: list[pd.DataFrame] = []

    if config.portfolio_mode in ["long_short", "long_only"]:
        long_threshold = filtered_df["prediction"].quantile(config.long_quantile)
        longs_entry = filtered_df[filtered_df["prediction"] >= long_threshold].head(config.top_n).copy()
        longs_exit_pool = filtered_df.head(config.top_n_exit).copy()

        entry_symbols = set(longs_entry["symbol"].astype(str))
        exit_pool_symbols = set(longs_exit_pool["symbol"].astype(str))
        kept_symbols = prev_long_symbols & exit_pool_symbols
        final_symbols = entry_symbols | kept_symbols

        longs = filtered_df[filtered_df["symbol"].astype(str).isin(final_symbols)].copy()
        longs = longs.sort_values("prediction", ascending=False)

        if not longs.empty:
            longs["side"] = "long"
            selected_parts.append(longs)

    if config.portfolio_mode in ["long_short", "short_only"]:
        short_threshold = filtered_df["prediction"].quantile(config.short_quantile)
        shorts_entry = filtered_df[filtered_df["prediction"] <= short_threshold].tail(config.bottom_n).copy()
        shorts_exit_pool = filtered_df.tail(config.bottom_n_exit).copy()

        entry_symbols = set(shorts_entry["symbol"].astype(str))
        exit_pool_symbols = set(shorts_exit_pool["symbol"].astype(str))
        kept_symbols = prev_short_symbols & exit_pool_symbols
        final_symbols = entry_symbols | kept_symbols

        shorts = filtered_df[filtered_df["symbol"].astype(str).isin(final_symbols)].copy()
        shorts = shorts.sort_values("prediction", ascending=True)

        if not shorts.empty:
            shorts["side"] = "short"
            selected_parts.append(shorts)

    if not selected_parts:
        return pd.DataFrame(columns=list(filtered_df.columns) + ["side", "weight"])

    positions = pd.concat(selected_parts, ignore_index=True)
    positions = positions.drop_duplicates(subset=["symbol", "side"]).copy()

    signal_strength = compute_signal_strength(filtered_df)
    if pd.isna(signal_strength) or signal_strength < config.min_signal_strength:
        return pd.DataFrame(columns=list(filtered_df.columns) + ["side", "weight"])

    return weight_positions(positions, filtered_df, config, signal_strength)



def get_calendar(prices: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(prices["trade_date"]).drop_duplicates())



def get_trade_window(
    signal_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    holding_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]] | None:
    """Traduz uma data de sinal em entrada, saída e dias ativos de retorno."""
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
    config: BacktestConfig,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    """Gera o livro de trades do backtest."""
    rebalance_dates = select_rebalance_dates(preds, step=strategy.rebalance_every_n_days)
    calendar = get_calendar(prices)

    trade_rows: list[dict[str, Any]] = []
    trade_id = 0
    prev_long_symbols: set[str] = set()
    prev_short_symbols: set[str] = set()

    for signal_date in rebalance_dates:
        window = get_trade_window(signal_date=signal_date, calendar=calendar, holding_days=config.holding_days)
        if window is None:
            continue

        entry_date, exit_date, active_return_dates = window
        day_df = preds[preds["trade_date"] == signal_date].copy()

        positions = select_positions(
            day_df=day_df,
            config=config,
            prev_long_symbols=prev_long_symbols,
            prev_short_symbols=prev_short_symbols,
        )

        if positions.empty:
            prev_long_symbols = set()
            prev_short_symbols = set()
            continue

        prev_long_symbols = set(positions.loc[positions["side"] == "long", "symbol"].astype(str))
        prev_short_symbols = set(positions.loc[positions["side"] == "short", "symbol"].astype(str))

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
    """Expande cada trade em posições diárias com contribuição de retorno."""
    price_panel = prices[["symbol", "trade_date", "daily_return"]].copy()
    expanded_rows: list[dict[str, Any]] = []

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

    daily_positions = daily_positions.merge(price_panel, on=["symbol", "trade_date"], how="left")
    daily_positions = daily_positions.dropna(subset=["daily_return"]).copy()

    daily_positions["position_return"] = np.where(
        daily_positions["side"] == "long",
        daily_positions["daily_return"],
        -daily_positions["daily_return"],
    )
    daily_positions["capital_scale"] = 1.0
    daily_positions["position_contribution"] = daily_positions["weight"] * daily_positions["position_return"]
    daily_positions["signed_weight"] = np.where(
        daily_positions["side"] == "long",
        daily_positions["weight"],
        -daily_positions["weight"],
    )
    return daily_positions



def normalize_active_lots_exposure(daily_positions: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a exposição quando há vários lotes simultâneos ativos."""
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



def apply_transaction_costs(
    daily_positions: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Adiciona linhas sintéticas de custo na entrada e na saída dos trades."""
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

    cost_rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        weight = float(trade["weight"])
        entry_date = pd.to_datetime(trade["entry_date"])
        exit_date = pd.to_datetime(trade["exit_date"])

        entry_active_lots = int(active_lots_map.get(entry_date, 1))
        exit_active_lots = int(active_lots_map.get(exit_date, 1))

        entry_scale = 1.0 / entry_active_lots if entry_active_lots > 0 else 1.0
        exit_scale = 1.0 / exit_active_lots if exit_active_lots > 0 else 1.0

        entry_cost = weight * entry_scale * (config.transaction_cost_per_side + config.slippage)
        exit_cost = weight * exit_scale * config.transaction_cost_per_side

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
    return out.sort_values(["trade_date", "trade_id", "row_type"]).reset_index(drop=True)



def compute_turnover(trades: pd.DataFrame) -> pd.DataFrame:
    """Mede o giro entre um rebalanceamento e o próximo."""
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

    rows: list[dict[str, Any]] = []
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
    """Consolida PnL diário, exposição e curva de patrimônio."""
    df = daily_positions.copy()

    if "row_type" not in df.columns:
        df["row_type"] = "market_return"

    pnl_by_day = (
        df.groupby("trade_date", as_index=False)
        .agg(daily_return=("position_contribution", "sum"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    market_df = df[df["row_type"] == "market_return"].copy()
    market_df["gross_weight_raw"] = market_df["weight"]
    market_df["gross_weight_effective"] = market_df["weight"] * market_df["capital_scale"]
    market_df["net_weight_effective"] = market_df["signed_weight"] * market_df["capital_scale"]
    market_df["long_contribution"] = np.where(market_df["side"] == "long", market_df["position_contribution"], 0.0)
    market_df["short_contribution"] = np.where(market_df["side"] == "short", market_df["position_contribution"], 0.0)

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
    for col in [
        "num_active_rows",
        "num_active_trades",
        "gross_exposure_raw",
        "gross_exposure_effective",
        "net_exposure_effective",
        "long_contribution",
        "short_contribution",
    ]:
        default = 0 if "num_" in col else 0.0
        grouped[col] = grouped[col].fillna(default)

    grouped["num_active_rows"] = grouped["num_active_rows"].astype(int)
    grouped["num_active_trades"] = grouped["num_active_trades"].astype(int)

    rows: list[dict[str, Any]] = []
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
    config: BacktestConfig,
    strategy: StrategyConfig,
    turnover_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calcula as métricas principais do backtest."""
    total_return = (equity_curve["end_capital"].iloc[-1] / config.initial_capital) - 1.0
    avg_daily_return = equity_curve["daily_return"].mean()
    daily_vol = equity_curve["daily_return"].std()

    sharpe = np.nan
    if pd.notna(daily_vol) and daily_vol > 0:
        sharpe = (avg_daily_return / daily_vol) * np.sqrt(252)

    avg_turnover = turnover_df["turnover_ratio"].mean() if not turnover_df.empty else None
    median_turnover = turnover_df["turnover_ratio"].median() if not turnover_df.empty else None
    max_turnover = turnover_df["turnover_ratio"].max() if not turnover_df.empty else None

    return {
        "strategy": config.backtest_strategy,
        "portfolio_mode": config.portfolio_mode,
        "initial_capital": float(config.initial_capital),
        "final_capital": float(equity_curve["end_capital"].iloc[-1]),
        "total_return": float(total_return),
        "avg_daily_return": float(avg_daily_return),
        "daily_volatility": float(daily_vol) if pd.notna(daily_vol) else None,
        "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "max_drawdown": float(equity_curve["drawdown"].min()),
        "positive_day_ratio": float((equity_curve["daily_return"] > 0).mean()),
        "num_days": int(len(equity_curve)),
        "top_n_longs_entry": int(config.top_n),
        "top_n_longs_exit": int(config.top_n_exit),
        "top_n_shorts_entry": int(config.bottom_n),
        "top_n_shorts_exit": int(config.bottom_n_exit),
        "transaction_cost_per_side": float(config.transaction_cost_per_side),
        "slippage": float(config.slippage),
        "holding_days": int(config.holding_days),
        "rebalance_every_n_days": int(strategy.rebalance_every_n_days),
        "normalize_active_lots": bool(strategy.normalize_active_lots),
        "long_avg_daily_return": float(equity_curve["long_contribution"].mean()),
        "short_avg_daily_return": float(equity_curve["short_contribution"].mean()),
        "long_total_contribution": float(equity_curve["long_contribution"].sum()),
        "short_total_contribution": float(equity_curve["short_contribution"].sum()),
        "avg_gross_exposure_effective": float(equity_curve["gross_exposure_effective"].mean()),
        "avg_net_exposure_effective": float(equity_curve["net_exposure_effective"].mean()),
        "avg_turnover": float(avg_turnover) if avg_turnover is not None and pd.notna(avg_turnover) else None,
        "median_turnover": float(median_turnover) if median_turnover is not None and pd.notna(median_turnover) else None,
        "max_turnover": float(max_turnover) if max_turnover is not None and pd.notna(max_turnover) else None,
    }



def analyze_by_year(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Monta um resumo anual para inspeção visual."""
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
    return yearly



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
    equity_curve = build_daily_equity_curve(daily_positions=daily_positions, initial_capital=config.initial_capital)
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
