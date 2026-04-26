from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = BASE_DIR / "artifacts" / "reports"
BACKTEST_DIR = REPORT_DIR / "backtest"
MODEL_REPORT_PATH = REPORT_DIR / "test_predictions.csv"
PRICE_TABLE = "market_data.daily_prices"

@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100000.0
    top_n: int = 10
    bottom_n: int = 6
    top_n_exit: int = 20
    bottom_n_exit: int = 14
    long_quantile: float = 0.9
    short_quantile: float = 0.05
    transaction_cost_per_side: float = 0.001
    slippage: float = 0.001
    holding_days: int = 63
    backtest_strategy: str = "block"
    portfolio_mode: str = "long_short"
    min_avg_daily_volume_20d: float = 100000
    min_avg_daily_traded_value_20d: float = 1000000.0
    min_side_weight: float = 0.2
    max_side_weight: float = 0.8
    long_min_signal_strength: float = 0.5
    short_min_signal_strength: float = 0.4
    target_signal_strength: float = 2.5

    use_dynamic_signal_filter: bool = False
    signal_filter_lookback: int = 12
    signal_filter_quantile: float = 0.6
    min_history_for_signal_filter: int = 6

    run_yearly_analysis: bool = True
    run_cost_sensitivity: bool = True
    cost_sensitivity_values: tuple[float, ...] = (0.001, 0.002, 0.003)

    use_macro_regime_scaling: bool = False

    # =========================
    # IC EXPOSURE SCALING
    # =========================
    use_ic_exposure_scaling: bool = False
    ic_exposure_lookback: int = 3
    min_history_for_ic_scaling: int = 2

    use_performance_scaling: bool = False
    performance_lookback: int = 20
    performance_soft_threshold: float = 0.0
    performance_hard_threshold: float = -0.001
    performance_soft_scale: float = 0.5
    performance_hard_scale: float = 0.0

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
    parser.add_argument("--bottom-n", type=int, default=6)
    parser.add_argument("--top-n-exit", type=int, default=20)
    parser.add_argument("--bottom-n-exit", type=int, default=14)
    parser.add_argument("--long-quantile", type=float, default=0.9)
    parser.add_argument("--short-quantile", type=float, default=0.05)
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
    parser.add_argument("--long-min-signal-strength", type=float, default=0.5)
    parser.add_argument("--short-min-signal-strength", type=float, default=0.4)
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

    parser.add_argument("--dynamic-signal-filter", action="store_true")
    parser.add_argument("--macro-regime-scaling", action="store_true")
    parser.add_argument("--no-ic-exposure-scaling", action="store_true")

    parser.add_argument("--performance-scaling", action="store_true")
    parser.add_argument("--performance-lookback", type=int, default=20)
    parser.add_argument("--performance-soft-threshold", type=float, default=0.0)
    parser.add_argument("--performance-hard-threshold", type=float, default=-0.001)
    parser.add_argument("--performance-soft-scale", type=float, default=0.5)
    parser.add_argument("--performance-hard-scale", type=float, default=0.0)

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
        long_min_signal_strength=args.long_min_signal_strength,
        short_min_signal_strength=args.short_min_signal_strength,
        target_signal_strength=args.target_signal_strength,
        run_yearly_analysis=not args.skip_yearly_analysis,
        run_cost_sensitivity=not args.skip_cost_sensitivity,
        cost_sensitivity_values=tuple(args.cost_sensitivity_values),
        use_dynamic_signal_filter=args.dynamic_signal_filter,
        use_macro_regime_scaling=args.macro_regime_scaling,
        use_ic_exposure_scaling=not args.no_ic_exposure_scaling,
        use_performance_scaling=args.performance_scaling,
        performance_lookback=args.performance_lookback,
        performance_soft_threshold=args.performance_soft_threshold,
        performance_hard_threshold=args.performance_hard_threshold,
        performance_soft_scale=args.performance_soft_scale,
        performance_hard_scale=args.performance_hard_scale,
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
