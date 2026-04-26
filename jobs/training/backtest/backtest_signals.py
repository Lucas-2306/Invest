from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .backtest_config import BacktestConfig
    from .backtest_data import apply_liquidity_filter_to_predictions
except ImportError:
    from backtest_config import BacktestConfig
    from backtest_data import apply_liquidity_filter_to_predictions


def should_trade_by_signal_history(
    signal_strength: float,
    historical_signal_strengths: list[float],
    config: BacktestConfig,
) -> bool:
    if not config.use_dynamic_signal_filter:
        return signal_strength >= config.min_signal_strength

    valid_history = [x for x in historical_signal_strengths if pd.notna(x)]
    if len(valid_history) < config.min_history_for_signal_filter:
        return signal_strength >= config.min_signal_strength

    lookback_history = valid_history[-config.signal_filter_lookback :]
    dynamic_threshold = float(np.quantile(lookback_history, config.signal_filter_quantile))
    final_threshold = max(config.min_signal_strength, dynamic_threshold)

    return signal_strength >= final_threshold

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

def apply_macro_regime_filter(
    exposure_scale: float,
    day_df: pd.DataFrame,
) -> float:
    """
    Ajusta exposição conforme regime de mercado.

    A lógica não zera trades.
    Ela apenas reduz exposição quando o ambiente está pior.
    """

    if day_df.empty:
        return exposure_scale

    row = day_df.iloc[0]

    ibov_above = row.get("ibov_above_sma_200")
    ibov_ret_21d = row.get("ibov_return_21d")
    ibov_vol_21d = row.get("ibov_vol_21d")

    sp500_above = row.get("sp500_above_sma_200")
    sp500_ret_21d = row.get("sp500_return_21d")

    macro_scale = 1.0

    # =========================
    # REGIME DE TENDÊNCIA
    # =========================
    if pd.notna(ibov_above) and ibov_above == 0:
        macro_scale *= 0.85

    if pd.notna(sp500_above) and sp500_above == 0:
        macro_scale *= 0.90

    # =========================
    # MOMENTUM DE MERCADO
    # =========================
    if pd.notna(ibov_ret_21d) and ibov_ret_21d < -0.03:
        macro_scale *= 0.85

    if pd.notna(sp500_ret_21d) and sp500_ret_21d < -0.03:
        macro_scale *= 0.90

    # =========================
    # VOLATILIDADE DE MERCADO
    # =========================
    if pd.notna(ibov_vol_21d) and ibov_vol_21d > 0.025:
        macro_scale *= 0.90

    # nunca reduz demais
    macro_scale = max(macro_scale, 0.60)

    return exposure_scale * macro_scale

def weight_positions(
    positions: pd.DataFrame,
    day_df: pd.DataFrame,
    config: BacktestConfig,
    signal_strength: float,
    historical_ics: list[float] | None = None,
) -> pd.DataFrame:
    """Calcula pesos intra-side e pesos agregados entre long e short."""
    longs = positions[positions["side"] == "long"].copy()
    shorts = positions[positions["side"] == "short"].copy()

    has_longs = not longs.empty
    has_shorts = not shorts.empty
    weighted_dfs: list[pd.DataFrame] = []

    # =========================
    # INTRA-WEIGHTS
    # =========================
    if has_longs:
        longs["score"] = longs["prediction"] - longs["prediction"].min() + 1e-6
        longs["intra_weight"] = longs["score"] / longs["score"].sum()

    if has_shorts:
        shorts["score"] = shorts["prediction"].max() - shorts["prediction"] + 1e-6
        shorts["intra_weight"] = shorts["score"] / shorts["score"].sum()

    # =========================
    # SIDE WEIGHTS
    # =========================
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

    # =========================
    # NORMALIZAÇÃO
    # =========================
    total_weight = positions["weight"].sum()
    if total_weight > 0:
        positions["weight"] = positions["weight"] / total_weight

    # =========================
    # SIGNAL STRENGTH SCALING
    # =========================
    exposure_scale = min(1.0, signal_strength / config.target_signal_strength)

    if config.use_macro_regime_scaling:
        exposure_scale = apply_macro_regime_filter(
            exposure_scale=exposure_scale,
            day_df=day_df,
        )

    # =========================
    # IC EXPOSURE SCALING
    # =========================
    ic_scale = 1.0
    mean_ic = None

    if config.use_ic_exposure_scaling and historical_ics:
        valid_ics = [x for x in historical_ics if pd.notna(x)]

        if len(valid_ics) >= config.min_history_for_ic_scaling:
            recent_ics = valid_ics[-config.ic_exposure_lookback :]
            mean_ic = float(np.mean(recent_ics))

            ic_scale = float(np.clip(1 + 3.0 * mean_ic, 0.7, 1.4))

    #if config.use_ic_exposure_scaling:
        #print("IC:", mean_ic, "scale:", ic_scale)

    positions["weight"] = positions["weight"] * exposure_scale * ic_scale

    return positions

def select_positions(
    day_df: pd.DataFrame,
    config: BacktestConfig,
    prev_long_symbols: set[str] | None = None,
    prev_short_symbols: set[str] | None = None,
    historical_signal_strengths: list[float] | None = None,
    historical_ics: list[float] | None = None,
    portfolio_returns: list[float] | None = None,
) -> pd.DataFrame:
    """Seleciona o portfólio do dia, aplicando buffer de saída e pesos."""
    filtered_df = apply_liquidity_filter_to_predictions(day_df, config)
    filtered_df = filtered_df.sort_values("prediction", ascending=False).copy()

    if filtered_df.empty:
        return pd.DataFrame(columns=list(filtered_df.columns) + ["side", "weight"])

    signal_strength = compute_signal_strength(filtered_df)

    if pd.isna(signal_strength):
        return pd.DataFrame(columns=list(filtered_df.columns) + ["side", "weight"])

    long_min_signal_strength = config.long_min_signal_strength
    short_min_signal_strength = config.short_min_signal_strength



    prev_long_symbols = prev_long_symbols or set()
    prev_short_symbols = prev_short_symbols or set()
    selected_parts: list[pd.DataFrame] = []

    if (
        config.portfolio_mode in ["long_short", "long_only"]
        and signal_strength >= long_min_signal_strength
    ):
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

    if (
        config.portfolio_mode in ["long_short", "short_only"]
        and signal_strength >= short_min_signal_strength
    ):
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

    return weight_positions(
        positions,
        filtered_df,
        config,
        signal_strength,
        historical_ics=historical_ics,
    )
