# src/features.py
# Cria features e targets (semanal/mensal) a partir do arquivo long format:
# data/raw/ibov_top40_long.csv
#
# Output:
# - data/processed/ibov_top40_features.parquet
# - data/processed/ibov_top40_features.csv

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


TargetHorizon = Literal["weekly", "monthly", "both"]


@dataclass
class FeatureConfig:
    input_path: str = "data/raw/ibov_top40_long.csv"
    out_dir: str = "data/processed"
    out_base: str = "ibov_top40_features"
    min_history: int = 252  # ~1 ano de pregões; filtra ativos/linhas muito curtas

    # horizontes de target em número de pregões
    horizon_weekly: int = 5
    horizon_monthly: int = 21

    # janelas de rolling (em pregões)
    vol_windows: tuple[int, ...] = (10, 21, 63)
    ma_windows: tuple[int, ...] = (10, 21, 63, 126, 252)
    mom_windows: tuple[int, ...] = (5, 21, 63, 126, 252)

    # RSI
    rsi_window: int = 14


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI clássico (Wilder). Produz NaN no início até ter janela.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing via ewm
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_long_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"date", "ticker", "open", "high", "low", "close", "volume", "adjustedClose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Arquivo não possui colunas esperadas. Faltando: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).copy()

    # Tipos numéricos
    num_cols = ["open", "high", "low", "close", "volume", "adjustedClose"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # adjustedClose pode vir vazio; se estiver, usa close como fallback
    df["adj_close"] = df["adjustedClose"].where(df["adjustedClose"].notna(), df["close"])

    df.loc[df["adj_close"] <= 0, "adj_close"] = np.nan

    # padroniza ticker
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # ordenação essencial para rolling/shift corretos
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Cria features por ticker usando apenas informações do passado.
    """
    out = df.copy()

    # Retornos (usando adj_close pra reduzir efeito de proventos/splits quando disponível)
    out["ret_1d"] = out.groupby("ticker")["adj_close"].pct_change(1)
    out["logret_1d"] = np.log(out.groupby("ticker")["adj_close"].shift(0)) - np.log(
        out.groupby("ticker")["adj_close"].shift(1)
    )

    # Momentum / retorno acumulado
    for w in cfg.mom_windows:
        out[f"mom_{w}d"] = out.groupby("ticker")["adj_close"].pct_change(w)

    # proteção contra divisão por zero / valores absurdos
    mom_cols = [f"mom_{w}d" for w in cfg.mom_windows]
    out[mom_cols] = out[mom_cols].replace([np.inf, -np.inf], np.nan)

    # Médias móveis e distância do preço para MA
    for w in cfg.ma_windows:
        ma = out.groupby("ticker")["adj_close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        out[f"ma_{w}"] = ma
        out[f"price_to_ma_{w}"] = (out["adj_close"] / ma) - 1

    # Volatilidade (desvio padrão dos log-retornos)
    for w in cfg.vol_windows:
        vol = out.groupby("ticker")["logret_1d"].transform(lambda s: s.rolling(w, min_periods=w).std())
        out[f"vol_{w}"] = vol

    # Volume: z-score rolling (volume anormal)
    # (usa log(volume) para reduzir escala)
    out["logvol"] = np.log(out["volume"].replace(0, np.nan))
    vol_mu = out.groupby("ticker")["logvol"].transform(lambda s: s.rolling(63, min_periods=63).mean())
    vol_sd = out.groupby("ticker")["logvol"].transform(lambda s: s.rolling(63, min_periods=63).std())
    out["logvol_z_63"] = (out["logvol"] - vol_mu) / vol_sd

    # Range intradiário relativo
    out["hl_range"] = (out["high"] / out["low"]) - 1
    out["oc_change"] = (out["close"] / out["open"]) - 1

    # RSI
    out[f"rsi_{cfg.rsi_window}"] = out.groupby("ticker")["adj_close"].transform(lambda s: _rsi(s, cfg.rsi_window))

    return out


def add_targets(df: pd.DataFrame, cfg: FeatureConfig, which: TargetHorizon = "both") -> pd.DataFrame:
    """
    Targets:
    - weekly: 5 pregões
    - monthly: 21 pregões

    Definição: target = 1 se retorno futuro > 0, senão 0.
    """
    out = df.copy()

    if which in ("weekly", "both"):
        h = cfg.horizon_weekly
        out["future_ret_w"] = out.groupby("ticker")["adj_close"].shift(-h) / out["adj_close"] - 1
        out["target_w"] = (out["future_ret_w"] > 0).astype("Int64")

    if which in ("monthly", "both"):
        h = cfg.horizon_monthly
        out["future_ret_m"] = out.groupby("ticker")["adj_close"].shift(-h) / out["adj_close"] - 1
        out["target_m"] = (out["future_ret_m"] > 0).astype("Int64")

    return out


def filter_min_history(df: pd.DataFrame, min_history: int) -> pd.DataFrame:
    """
    Remove tickers com histórico muito curto (e também remove o começo onde rolling ainda é NaN).
    """
    counts = df.groupby("ticker")["date"].transform("count")
    df = df[counts >= min_history].copy()
    return df


def finalize_dataset(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Remove linhas com NaNs críticos (por causa de rolling) e mantém colunas úteis.
    """
    # Exige pelo menos algumas features principais e targets
    must_have = [
        "ret_1d",
        f"rsi_{cfg.rsi_window}",
        "vol_21",
        "price_to_ma_21",
        "mom_21d",
    ]
    for c in must_have:
        if c not in df.columns:
            raise ValueError(f"Coluna esperada não encontrada: {c}")

    # remove linhas onde features-chave ainda são NaN
    df2 = df.dropna(subset=must_have).copy()

    # (targets podem ser NaN nas últimas linhas por ticker — removemos quando for treinar)
    df2 = df2.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df2


def save_outputs(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    parquet_path = os.path.join(cfg.out_dir, f"{cfg.out_base}.parquet")
    csv_path = os.path.join(cfg.out_dir, f"{cfg.out_base}.csv")

    # Parquet (melhor)
    df.to_parquet(parquet_path, index=False)

    # CSV (compatibilidade)
    df.to_csv(csv_path, index=False)

    print(f"Salvo: {parquet_path}")
    print(f"Salvo: {csv_path}")
    print(f"Linhas: {len(df):,} | Tickers: {df['ticker'].nunique()}")


def main():
    cfg = FeatureConfig()

    df = load_long_dataset(cfg.input_path)
    df = filter_min_history(df, cfg.min_history)

    df = build_features(df, cfg)
    df = add_targets(df, cfg, which="both")

    df = finalize_dataset(df, cfg)
    save_outputs(df, cfg)


if __name__ == "__main__":
    main()
