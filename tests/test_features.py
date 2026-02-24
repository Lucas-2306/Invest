import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize("market", ["US", "BR"])
def test_features_basic_structure(market):
    df = pd.read_parquet(f"data/features/{market}/features.parquet")

    # colunas essenciais
    required = {"date", "ticker", "market", "target_ret_fwd"}
    missing = required - set(df.columns)
    assert not missing, f"Faltando colunas: {missing}"

    # target não deve ser todo zero ou NaN
    assert df["target_ret_fwd"].notna().all(), "Target contém NaN"
    assert df["target_ret_fwd"].std() > 0, "Target parece constante"

    # não deve haver NaNs nas features
    feature_cols = [c for c in df.columns if c not in ["date", "ticker", "market", "target_ret_fwd"]]
    assert not df[feature_cols].isna().any().any(), "Há NaNs nas features"


@pytest.mark.parametrize("market", ["US", "BR"])
def test_target_is_forward_return(market):
    """
    Verifica se target corresponde a retorno futuro t+1
    """
    prices = pd.read_parquet(f"data/clean/{market}/prices.parquet")
    features = pd.read_parquet(f"data/features/{market}/features.parquet")

    prices["date"] = pd.to_datetime(prices["date"])
    features["date"] = pd.to_datetime(features["date"])

    # pega um ticker aleatório
    ticker = features["ticker"].iloc[0]

    p = prices[prices["ticker"] == ticker].sort_values("date").copy()
    f = features[features["ticker"] == ticker].sort_values("date").copy()

    # calcula retorno futuro manual
    p["manual_target"] = p["close"].pct_change(1).shift(-1)

    merged = pd.merge(
        f[["date", "target_ret_fwd"]],
        p[["date", "manual_target"]],
        on="date",
        how="inner",
    )

    diff = (merged["target_ret_fwd"] - merged["manual_target"]).abs().mean()

    assert diff < 1e-10, "Target não corresponde ao retorno futuro esperado"


@pytest.mark.parametrize("market", ["US", "BR"])
def test_no_leakage_in_features(market):
    """
    Verifica se nenhuma feature tem correlação absurda com target (indicando leakage grosseiro)
    """
    df = pd.read_parquet(f"data/features/{market}/features.parquet")

    feature_cols = [c for c in df.columns if c not in ["date", "ticker", "market", "target_ret_fwd"]]

    corrs = df[feature_cols + ["target_ret_fwd"]].corr(numeric_only=True)["target_ret_fwd"].drop("target_ret_fwd")

    # Se houver algo com correlação > 0.9 é quase certeza de vazamento
    assert (corrs.abs() < 0.9).all(), "Possível data leakage detectado"
