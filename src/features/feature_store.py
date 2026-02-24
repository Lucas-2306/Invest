import argparse
from pathlib import Path
import yaml
import pandas as pd

from src.core.logging import get_logger
from src.core.io import load_parquet, save_parquet
from src.core.paths import DATA_DIR

from src.features.transforms.returns import add_return_lags
from src.features.transforms.volatility import add_volatility
from src.features.transforms.momentum import add_momentum
from src.features.transforms.volume import add_dollar_volume

log = get_logger("features")

def features_path(market: str) -> Path:
    return DATA_DIR / "features" / market.upper() / "features.parquet"

def load_cfg(path: str = "configs/features.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_features(market: str) -> pd.DataFrame:
    market = market.upper()
    prices_path = DATA_DIR / "clean" / market / "prices.parquet"
    df = load_parquet(prices_path)

    # garantir tipos e ordenação
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cfg = load_cfg()
    horizon = int(cfg["label"]["horizon_days"])

    lags = cfg["features"]["return_lags"]
    vol_w = cfg["features"]["vol_windows"]
    mom_w = cfg["features"]["mom_windows"]
    dv_w = cfg["features"]["dollarvol_windows"]

    df = add_return_lags(df, lags=lags)
    df = add_volatility(df, windows=vol_w)
    df = add_momentum(df, windows=mom_w)
    df = add_dollar_volume(df, windows=dv_w)

    # target: retorno futuro
    df["target_ret_fwd"] = df.groupby("ticker")["close"].pct_change(horizon).shift(-horizon)

    # mantém colunas úteis
    feature_cols = [c for c in df.columns if c.startswith(("ret_lag_", "vol_", "mom_", "log_dvol_ma_"))]
    keep = ["date", "ticker", "market", "target_ret_fwd"] + feature_cols

    out = df[keep].copy()

    # drop rows sem features/target (rolling + shift gera NaN)
    before = len(out)
    out = out.dropna()
    log.info(f"[{market}] build_features: {before} -> {len(out)} após dropna")

    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["US", "BR"])
    args = p.parse_args()

    df = build_features(args.market)
    out_path = features_path(args.market)
    save_parquet(df, out_path)
    log.info(f"[{args.market}] salvo features em {out_path} | linhas={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
