from pathlib import Path
import pandas as pd

from src.core.paths import DATA_DIR

def load_features(market: str) -> pd.DataFrame:
    path = DATA_DIR / "features" / market.upper() / "features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)

def make_xy(df: pd.DataFrame):
    y = df["target_ret_fwd"].astype("float64")
    X = df.drop(columns=["target_ret_fwd"])
    # Features numéricas = tudo exceto chaves
    drop_keys = ["date", "ticker", "market"]
    feat_cols = [c for c in X.columns if c not in drop_keys]
    X_num = X[feat_cols].astype("float64")
    meta = df[["date", "ticker", "market"]].copy()
    return X_num, y, meta, feat_cols
