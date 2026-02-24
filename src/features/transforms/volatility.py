import pandas as pd

def add_volatility(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    if "ret_1d" not in out.columns:
        out["ret_1d"] = out.groupby("ticker")["close"].pct_change(1)

    g = out.groupby("ticker")["ret_1d"]
    for w in windows:
        out[f"vol_{w}"] = g.rolling(w).std().reset_index(level=0, drop=True)
    return out
