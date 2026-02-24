import pandas as pd

def add_momentum(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"mom_{w}"] = out.groupby("ticker")["close"].pct_change(w)
    return out
