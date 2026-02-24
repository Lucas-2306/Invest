import numpy as np
import pandas as pd

def add_dollar_volume(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    # log(Volume * Close) para estabilizar escala
    dollar_vol = (out["volume"].astype("float64") * out["close"].astype("float64"))
    out["log_dvol"] = np.log(dollar_vol.replace(0, np.nan))

    g = out.groupby("ticker")["log_dvol"]
    for w in windows:
        out[f"log_dvol_ma_{w}"] = g.rolling(w).mean().reset_index(level=0, drop=True)
    return out
