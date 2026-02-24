import pandas as pd

def add_return_lags(df, lags):
    out = df.copy()

    # retorno diário
    out["ret_1d"] = out.groupby("ticker")["close"].pct_change(1)

    # lags do retorno diário
    for lag in lags:
        out[f"ret_lag_{lag}"] = (
            out.groupby("ticker")["ret_1d"].shift(lag)
        )

    return out
