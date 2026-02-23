import argparse
import pandas as pd

from src.core.logging import get_logger
from src.core.io import load_parquet, save_parquet
from src.core.paths import raw_prices_path, clean_prices_path

log = get_logger("clean")

REQUIRED_COLS = ["date","ticker","open","high","low","close","adj_close","volume","market"]

def _sanity_report(df: pd.DataFrame, market: str) -> None:
    log.info(f"[{market}] linhas={len(df)} tickers={df['ticker'].nunique()} datas={df['date'].min().date()}..{df['date'].max().date()}")

    # cobertura por ticker (informativo)
    pivot = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="first")
    coverage = pivot.notna().mean().sort_values()
    worst = coverage.head(10)
    if len(worst) > 0:
        log.info(f"[{market}] pior cobertura (top 10):")
        for t, c in worst.items():
            log.info(f"  {t}: {c:.2%}")

def clean(market: str) -> None:
    market = market.upper()
    in_path = raw_prices_path(market)
    out_path = clean_prices_path(market)

    df = load_parquet(in_path)

    # garante colunas e ordem
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[REQUIRED_COLS].copy()

    # tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df["market"] = df["market"].astype(str)

    for c in ["open","high","low","close","adj_close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop linhas inválidas essenciais
    df = df.dropna(subset=["date","ticker"])

    # duplicatas (mantém a última)
    before = len(df)
    df = df.drop_duplicates(subset=["market","ticker","date"], keep="last")
    log.info(f"[{market}] removidas duplicatas: {before - len(df)}")

    # ordena
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # sanity checks (filtra absurdos)
    invalid_ohlc = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    bad_ohlc = int(invalid_ohlc.fillna(False).sum())
    if bad_ohlc:
        log.info(f"[{market}] removendo linhas com OHLC inválido: {bad_ohlc}")
        df = df[~invalid_ohlc.fillna(False)]

    invalid_vol = df["volume"] < 0
    bad_vol = int(invalid_vol.fillna(False).sum())
    if bad_vol:
        log.info(f"[{market}] removendo linhas com volume negativo: {bad_vol}")
        df = df[~invalid_vol.fillna(False)]

    _sanity_report(df, market)

    save_parquet(df, out_path)
    log.info(f"[{market}] salvo clean em {out_path} | linhas={len(df)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=["US","BR"])
    args = parser.parse_args()
    clean(args.market)

if __name__ == "__main__":
    main()
