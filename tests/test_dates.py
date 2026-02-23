import pandas as pd
import pytest

@pytest.mark.parametrize("market", ["US", "BR"])
def test_clean_prices_date_and_keys(market: str):
    path = f"data/clean/{market}/prices.parquet"
    df = pd.read_parquet(path)

    # colunas mínimas
    required = {"date", "ticker", "market", "close"}
    missing = required - set(df.columns)
    assert not missing, f"Faltando colunas: {missing}"

    # date parseável e sem NaT
    dates = pd.to_datetime(df["date"], errors="coerce")
    assert dates.notna().all(), "Existem datas não parseáveis (NaT) no dataset clean"

    # market consistente
    assert df["market"].nunique() == 1, "Dataset deve conter apenas 1 market"
    assert df["market"].iloc[0] == market, f"market no arquivo não bate: esperado {market}"

    # chave única (market, ticker, date)
    dup = df.duplicated(subset=["market", "ticker", "date"]).sum()
    assert dup == 0, f"Há {dup} duplicatas na chave (market,ticker,date)"

    # ordenação por ticker/date (não precisa ser perfeito global, mas por ticker sim)
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    # verifica se dentro de cada ticker está não-decrescente
    bad = 0
    for _, g in df2.groupby("ticker", sort=False):
        if not g["date"].is_monotonic_increasing:
            bad += 1

    assert bad == 0, f"{bad} tickers com datas fora de ordem no arquivo"
