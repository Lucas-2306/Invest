# src/data_loader.py
# Baixa dados OHLCV da brapi.dev e salva em data/raw
# - 1 arquivo por ticker
# - 1 arquivo consolidado (long format)

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Iterable

import requests
import pandas as pd


Interval = Literal["1d", "1wk", "1mo"]


# Top 40 do Ibovespa por participação (snapshot do ranking exibido no StatusInvest).
# Fonte: https://statusinvest.com.br/indices/ibovespa (Composição do IBOV - Por papel)
# OBS: A carteira muda periodicamente; se quiser manter sempre atualizado, você pode trocar a fonte depois.
IBOV_TOP40 = [
    "VALE3",
    "ITUB4",
    "PETR4",
    "PETR3",
    "AXIA3",
    "BBDC4",
    "SBSP3",
    "B3SA3",
    "ITSA4",
    "BPAC11",
    "WEGE3",
    "BBAS3",
    "ABEV3",
    "EMBJ3",
    "EQTL3",
    "RENT3",
    "RDOR3",
    "PRIO3",
    "ENEV3",
    "VBBR3",
    "SUZB3",
    "RADL3",
    "AXIA7",
    "VIVT3",
    "UGPA3",
    "GGBR4",
    "BBDC3",
    "CPLE3",
    "CMIG4",
    "TIMS3",
    "BBSE3",
    "TOTS3",
    "RAIL3",
    "ENGI11",
    "AXIA6",
    "MOTV3",
    "KLBN11",
    "LREN3",
    "ALOS3",
    "MBRF3",
]


@dataclass
class BrapiConfig:
    api_key: str
    base_url: str = "https://brapi.dev/api/quote"
    timeout: int = 30
    sleep_seconds: float = 0.25  # pequeno delay para evitar rate limit


class BrapiDataLoader:
    def __init__(self, config: BrapiConfig):
        self.config = config

    def fetch_ohlcv(self, ticker: str, range_: str = "10y", interval: Interval = "1d") -> pd.DataFrame:
        """
        Baixa OHLCV histórico para um ticker (ex: PETR4).
        Retorna DataFrame com colunas padronizadas:
        date, open, high, low, close, volume, adjustedClose
        """
        url = f"{self.config.base_url}/{ticker}"
        params = {"range": range_, "interval": interval}
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        resp = requests.get(url, params=params, headers=headers, timeout=self.config.timeout)
        resp.raise_for_status()
        payload = resp.json()

        results = payload.get("results")
        if not results:
            raise ValueError(f"Sem resultados para ticker={ticker}. Resposta: {payload}")

        hist = results[0].get("historicalDataPrice")
        if not hist:
            raise ValueError(f"Sem histórico (historicalDataPrice) para ticker={ticker}. Resposta: {payload}")

        df = pd.DataFrame(hist)

        # Normalmente "date" vem em timestamp (segundos)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")

        expected_cols = ["date", "open", "high", "low", "close", "volume", "adjustedClose"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[expected_cols].sort_values("date").reset_index(drop=True)

        # Tipagem numérica
        for col in ["open", "high", "low", "close", "volume", "adjustedClose"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remover linhas inválidas
        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)

        return df

    def save_raw(self, df: pd.DataFrame, ticker: str, out_dir: str = "data/raw") -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{ticker}.csv")
        df.to_csv(path, index=False)
        return path

    def fetch_and_save(self, ticker: str, range_: str = "10y", interval: Interval = "1d", out_dir: str = "data/raw") -> str:
        df = self.fetch_ohlcv(ticker=ticker, range_=range_, interval=interval)
        return self.save_raw(df, ticker=ticker, out_dir=out_dir)

    def fetch_many(
        self,
        tickers: Iterable[str],
        range_: str = "10y",
        interval: Interval = "1d",
        out_dir: str = "data/raw",
        consolidated_filename: str = "ibov_top40_long.csv",
    ) -> pd.DataFrame:
        """
        Baixa vários tickers, salva 1 CSV por ticker e também um CSV consolidado (long format).
        Retorna o DataFrame consolidado.
        """
        os.makedirs(out_dir, exist_ok=True)

        all_frames: list[pd.DataFrame] = []
        failures: list[tuple[str, str]] = []

        for i, ticker in enumerate(tickers, start=1):
            try:
                df = self.fetch_ohlcv(ticker=ticker, range_=range_, interval=interval)
                self.save_raw(df, ticker=ticker, out_dir=out_dir)

                df_long = df.copy()
                df_long.insert(1, "ticker", ticker)  # date, ticker, ...
                all_frames.append(df_long)

                print(f"[{i}] OK  - {ticker} ({len(df)} linhas)")
            except Exception as e:
                failures.append((ticker, str(e)))
                print(f"[{i}] ERR - {ticker} -> {e}")

            time.sleep(self.config.sleep_seconds)

        if not all_frames:
            raise RuntimeError(f"Falhou para todos os tickers. Exemplo de erro: {failures[:1]}")

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

        consolidated_path = os.path.join(out_dir, consolidated_filename)
        combined.to_csv(consolidated_path, index=False)
        print(f"\nConsolidado salvo em: {consolidated_path}")

        if failures:
            fail_path = os.path.join(out_dir, "failures_top40.txt")
            with open(fail_path, "w", encoding="utf-8") as f:
                for t, msg in failures:
                    f.write(f"{t}\t{msg}\n")
            print(f"Falhas registradas em: {fail_path} ({len(failures)} tickers)")

        return combined


def load_api_key_from_env(env_var: str = "BRAPI_API_KEY") -> str:
    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(
            f"API key não encontrada em {env_var}. "
            f"Defina a variável de ambiente (export {env_var}=...) ou use um arquivo .env carregado pelo seu projeto."
        )
    return key


if __name__ == "__main__":
    # Exemplo: python -m src.data_loader
    api_key = load_api_key_from_env()
    loader = BrapiDataLoader(BrapiConfig(api_key=api_key))

    # Ajuste aqui se quiser:
    RANGE = "10y"     # "5y", "10y", "max" (dependendo do suporte)
    INTERVAL: Interval = "1d"

    loader.fetch_many(
        tickers=IBOV_TOP40,
        range_=RANGE,
        interval=INTERVAL,
        out_dir="data/raw",
        consolidated_filename="ibov_top40_long.csv",
    )
