from abc import ABC, abstractmethod
import pandas as pd

class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_prices(self, tickers: list[str], start: str, end: str | None, interval: str) -> pd.DataFrame:
        """
        Retorna DataFrame no schema padrão:
        columns: date, ticker, open, high, low, close, adj_close, volume, market
        """
        raise NotImplementedError