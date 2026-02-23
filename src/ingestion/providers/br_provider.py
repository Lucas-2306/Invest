import pandas as pd
import yfinance as yf

from .base import MarketDataProvider
from .us_provider import _normalize_yf_download, _empty_prices_df  # reuso

class YahooBRProvider(MarketDataProvider):
    market = "BR"

    def fetch_prices(self, tickers: list[str], start: str, end: str | None, interval: str = "1d") -> pd.DataFrame:
        if not tickers:
            return _empty_prices_df()

        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        return _normalize_yf_download(df, tickers=tickers, market=self.market)
