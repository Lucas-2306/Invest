import pandas as pd
import yfinance as yf

from .base import MarketDataProvider

class YahooUSProvider(MarketDataProvider):
    market = "US"

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


def _empty_prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "market"]
    )


def _normalize_yf_download(df: pd.DataFrame, tickers: list[str], market: str) -> pd.DataFrame:
    """
    Normaliza saída do yfinance.download para schema padrão:
    date, ticker, open, high, low, close, adj_close, volume, market
    """

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    # Multi-ticker: colunas MultiIndex (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        frames: list[pd.DataFrame] = []
        top_level = set(df.columns.get_level_values(0))

        missing = [t for t in tickers if t not in top_level]
        if missing:
            # print simples (Sprint 1); depois a gente usa logger
            print(f"[WARN] faltando no Yahoo ({market}): {missing}")

        for t in tickers:
            if t not in top_level:
                continue
            tmp = df[t].copy()
            tmp = tmp.reset_index()

            # primeira coluna após reset_index costuma ser Date/Datetime
            date_col = tmp.columns[0]
            tmp = tmp.rename(columns={date_col: "date", **rename_map})

            tmp["ticker"] = t
            tmp["market"] = market
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.date.astype(str)

            # garante colunas mesmo se algum campo não vier
            for col in ["open","high","low","close","adj_close","volume"]:
                if col not in tmp.columns:
                    tmp[col] = pd.NA

            frames.append(tmp[["date","ticker","open","high","low","close","adj_close","volume","market"]])

        return pd.concat(frames, ignore_index=True) if frames else _empty_prices_df()

    # Single-ticker: colunas simples
    out = df.copy().reset_index()
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "date", **rename_map})

    out["ticker"] = tickers[0] if tickers else "UNKNOWN"
    out["market"] = market
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)

    for col in ["open","high","low","close","adj_close","volume"]:
        if col not in out.columns:
            out[col] = pd.NA

    return out[["date","ticker","open","high","low","close","adj_close","volume","market"]]
