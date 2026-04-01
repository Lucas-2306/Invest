from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooProvider:
    def _to_yahoo_symbol(self, symbol: str) -> str:
        return f"{symbol}.SA"

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            flattened = []
            for col in df.columns:
                parts = [str(part) for part in col if part is not None and str(part) != ""]
                flattened.append("_".join(parts))
            df.columns = flattened
        else:
            df.columns = [str(col) for col in df.columns]
        return df

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        normalized = {str(col).lower(): str(col) for col in df.columns}

        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]

        for col in df.columns:
            col_lower = str(col).lower()
            for candidate in candidates:
                if candidate.lower() in col_lower:
                    return str(col)

        return None

    def get_daily_history(
        self,
        symbol: str,
        period: str = "max",
        interval: str = "1d",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        yahoo_symbol = self._to_yahoo_symbol(symbol)

        download_kwargs: dict[str, Any] = {
            "tickers": yahoo_symbol,
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
            "threads": False,
            "group_by": "column",
        }

        if start_date is not None:
            download_kwargs["start"] = start_date.isoformat()
            # yfinance trata end como exclusivo; somamos 1 dia para incluir a data final desejada
            effective_end = end_date if end_date is not None else date.today()
            download_kwargs["end"] = (effective_end + timedelta(days=1)).isoformat()
        else:
            download_kwargs["period"] = period

        df = yf.download(**download_kwargs)

        if df is None or df.empty:
            logger.warning("Yahoo sem dados para %s", symbol)
            return []

        df = self._flatten_columns(df).reset_index()

        date_col = self._find_column(df, ["Date", "Datetime"])
        open_col = self._find_column(df, ["Open"])
        high_col = self._find_column(df, ["High"])
        low_col = self._find_column(df, ["Low"])
        close_col = self._find_column(df, ["Close"])
        adj_close_col = self._find_column(df, ["Adj Close", "Adj_Close", "adjclose"])
        volume_col = self._find_column(df, ["Volume"])

        if date_col is None:
            raise ValueError(f"Não foi possível localizar a coluna de data para {symbol}. Colunas: {list(df.columns)}")

        def safe_float(value: Any) -> float | None:
            if pd.isna(value):
                return None
            return float(value)

        def safe_int(value: Any) -> int | None:
            if pd.isna(value):
                return None
            return int(value)

        records: list[dict[str, Any]] = []

        for row in df.to_dict(orient="records"):
            trade_date = row.get(date_col)

            if trade_date is None or pd.isna(trade_date):
                continue

            if hasattr(trade_date, "to_pydatetime"):
                trade_date = trade_date.to_pydatetime().date()
            elif hasattr(trade_date, "date"):
                trade_date = trade_date.date()

            records.append(
                {
                    "trade_date": trade_date,
                    "open_price": safe_float(row.get(open_col)) if open_col else None,
                    "high_price": safe_float(row.get(high_col)) if high_col else None,
                    "low_price": safe_float(row.get(low_col)) if low_col else None,
                    "close_price": safe_float(row.get(close_col)) if close_col else None,
                    "adjusted_close_price": safe_float(row.get(adj_close_col)) if adj_close_col else None,
                    "volume": safe_int(row.get(volume_col)) if volume_col else None,
                }
            )

        logger.info("Yahoo retornou %s linhas para %s", len(records), symbol)
        return records