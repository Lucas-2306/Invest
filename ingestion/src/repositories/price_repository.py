from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from models import DailyPriceUpsert


class PriceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, symbol_id: str, price: DailyPriceUpsert) -> None:
        sql = text("""
            INSERT INTO market_data.daily_prices (
                symbol_id, trade_date, open_price, high_price, low_price, close_price,
                adjusted_close_price, volume, trades, vwap, source, ingested_at
            ) VALUES (
                :symbol_id, :trade_date, :open_price, :high_price, :low_price, :close_price,
                :adjusted_close_price, :volume, :trades, :vwap, :source, NOW()
            )
            ON CONFLICT (symbol_id, trade_date)
            DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                adjusted_close_price = EXCLUDED.adjusted_close_price,
                volume = EXCLUDED.volume,
                trades = EXCLUDED.trades,
                vwap = EXCLUDED.vwap,
                source = EXCLUDED.source,
                ingested_at = NOW()
        """)
        payload = {
            "symbol_id": symbol_id,
            **price.__dict__,
        }
        self.session.execute(sql, payload)

    def get_last_trade_date(self, symbol_id: str) -> date | None:
        sql = text("""
            SELECT MAX(trade_date)
            FROM market_data.daily_prices
            WHERE symbol_id = :symbol_id
        """)
        row = self.session.execute(sql, {"symbol_id": symbol_id}).fetchone()
        return row[0] if row and row[0] is not None else None

    def get_last_trade_dates(self) -> dict[str, date]:
        sql = text("""
            SELECT symbol_id, MAX(trade_date) AS last_trade_date
            FROM market_data.daily_prices
            GROUP BY symbol_id
        """)
        rows = self.session.execute(sql).fetchall()
        return {str(row[0]): row[1] for row in rows if row[1] is not None}