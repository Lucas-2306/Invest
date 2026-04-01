from sqlalchemy import text
from sqlalchemy.orm import Session

from models import SymbolUpsert


class SymbolRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, symbol_data: SymbolUpsert, company_id: str) -> str:
        sql = text("""
            INSERT INTO market_data.symbols (
                company_id, symbol, asset_type, exchange, currency, is_active, last_seen_at, updated_at
            ) VALUES (
                :company_id, :symbol, :asset_type, :exchange, :currency, :is_active, NOW(), NOW()
            )
            ON CONFLICT (symbol)
            DO UPDATE SET
                company_id = EXCLUDED.company_id,
                asset_type = EXCLUDED.asset_type,
                exchange = EXCLUDED.exchange,
                currency = EXCLUDED.currency,
                is_active = EXCLUDED.is_active,
                last_seen_at = NOW(),
                updated_at = NOW()
            RETURNING id
        """)
        payload = {
            "company_id": company_id,
            **symbol_data.__dict__,
        }
        row = self.session.execute(sql, payload).fetchone()
        return str(row[0])

    def list_symbols(self) -> list[str]:
        sql = text("""
            SELECT symbol
            FROM market_data.symbols
            WHERE is_active = TRUE
            ORDER BY symbol
        """)
        rows = self.session.execute(sql).fetchall()
        return [row[0] for row in rows]

    def get_symbol_id(self, symbol: str) -> str | None:
        sql = text("""
            SELECT id
            FROM market_data.symbols
            WHERE symbol = :symbol
            LIMIT 1
        """)
        row = self.session.execute(sql, {"symbol": symbol}).fetchone()
        return str(row[0]) if row else None

    def list_symbols_with_company_ids(self) -> list[dict[str, str]]:
        sql = text("""
            SELECT symbol, id, company_id
            FROM market_data.symbols
            WHERE is_active = TRUE
            ORDER BY symbol
        """)
        rows = self.session.execute(sql).fetchall()
        return [
            {
                "symbol": row[0],
                "symbol_id": str(row[1]),
                "company_id": str(row[2]),
            }
            for row in rows
        ]