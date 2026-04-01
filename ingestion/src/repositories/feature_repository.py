from sqlalchemy import text
from sqlalchemy.orm import Session


class FeatureRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_last_feature_date(self, symbol_id: str):
        sql = text("""
            SELECT MAX(trade_date)
            FROM market_data.features
            WHERE symbol_id = :symbol_id
        """)
        row = self.session.execute(sql, {"symbol_id": symbol_id}).fetchone()
        return row[0] if row and row[0] is not None else None

    def upsert_dataframe(self, symbol_id: str, df) -> None:
        records = []

        for _, row in df.iterrows():
            records.append({
                "symbol_id": symbol_id,
                "trade_date": row["trade_date"],
                "return_1d": row.get("return_1d"),
                "return_5d": row.get("return_5d"),
                "return_21d": row.get("return_21d"),
                "log_return_1d": row.get("log_return_1d"),
                "sma_20": row.get("sma_20"),
                "sma_50": row.get("sma_50"),
                "price_sma_20_ratio": row.get("price_sma_20_ratio"),
                "volatility_21d": row.get("volatility_21d"),
                "volume_ratio_20d": row.get("volume_ratio_20d"),
                "high_low_ratio": row.get("high_low_ratio"),
                "gap": row.get("gap"),
                "target_5d": row.get("target_5d"),
            })

        if not records:
            return

        sql = text("""
            INSERT INTO market_data.features (
                symbol_id,
                trade_date,
                return_1d,
                return_5d,
                return_21d,
                log_return_1d,
                sma_20,
                sma_50,
                price_sma_20_ratio,
                volatility_21d,
                volume_ratio_20d,
                high_low_ratio,
                gap,
                target_5d,
                created_at
            ) VALUES (
                :symbol_id,
                :trade_date,
                :return_1d,
                :return_5d,
                :return_21d,
                :log_return_1d,
                :sma_20,
                :sma_50,
                :price_sma_20_ratio,
                :volatility_21d,
                :volume_ratio_20d,
                :high_low_ratio,
                :gap,
                :target_5d,
                NOW()
            )
            ON CONFLICT (symbol_id, trade_date)
            DO UPDATE SET
                return_1d = EXCLUDED.return_1d,
                return_5d = EXCLUDED.return_5d,
                return_21d = EXCLUDED.return_21d,
                log_return_1d = EXCLUDED.log_return_1d,
                sma_20 = EXCLUDED.sma_20,
                sma_50 = EXCLUDED.sma_50,
                price_sma_20_ratio = EXCLUDED.price_sma_20_ratio,
                volatility_21d = EXCLUDED.volatility_21d,
                volume_ratio_20d = EXCLUDED.volume_ratio_20d,
                high_low_ratio = EXCLUDED.high_low_ratio,
                gap = EXCLUDED.gap,
                target_5d = EXCLUDED.target_5d,
                created_at = NOW()
        """)

        self.session.execute(sql, records)