from core.db import SessionLocal
from core.logging import setup_logging
from apps.ingestion.services.ingest_daily_prices import run as ingest_prices
from apps.ingestion.services.compute_features import run as compute_features


def main() -> None:
    setup_logging()

    session = SessionLocal()
    try:
        print("\n=== Running incremental price ingestion ===")
        ingest_prices(
            session,
            period="max",
            interval="1d",
            overlap_days=5,
            force_full_reload=False,
        )

        print("\n=== Running incremental feature computation ===")
        compute_features(session)

    finally:
        session.close()


if __name__ == "__main__":
    main()