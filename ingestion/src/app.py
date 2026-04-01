from db import SessionLocal
from logging_config import setup_logging
from services.bootstrap_universe import run as bootstrap_universe
from services.ingest_company_profiles import run as ingest_profiles
from services.ingest_daily_prices import run as ingest_prices


def main() -> None:
    setup_logging()

    session = SessionLocal()
    try:
        bootstrap_universe(session)
        ingest_profiles(session)
        ingest_prices(session, range_="1y", interval="1d")
    finally:
        session.close()


if __name__ == "__main__":
    main()