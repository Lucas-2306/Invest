from core.db import SessionLocal
from core.logging import setup_logging
from apps.ingestion.services.ingest_daily_prices import run

setup_logging()

session = SessionLocal()
try:
    run(session, period="max", interval="1d", overlap_days=5, force_full_reload=False)
finally:
    session.close()