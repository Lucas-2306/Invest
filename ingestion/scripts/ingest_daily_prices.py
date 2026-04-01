from db import SessionLocal
from logging_config import setup_logging
from services.ingest_daily_prices import run

setup_logging()

session = SessionLocal()
try:
    run(session, period="max", interval="1d", overlap_days=5, force_full_reload=False)
finally:
    session.close()