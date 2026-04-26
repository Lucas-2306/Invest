from core.db import SessionLocal
from core.logging import setup_logging
from apps.ingestion.services.market_features import run

setup_logging()

session = SessionLocal()
try:
    run(session)
finally:
    session.close()