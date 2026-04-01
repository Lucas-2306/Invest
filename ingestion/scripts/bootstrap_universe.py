from db import SessionLocal
from logging_config import setup_logging
from services.bootstrap_universe import run

setup_logging()

session = SessionLocal()
try:
    run(session)
finally:
    session.close()