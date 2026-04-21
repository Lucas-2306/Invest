from core.db import SessionLocal
from core.logging import setup_logging
from apps.ingestion.services.compute_features import run


def main():
    setup_logging()
    session = SessionLocal()

    try:
        run(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()