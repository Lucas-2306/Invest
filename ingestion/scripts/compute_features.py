from db import SessionLocal
from logging_config import setup_logging
from services.compute_features import run


def main():
    setup_logging()
    session = SessionLocal()

    try:
        run(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()