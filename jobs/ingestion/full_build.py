from core.db import SessionLocal
from core.logging import setup_logging
from apps.ingestion.services.bootstrap_universe import run as bootstrap_universe
from apps.ingestion.services.enrich_companies import run as enrich_companies
from apps.ingestion.services.ingest_daily_prices import run as ingest_prices
from apps.ingestion.services.compute_features import run as compute_features
from apps.ingestion.services.market_features import run as market_features


def run_step(name, fn):
    session = SessionLocal()
    try:
        print(f"\n=== Running {name} ===")
        fn(session)
    except Exception as e:
        print(f"Erro na etapa {name}: {e}")
    finally:
        session.close()


def main() -> None:
    setup_logging()

    run_step("bootstrap_universe", bootstrap_universe)
    run_step("enrich_companies", enrich_companies)
    run_step("ingest_daily_prices", ingest_prices)
    run_step("ingest_market_features", market_features)
    run_step("compute_features", compute_features)


if __name__ == "__main__":
    main()