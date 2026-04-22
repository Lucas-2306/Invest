import logging
from datetime import date

from sqlalchemy import text

from domain.models import CompanyUpsert
from apps.ingestion.providers.brapi_provider import BrapiProvider
from apps.ingestion.repositories.company_repository import CompanyRepository
from apps.ingestion.repositories.symbol_repository import SymbolRepository

logger = logging.getLogger(__name__)


def run(session) -> None:
    provider = BrapiProvider()
    company_repo = CompanyRepository(session)
    symbol_repo = SymbolRepository(session)

    rows = symbol_repo.list_symbols_with_company_ids()
    logger.info("Iniciando enriquecimento de empresas para %s símbolos", len(rows))

    processed = 0

    for row in rows:
        symbol = row["symbol"]
        symbol_id = row["symbol_id"]
        company_id = row["company_id"]

        try:
            profile = provider.get_rich_profile(symbol)
            if not profile:
                logger.warning("Sem profile rico para %s", symbol)
                continue

            summary_profile = profile.get("summaryProfile", {}) or {}
            stats = profile.get("defaultKeyStatistics", {}) or {}
            financial = profile.get("financialData", {}) or {}

            company_payload = CompanyUpsert(
                company_name=profile.get("longName") or profile.get("shortName") or symbol,
                trading_name=profile.get("shortName"),
                sector=summary_profile.get("sector"),
                subsector=summary_profile.get("industry"),
                segment=summary_profile.get("industry"),
                description=summary_profile.get("longBusinessSummary"),
                website=summary_profile.get("website"),
                cnpj=None,
            )

            company_repo.update_company_by_id(company_id, company_payload)

            sql = text("""
                INSERT INTO market_data.fundamentals_snapshot (
                    symbol_id, reference_date, market_cap, enterprise_value, shares_outstanding,
                    price_to_earnings, price_to_book, eps, roe, roa, debt_to_equity,
                    dividend_yield, beta, fifty_two_week_high, fifty_two_week_low, source
                ) VALUES (
                    :symbol_id, :reference_date, :market_cap, :enterprise_value, :shares_outstanding,
                    :price_to_earnings, :price_to_book, :eps, :roe, :roa, :debt_to_equity,
                    :dividend_yield, :beta, :fifty_two_week_high, :fifty_two_week_low, :source
                )
                ON CONFLICT (symbol_id, reference_date)
                DO UPDATE SET
                    market_cap = COALESCE(EXCLUDED.market_cap, market_data.fundamentals_snapshot.market_cap),
                    enterprise_value = COALESCE(EXCLUDED.enterprise_value, market_data.fundamentals_snapshot.enterprise_value),
                    shares_outstanding = COALESCE(EXCLUDED.shares_outstanding, market_data.fundamentals_snapshot.shares_outstanding),
                    price_to_earnings = COALESCE(EXCLUDED.price_to_earnings, market_data.fundamentals_snapshot.price_to_earnings),
                    price_to_book = COALESCE(EXCLUDED.price_to_book, market_data.fundamentals_snapshot.price_to_book),
                    eps = COALESCE(EXCLUDED.eps, market_data.fundamentals_snapshot.eps),
                    roe = COALESCE(EXCLUDED.roe, market_data.fundamentals_snapshot.roe),
                    roa = COALESCE(EXCLUDED.roa, market_data.fundamentals_snapshot.roa),
                    debt_to_equity = COALESCE(EXCLUDED.debt_to_equity, market_data.fundamentals_snapshot.debt_to_equity),
                    dividend_yield = COALESCE(EXCLUDED.dividend_yield, market_data.fundamentals_snapshot.dividend_yield),
                    beta = COALESCE(EXCLUDED.beta, market_data.fundamentals_snapshot.beta),
                    fifty_two_week_high = COALESCE(EXCLUDED.fifty_two_week_high, market_data.fundamentals_snapshot.fifty_two_week_high),
                    fifty_two_week_low = COALESCE(EXCLUDED.fifty_two_week_low, market_data.fundamentals_snapshot.fifty_two_week_low),
                    source = EXCLUDED.source,
                    ingested_at = NOW()
            """)

            provider_reference_date = (
                profile.get("regularMarketTime")
                or profile.get("earningsTimestamp")
                or None
            )

            payload = {
                "symbol_id": symbol_id,
                "reference_date": date.today(),
                "market_cap": profile.get("marketCap"),
                "enterprise_value": stats.get("enterpriseValue"),
                "shares_outstanding": stats.get("sharesOutstanding"),
                "price_to_earnings": profile.get("priceEarnings"),
                "price_to_book": stats.get("priceToBook"),
                "eps": profile.get("earningsPerShare") or stats.get("trailingEps"),
                "roe": financial.get("returnOnEquity"),
                "roa": financial.get("returnOnAssets"),
                "debt_to_equity": financial.get("debtToEquity"),
                "dividend_yield": profile.get("dividendYield"),
                "beta": stats.get("beta"),
                "fifty_two_week_high": profile.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": profile.get("fiftyTwoWeekLow"),
                "source": "brapi",
            }

            session.execute(sql, payload)
            session.commit()

            processed += 1
            if processed % 25 == 0:
                logger.info("Enriquecidos %s/%s símbolos", processed, len(rows))

        except Exception as exc:
            session.rollback()
            logger.exception("Erro ao enriquecer empresa %s: %s", symbol, exc)