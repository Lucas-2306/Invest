import logging
import re

from domain.models import CompanyUpsert, SymbolUpsert
from apps.ingestion.providers.brapi_provider import BrapiProvider
from apps.ingestion.repositories.company_repository import CompanyRepository
from apps.ingestion.repositories.symbol_repository import SymbolRepository

logger = logging.getLogger(__name__)


def run(session) -> None:
    provider = BrapiProvider()
    company_repo = CompanyRepository(session)
    symbol_repo = SymbolRepository(session)

    symbols = provider.list_symbols()

    symbols = [s for s in symbols if re.match(r"^[A-Z]{4}\d{1,2}$", s)]

    logger.info("Total de símbolos encontrados: %s", len(symbols))

    for symbol in symbols:
        try:
            company = CompanyUpsert(company_name=symbol)
            company_id = company_repo.upsert(company)

            symbol_data = SymbolUpsert(
                symbol=symbol,
                company_name=symbol,
                asset_type="stock",
                exchange="B3",
                currency="BRL",
                is_active=True,
            )

            symbol_repo.upsert(symbol_data, company_id)
            session.commit()

        except Exception as exc:
            session.rollback()
            logger.exception("Erro ao processar símbolo %s: %s", symbol, exc)