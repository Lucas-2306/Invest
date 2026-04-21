import logging
from datetime import date, timedelta

from domain.models import DailyPriceUpsert
from apps.ingestion.providers.yahoo_provider import YahooProvider
from apps.ingestion.repositories.price_repository import PriceRepository
from apps.ingestion.repositories.symbol_repository import SymbolRepository

logger = logging.getLogger(__name__)


def run(
    session,
    period: str = "max",
    interval: str = "1d",
    overlap_days: int = 5,
    force_full_reload: bool = False,
) -> None:
    provider = YahooProvider()
    symbol_repo = SymbolRepository(session)
    price_repo = PriceRepository(session)

    symbol_ids = symbol_repo.list_symbol_ids()
    last_trade_dates = price_repo.get_last_trade_dates()

    symbols = sorted(symbol_ids.keys())
    logger.info("Iniciando ingestão de preços diários para %s símbolos", len(symbols))

    processed = 0

    for symbol in symbols:
        try:
            symbol_id = symbol_ids[symbol]
            last_trade_date = last_trade_dates.get(symbol_id)

            if last_trade_date is None or force_full_reload:
                logger.info("Carga completa para %s com período %s", symbol, period)
                history = provider.get_daily_history(
                    symbol=symbol,
                    period=period,
                    interval=interval,
                )
            else:
                start_date = last_trade_date - timedelta(days=overlap_days)
                end_date = date.today()

                history = provider.get_daily_history(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                )

            if not history:
                continue

            for item in history:
                price = DailyPriceUpsert(
                    symbol=symbol,
                    trade_date=item["trade_date"],
                    open_price=item.get("open_price"),
                    high_price=item.get("high_price"),
                    low_price=item.get("low_price"),
                    close_price=item.get("close_price"),
                    adjusted_close_price=item.get("adjusted_close_price"),
                    volume=item.get("volume"),
                    source="yahoo",
                )
                price_repo.upsert(symbol_id, price)

            session.commit()
            processed += 1

            if processed % 25 == 0:
                logger.info("Processados %s/%s símbolos", processed, len(symbols))

        except Exception as exc:
            session.rollback()
            logger.exception("Erro ao ingerir histórico de %s: %s", symbol, exc)