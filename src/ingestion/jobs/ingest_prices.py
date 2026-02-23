import argparse
import yaml

from src.core.logging import get_logger
from src.core.paths import raw_prices_path
from src.core.io import save_parquet
from src.ingestion.providers.us_provider import YahooUSProvider
from src.ingestion.providers.br_provider import YahooBRProvider

log = get_logger("ingest")

def load_config(path: str = "configs/markets.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_provider(market: str):
    market = market.upper()
    if market == "US":
        return YahooUSProvider()
    if market == "BR":
        return YahooBRProvider()
    raise ValueError(f"Market inválido: {market}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=["US", "BR"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    cfg = load_config()
    defaults = cfg["defaults"]
    market_cfg = cfg["markets"][args.market]

    start = args.start or defaults["start_date"]
    end = args.end if args.end is not None else defaults.get("end_date", None)
    interval = defaults.get("interval", "1d")
    tickers = market_cfg["tickers"]

    provider = get_provider(args.market)

    log.info(f"[{args.market}] baixando {len(tickers)} tickers | start={start} end={end} interval={interval}")
    df = provider.fetch_prices(tickers=tickers, start=start, end=end, interval=interval)

    out_path = raw_prices_path(args.market)
    save_parquet(df, out_path)
    log.info(f"[{args.market}] salvo em {out_path} | linhas={len(df)}")

if __name__ == "__main__":
    main()