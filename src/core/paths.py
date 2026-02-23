from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../src/core -> .../ (repo root)
DATA_DIR = ROOT / "data"

def raw_prices_path(market: str) -> Path:
    return DATA_DIR / "raw" / market.upper() / "prices.parquet"

def clean_prices_path(market: str) -> Path:
    return DATA_DIR / "clean" / market.upper() / "prices.parquet"