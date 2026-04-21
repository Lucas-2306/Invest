from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class CompanyUpsert:
    company_name: str
    trading_name: Optional[str] = None
    sector: Optional[str] = None
    subsector: Optional[str] = None
    segment: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    cnpj: Optional[str] = None


@dataclass
class SymbolUpsert:
    symbol: str
    company_name: str
    asset_type: str = "stock"
    exchange: str = "B3"
    currency: str = "BRL"
    is_active: bool = True


@dataclass
class DailyPriceUpsert:
    symbol: str
    trade_date: date
    open_price: Optional[float]
    high_price: Optional[float]
    low_price: Optional[float]
    close_price: Optional[float]
    adjusted_close_price: Optional[float]
    volume: Optional[int]
    trades: Optional[int] = None
    vwap: Optional[float] = None
    source: str = "brapi"