from abc import ABC, abstractmethod
from typing import Any


class MarketUniverseProvider(ABC):
    @abstractmethod
    def list_symbols(self) -> list[str]:
        raise NotImplementedError


class CompanyProfileProvider(ABC):
    @abstractmethod
    def get_quote_or_profile(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError


class PriceHistoryProvider(ABC):
    @abstractmethod
    def get_daily_history(self, symbol: str, period: str = "5y", interval: str = "1d") -> list[dict[str, Any]]:
        raise NotImplementedError