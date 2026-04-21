from __future__ import annotations

import logging
from typing import Any

import requests
from requests import HTTPError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from core.config import settings
from apps.ingestion.providers.base import CompanyProfileProvider, MarketUniverseProvider

logger = logging.getLogger(__name__)


class BrapiProvider(MarketUniverseProvider, CompanyProfileProvider):
    def __init__(self) -> None:
        self.base_url = settings.brapi_base_url.rstrip("/")
        self.session = requests.Session()

        if settings.brapi_token:
            self.session.headers.update({"Authorization": f"Bearer {settings.brapi_token}"})

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=30)

        if response.status_code == 400:
            raise HTTPError(response=response)

        try:
            response.raise_for_status()
        except HTTPError:
            logger.error("Erro HTTP %s em %s", response.status_code, response.url)
            raise

        return response.json()

    def list_symbols(self) -> list[str]:
        data = self._get("quote/list", params={"type": "stock"})
        results = data.get("stocks", []) or data.get("indexes", []) or data.get("results", [])

        symbols: list[str] = []
        for item in results:
            symbol = item.get("stock") or item.get("symbol")
            if symbol:
                symbols.append(symbol)

        return sorted(set(symbols))

    def get_quote(self, symbol: str) -> dict[str, Any]:
        data = self._get(f"quote/{symbol}")
        results = data.get("results", [])
        return results[0] if results else {}

    from requests.exceptions import HTTPError

    def get_rich_profile(self, symbol: str) -> dict[str, Any]:
        try:
            return self._fetch_profile(symbol, full=True)

        except HTTPError as e:
            response = e.response

            if response is not None and "MODULES_NOT_AVAILABLE" in response.text:
                logger.warning("Fallback para summaryProfile apenas: %s", symbol)

                return self._fetch_profile(symbol, full=False)

            raise

    def _fetch_profile(self, symbol: str, full: bool) -> dict[str, Any]:
        modules = (
            "summaryProfile,defaultKeyStatistics,financialData"
            if full
            else "summaryProfile"
        )

        data = self._get(
            f"quote/{symbol}",
            params={"modules": modules},
        )

        results = data.get("results", [])
        return results[0] if results else {}
    
    def get_quote_or_profile(self, symbol: str) -> dict[str, Any]:
        return self.get_rich_profile(symbol)