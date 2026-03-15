"""
NewsAPI client: fetches recent financial headlines for a given symbol.
Free tier: 100 requests/day — cache aggressively.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class NewsAPIClient:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = None
        if api_key:
            try:
                from newsapi import NewsApiClient
                self._client = NewsApiClient(api_key=api_key)
            except ImportError:
                logger.warning("newsapi-python not installed")

    def get_headlines(self, symbol: str, minutes: int = 30) -> list[str]:
        """
        Return a list of recent headline strings for the symbol.
        Returns empty list on any error (non-blocking).
        """
        if not self._client:
            return []

        # Map ticker → search query
        query = self._symbol_to_query(symbol)

        from_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        try:
            response = self._client.get_everything(
                q=query,
                from_param=from_time.strftime("%Y-%m-%dT%H:%M:%S"),
                language="en",
                sort_by="publishedAt",
                page_size=10,
            )
            articles = response.get("articles", [])
            headlines = []
            for a in articles:
                title = a.get("title", "")
                desc = a.get("description", "")
                if title:
                    headlines.append(f"{title}. {desc}".strip(". "))
            return headlines
        except Exception as exc:
            logger.warning("NewsAPI error for %s: %s", symbol, exc)
            return []

    def _symbol_to_query(self, symbol: str) -> str:
        # Forex: EUR_USD → "EUR USD forex"
        if "_" in symbol:
            parts = symbol.split("_")
            return f"{parts[0]} {parts[1]} forex currency"
        # ETFs
        etf_map = {"SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF"}
        if symbol in etf_map:
            return etf_map[symbol]
        # Stocks: just use the ticker
        return symbol
