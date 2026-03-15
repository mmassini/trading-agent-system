"""
X (Twitter) API v2 client for FinTwit sentiment.
Optional: if TWITTER_BEARER_TOKEN is not set, returns empty list silently.
X API Basic ($100/mo) required for search access.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TwitterClient:
    def __init__(self, bearer_token: str = ""):
        self._bearer_token = bearer_token
        self._client = None

        if bearer_token:
            try:
                import tweepy
                self._client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
                logger.info("Twitter/X client initialized")
            except ImportError:
                logger.warning("tweepy not installed — Twitter disabled")

    @property
    def is_enabled(self) -> bool:
        return self._client is not None

    def get_recent_tweets(self, symbol: str, max_results: int = 20) -> list[str]:
        """
        Return recent tweet texts mentioning the symbol.
        Returns empty list if not configured or on error.
        """
        if not self._client:
            return []

        query = self._build_query(symbol)

        try:
            response = self._client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=["text", "created_at"],
            )
            if response.data is None:
                return []
            return [tweet.text for tweet in response.data]
        except Exception as exc:
            logger.warning("Twitter search error for %s: %s", symbol, exc)
            return []

    def _build_query(self, symbol: str) -> str:
        if "_" in symbol:
            # Forex: EUR_USD → "$EURUSD OR EUR/USD"
            parts = symbol.split("_")
            return f"${parts[0]}{parts[1]} OR {parts[0]}/{parts[1]} lang:en -is:retweet"
        return f"${symbol} lang:en -is:retweet"
