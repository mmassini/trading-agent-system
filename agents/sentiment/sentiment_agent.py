"""
Sentiment Agent: aggregates news and social signals into a per-symbol score.
Called by the Orchestrator as a lightweight subagent (no Claude SDK needed here —
sentiment is deterministic scoring, not reasoning).
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from agents.sentiment.finbert_scorer import score_text
from agents.sentiment.newsapi_client import NewsAPIClient
from agents.sentiment.sentiment_cache import SentimentCache, SentimentScore
from agents.sentiment.twitter_client import TwitterClient

logger = logging.getLogger(__name__)


class SentimentAgent:
    """
    Produces a SentimentScore for a symbol by:
      1. Fetching headlines from NewsAPI
      2. Fetching tweets from X (if configured)
      3. Scoring each text with FinBERT
      4. Returning a recency-weighted average score

    Results are cached for 5 minutes to stay within API rate limits.
    """

    def __init__(
        self,
        newsapi_key: str = "",
        twitter_bearer_token: str = "",
        cache_ttl: int = 300,
    ):
        self._news = NewsAPIClient(newsapi_key)
        self._twitter = TwitterClient(twitter_bearer_token)
        self._cache = SentimentCache(ttl_seconds=cache_ttl)

    def get_score(self, symbol: str, force_refresh: bool = False) -> SentimentScore:
        """
        Return the aggregated sentiment score for a symbol.
        Uses cache if available and not expired.
        """
        if not force_refresh:
            cached = self._cache.get_symbol_score(symbol)
            if cached is not None:
                return cached

        texts: list[str] = []

        # ── Collect news headlines ──────────────────────────────────────────
        headlines = self._news.get_headlines(symbol, minutes=30)
        texts.extend(headlines)

        # ── Collect tweets (optional) ───────────────────────────────────────
        if self._twitter.is_enabled:
            tweets = self._twitter.get_recent_tweets(symbol, max_results=15)
            texts.extend(tweets)

        if not texts:
            result = SentimentScore(
                symbol=symbol,
                score=0.0,
                source_count=0,
                computed_at=datetime.now(timezone.utc),
            )
            self._cache.set_symbol_score(result)
            return result

        # ── Score each text (with text-level cache) ─────────────────────────
        scores = []
        for text in texts:
            cached_score = self._cache.get_text_score(text)
            if cached_score is not None:
                scores.append(cached_score)
            else:
                s = score_text(text)
                self._cache.set_text_score(text, s)
                scores.append(s)

        # Simple average (all sources weighted equally)
        avg_score = sum(scores) / len(scores) if scores else 0.0

        result = SentimentScore(
            symbol=symbol,
            score=round(avg_score, 4),
            source_count=len(scores),
            computed_at=datetime.now(timezone.utc),
        )
        self._cache.set_symbol_score(result)

        logger.debug(
            "Sentiment %s: %.3f (from %d sources)", symbol, result.score, result.source_count
        )
        return result

    def get_scores_batch(self, symbols: list[str]) -> dict[str, SentimentScore]:
        """Score multiple symbols. Results cached independently per symbol."""
        return {s: self.get_score(s) for s in symbols}
