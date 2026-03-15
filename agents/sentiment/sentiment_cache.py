"""
TTL cache for sentiment scores.
Prevents re-scoring the same article/symbol within the cache window.
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    symbol: str
    score: float           # -1.0 to +1.0
    source_count: int
    computed_at: datetime


class SentimentCache:
    """
    Two-level cache:
      1. Per-text hash (5 min TTL): avoid re-scoring same article
      2. Per-symbol aggregated score (5 min TTL): orchestrator reads this
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._text_cache: dict[str, tuple[float, float]] = {}    # hash → (score, expires_at)
        self._symbol_cache: dict[str, tuple[SentimentScore, float]] = {}  # symbol → (score, expires_at)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_text_score(self, text: str) -> Optional[float]:
        h = self._hash(text)
        entry = self._text_cache.get(h)
        if entry and time.time() < entry[1]:
            return entry[0]
        return None

    def set_text_score(self, text: str, score: float):
        h = self._hash(text)
        self._text_cache[h] = (score, time.time() + self._ttl)

    def get_symbol_score(self, symbol: str) -> Optional[SentimentScore]:
        entry = self._symbol_cache.get(symbol)
        if entry and time.time() < entry[1]:
            return entry[0]
        return None

    def set_symbol_score(self, score: SentimentScore):
        self._symbol_cache[score.symbol] = (score, time.time() + self._ttl)

    def invalidate_symbol(self, symbol: str):
        self._symbol_cache.pop(symbol, None)

    def text_hash(self, text: str) -> str:
        return self._hash(text)
