"""
Thread-safe in-memory ring buffer for real-time bars.
All agents read from this buffer without touching the DB for low-latency access.
"""
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BarData:
    symbol: str
    timeframe: str      # "1m" | "5m" | "15m"
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


class SymbolBuffer:
    """Circular buffer of BarData for a single symbol+timeframe."""

    def __init__(self, maxlen: int = 200):
        self._data: deque[BarData] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, bar: BarData):
        with self._lock:
            self._data.append(bar)

    def get_latest(self, n: int = 100) -> list[BarData]:
        with self._lock:
            return list(self._data)[-n:]

    def get_last(self) -> Optional[BarData]:
        with self._lock:
            return self._data[-1] if self._data else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class DataBuffer:
    """
    Central registry of per-symbol, per-timeframe ring buffers.

    Usage:
        buffer = DataBuffer()
        buffer.push(bar)
        bars = buffer.get_latest("AAPL", "1m", n=50)
    """

    def __init__(self, maxlen: int = 200):
        self._maxlen = maxlen
        self._buffers: dict[str, SymbolBuffer] = {}
        self._lock = threading.Lock()

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def _get_or_create(self, symbol: str, timeframe: str) -> SymbolBuffer:
        key = self._key(symbol, timeframe)
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SymbolBuffer(maxlen=self._maxlen)
            return self._buffers[key]

    def push(self, bar: BarData):
        self._get_or_create(bar.symbol, bar.timeframe).push(bar)

    def get_latest(self, symbol: str, timeframe: str, n: int = 100) -> list[BarData]:
        return self._get_or_create(symbol, timeframe).get_latest(n)

    def get_last(self, symbol: str, timeframe: str) -> Optional[BarData]:
        return self._get_or_create(symbol, timeframe).get_last()

    def has_enough_data(self, symbol: str, timeframe: str, min_bars: int = 50) -> bool:
        return len(self._get_or_create(symbol, timeframe)) >= min_bars

    def symbols(self) -> list[str]:
        with self._lock:
            seen = set()
            result = []
            for key in self._buffers:
                sym = key.split(":")[0]
                if sym not in seen:
                    seen.add(sym)
                    result.append(sym)
            return result
