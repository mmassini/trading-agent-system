"""
Aggregates real-time 1-minute bars into 5-minute and 15-minute bars.
Also persists completed bars to SQLite.
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Optional

from agents.data_ingest.data_buffer import BarData, DataBuffer

logger = logging.getLogger(__name__)


class BarAggregator:
    """
    Receives 1-minute bars, builds multi-timeframe bars (5m, 15m),
    pushes all timeframes to the DataBuffer, and optionally persists to DB.
    """

    def __init__(
        self,
        buffer: DataBuffer,
        on_bar_complete: Optional[Callable[[BarData], None]] = None,
    ):
        self._buffer = buffer
        self._on_bar_complete = on_bar_complete  # DB persistence callback
        self._partial: dict[str, dict[str, dict]] = defaultdict(
            lambda: {"5m": None, "15m": None}
        )

    def on_1m_bar(self, bar: BarData):
        """Called for each completed 1-minute bar."""
        # Push 1m bar directly
        self._buffer.push(bar)
        if self._on_bar_complete:
            self._on_bar_complete(bar)

        # Aggregate into 5m and 15m
        for timeframe, period in [("5m", 5), ("15m", 15)]:
            self._aggregate(bar, timeframe, period)

    def _aggregate(self, bar_1m: BarData, timeframe: str, period: int):
        sym = bar_1m.symbol
        partial = self._partial[sym][timeframe]
        minute = bar_1m.timestamp.minute

        # Determine if we're at the start of a new N-minute block
        block_start = (minute // period) * period
        is_new_block = partial is None or partial["block_start"] != block_start

        if is_new_block:
            # Emit previous block if it exists
            if partial is not None:
                completed = BarData(
                    symbol=sym,
                    timeframe=timeframe,
                    open=partial["open"],
                    high=partial["high"],
                    low=partial["low"],
                    close=partial["close"],
                    volume=partial["volume"],
                    timestamp=partial["timestamp"],
                )
                self._buffer.push(completed)
                if self._on_bar_complete:
                    self._on_bar_complete(completed)

            # Start fresh block
            self._partial[sym][timeframe] = {
                "block_start": block_start,
                "open": bar_1m.open,
                "high": bar_1m.high,
                "low": bar_1m.low,
                "close": bar_1m.close,
                "volume": bar_1m.volume,
                "timestamp": bar_1m.timestamp,
            }
        else:
            # Update existing block
            partial["high"] = max(partial["high"], bar_1m.high)
            partial["low"] = min(partial["low"], bar_1m.low)
            partial["close"] = bar_1m.close
            partial["volume"] += bar_1m.volume
