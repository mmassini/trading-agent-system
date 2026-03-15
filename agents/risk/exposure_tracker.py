"""
Tracks concurrent open positions and enforces the 3-position limit.
"""
import logging

from storage.database import Database

logger = logging.getLogger(__name__)

MAX_POSITIONS = 3


class ExposureTracker:
    def __init__(self, db: Database, max_positions: int = MAX_POSITIONS):
        self._db = db
        self._max = max_positions

    def can_open_new_position(self) -> bool:
        """Return True if another position can be opened."""
        count = self._db.count_open_trades()
        allowed = count < self._max
        if not allowed:
            logger.info("Position limit reached: %d/%d open", count, self._max)
        return allowed

    def open_count(self) -> int:
        return self._db.count_open_trades()

    def slots_available(self) -> int:
        return max(0, self._max - self.open_count())
