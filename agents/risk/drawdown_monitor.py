"""
Daily drawdown monitor: halts trading if the portfolio drops 6% from session open.
"""
import logging
from typing import Optional

from storage.database import Database

logger = logging.getLogger(__name__)

MAX_DAILY_DRAWDOWN = 0.06  # 6%


class DrawdownMonitor:
    """
    Tracks daily P&L and signals when the daily drawdown limit is breached.
    The orchestrator checks this before every new trade.
    """

    def __init__(self, db: Database, max_drawdown: float = MAX_DAILY_DRAWDOWN):
        self._db = db
        self._max_drawdown = max_drawdown
        self._halt_notified = False

    def is_halt_triggered(self) -> bool:
        """
        Return True if daily drawdown has exceeded the limit.
        When True, the orchestrator must not open new positions.
        """
        drawdown = self.current_drawdown_pct()
        if drawdown is None:
            return False  # No data yet → allow trading

        if drawdown >= self._max_drawdown:
            if not self._halt_notified:
                logger.warning(
                    "DRAWDOWN HALT: daily drawdown %.2f%% >= limit %.2f%%",
                    drawdown * 100,
                    self._max_drawdown * 100,
                )
                self._halt_notified = True
            return True

        self._halt_notified = False
        return False

    def current_drawdown_pct(self) -> Optional[float]:
        """
        Returns current daily drawdown as a fraction (0.0 – 1.0).
        Returns None if no session-open balance is available.
        """
        start_balance = self._db.get_balance_at_session_open()
        current_balance = self._db.get_latest_balance("combined")

        if start_balance is None or current_balance is None:
            return None

        if start_balance <= 0:
            return None

        drawdown = (start_balance - current_balance) / start_balance
        return max(drawdown, 0.0)  # Negative drawdown (profit) returns 0

    def remaining_risk_budget(self) -> Optional[float]:
        """
        Returns remaining daily loss budget as a dollar amount.
        Useful for reporting.
        """
        start_balance = self._db.get_balance_at_session_open()
        if start_balance is None:
            return None

        max_loss = start_balance * self._max_drawdown
        drawdown_pct = self.current_drawdown_pct() or 0.0
        current_loss = start_balance * drawdown_pct
        return max(max_loss - current_loss, 0.0)
