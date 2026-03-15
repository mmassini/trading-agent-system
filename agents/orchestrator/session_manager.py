"""
Session manager: controls when the system is allowed to place new trades.

Rules:
  - US Stocks: NYSE/NASDAQ open 13:30-20:00 UTC, Mon-Fri
  - Blackout: first and last 5 minutes of each session
  - Forex (OANDA): 24/5 but avoid Sunday open gap and Friday close
  - Daily drawdown halt overrides everything
"""
import logging
from datetime import datetime, time, timezone

import pytz

logger = logging.getLogger(__name__)

# Market sessions (UTC)
NYSE_OPEN_UTC = time(13, 30)
NYSE_CLOSE_UTC = time(20, 0)
BLACKOUT_MINUTES = 5

ET_TZ = pytz.timezone("America/New_York")


class SessionManager:
    def __init__(self, blackout_minutes: int = BLACKOUT_MINUTES):
        self._blackout = blackout_minutes

    def is_stock_session_active(self) -> bool:
        """
        Return True if US stock market is open and we're outside blackout windows.
        """
        now = datetime.now(timezone.utc)

        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        now_time = now.time().replace(second=0, microsecond=0)

        # NYSE hours (UTC): 13:30 – 20:00
        open_time = time(NYSE_OPEN_UTC.hour, NYSE_OPEN_UTC.minute + self._blackout)
        close_time = time(NYSE_CLOSE_UTC.hour, NYSE_CLOSE_UTC.minute - self._blackout)

        if open_time <= now_time <= close_time:
            return True

        return False

    def is_forex_session_active(self) -> bool:
        """
        Return True for forex trading. Avoids Sunday open gap and Friday close.
        Forex is 24/5 but liquidity is low on Sunday night open.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Mon, 6=Sun
        now_time = now.time()

        # Saturday: fully closed
        if weekday == 5:
            return False

        # Sunday: avoid open gap (closed until 21:00 UTC)
        if weekday == 6 and now_time < time(21, 0):
            return False

        # Friday: close early (21:00 UTC = US market close)
        if weekday == 4 and now_time >= time(21, 0):
            return False

        return True

    def is_tradeable_for_symbol(self, symbol: str) -> bool:
        """
        Return True if we can trade this symbol right now.
        Forex symbols contain underscore (e.g., EUR_USD).
        """
        if "_" in symbol:
            return self.is_forex_session_active()
        return self.is_stock_session_active()

    def minutes_to_stock_open(self) -> int:
        """Returns minutes until next US stock session open. 0 if currently open."""
        if self.is_stock_session_active():
            return 0

        now = datetime.now(timezone.utc)
        next_open = now.replace(
            hour=NYSE_OPEN_UTC.hour,
            minute=NYSE_OPEN_UTC.minute + self._blackout,
            second=0,
            microsecond=0,
        )

        # If past today's open, move to next business day
        if now.time() >= time(NYSE_CLOSE_UTC.hour, NYSE_CLOSE_UTC.minute):
            from datetime import timedelta
            next_open += timedelta(days=1)
            # Skip weekend
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)

        delta = (next_open - now).total_seconds() / 60
        return max(int(delta), 0)
