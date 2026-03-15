"""
OANDA real-time forex price stream.
Converts tick prices into synthetic 1-minute bars and feeds the BarAggregator.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import oandapyV20
from oandapyV20.endpoints.pricing import PricingStream

from agents.data_ingest.bar_aggregator import BarAggregator
from agents.data_ingest.data_buffer import BarData

logger = logging.getLogger(__name__)


class OandaStreamHandler:
    """
    Streams prices from OANDA for forex instruments.
    Aggregates ticks into 1-minute bars and forwards to the BarAggregator.
    """

    def __init__(
        self,
        access_token: str,
        account_id: str,
        instruments: list[str],
        aggregator: BarAggregator,
        environment: str = "practice",
    ):
        self._access_token = access_token
        self._account_id = account_id
        self._instruments = instruments
        self._aggregator = aggregator
        self._environment = environment
        self._running = False

        # In-progress 1-min bars per instrument
        self._partials: dict[str, dict] = {}

    def _flush_partial(self, instrument: str, current_minute: int):
        """Emit a completed 1-min bar if the minute has changed."""
        partial = self._partials.get(instrument)
        if partial and partial["minute"] != current_minute:
            bar = BarData(
                symbol=instrument,
                timeframe="1m",
                open=partial["open"],
                high=partial["high"],
                low=partial["low"],
                close=partial["close"],
                volume=partial["tick_count"],  # Tick count as volume proxy
                timestamp=partial["timestamp"],
            )
            self._aggregator.on_1m_bar(bar)
            del self._partials[instrument]

    def _update_partial(self, instrument: str, mid_price: float, ts: datetime):
        minute = ts.minute
        self._flush_partial(instrument, minute)

        if instrument not in self._partials:
            self._partials[instrument] = {
                "minute": minute,
                "open": mid_price,
                "high": mid_price,
                "low": mid_price,
                "close": mid_price,
                "tick_count": 1,
                "timestamp": ts.replace(second=0, microsecond=0),
            }
        else:
            p = self._partials[instrument]
            p["high"] = max(p["high"], mid_price)
            p["low"] = min(p["low"], mid_price)
            p["close"] = mid_price
            p["tick_count"] += 1

    def start(self):
        """
        Blocking: streams forex prices.
        Run in a dedicated thread.
        """
        backoff = 2
        while True:
            try:
                self._running = True
                logger.info("Connecting to OANDA stream for: %s", self._instruments)
                api = oandapyV20.API(
                    access_token=self._access_token,
                    environment=self._environment,
                )
                params = {"instruments": ",".join(self._instruments)}
                r = PricingStream(accountID=self._account_id, params=params)

                for tick in api.request(r):
                    if not self._running:
                        break
                    if tick.get("type") != "PRICE":
                        continue

                    instrument = tick["instrument"]
                    bid = float(tick["bids"][0]["price"])
                    ask = float(tick["asks"][0]["price"])
                    mid = (bid + ask) / 2
                    ts = datetime.fromisoformat(
                        tick["time"].replace("Z", "+00:00")
                    )
                    self._update_partial(instrument, mid, ts)

                backoff = 2

            except Exception as exc:
                logger.warning(
                    "OANDA stream error: %s — reconnecting in %ds", exc, backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self):
        self._running = False
