"""
Alpaca real-time stock data stream via WebSocket.
Feeds 1-minute bars into the BarAggregator.
"""
import asyncio
import logging
from datetime import timezone

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar

from agents.data_ingest.bar_aggregator import BarAggregator
from agents.data_ingest.data_buffer import BarData

logger = logging.getLogger(__name__)


class AlpacaStreamHandler:
    """
    Subscribes to 1-minute bars for a list of symbols via Alpaca WebSocket.
    Automatically reconnects with exponential backoff on disconnect.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbols: list[str],
        aggregator: BarAggregator,
        paper: bool = True,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._symbols = symbols
        self._aggregator = aggregator
        self._paper = paper
        self._stream: StockDataStream | None = None

    async def _bar_handler(self, bar: Bar):
        bar_data = BarData(
            symbol=bar.symbol,
            timeframe="1m",
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            timestamp=bar.timestamp.replace(tzinfo=timezone.utc)
            if bar.timestamp.tzinfo is None
            else bar.timestamp,
        )
        self._aggregator.on_1m_bar(bar_data)
        logger.debug("Alpaca bar: %s @ %.4f", bar.symbol, bar_data.close)

    def start(self):
        """
        Blocking: starts the WebSocket stream.
        Should be run in a dedicated thread via threading.Thread(target=...).
        """
        backoff = 2
        while True:
            try:
                logger.info("Connecting to Alpaca stream for: %s", self._symbols)
                self._stream = StockDataStream(
                    self._api_key,
                    self._secret_key,
                    feed=DataFeed.IEX,  # IEX = free tier; SIP = paid
                )
                self._stream.subscribe_bars(self._bar_handler, *self._symbols)
                self._stream.run()
                backoff = 2  # Reset on clean exit
            except Exception as exc:
                logger.warning(
                    "Alpaca stream disconnected: %s — reconnecting in %ds", exc, backoff
                )
                import time
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self):
        if self._stream:
            self._stream.stop()
