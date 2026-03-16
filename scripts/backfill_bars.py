"""
Backfill historical 1-minute bars from Alpaca and OANDA.
Run before the first model training to seed the database.

Usage:
    python scripts/backfill_bars.py [--days=90]
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_stocks(symbols: list[str], days: int, db):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from storage.schema import Bar

    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET_KEY"]
    client = StockHistoricalDataClient(api_key, secret)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    for symbol in symbols:
        logger.info("Backfilling %s...", symbol)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            bar_list = bars[symbol] if symbol in bars else []

            with db.session() as s:
                for bar in bar_list:
                    s.merge(Bar(
                        symbol=symbol,
                        timeframe="1m",
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                        timestamp=bar.timestamp,
                        source="backfill",
                    ))
            logger.info("  %s: %d bars saved", symbol, len(bar_list))
        except Exception as exc:
            logger.error("  Failed for %s: %s", symbol, exc)


def backfill_forex(instruments: list[str], days: int, db):
    import oandapyV20
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    from storage.schema import Bar

    token = os.environ["OANDA_ACCESS_TOKEN"]
    account_id = os.environ["OANDA_ACCOUNT_ID"]
    env = os.environ.get("OANDA_ENVIRONMENT", "practice")
    api = oandapyV20.API(access_token=token, environment=env)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    for instrument in instruments:
        logger.info("Backfilling %s...", instrument)
        try:
            params = {
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "granularity": "M1",
                "price": "M",  # Mid prices
            }
            r = InstrumentsCandles(instrument=instrument, params=params)
            api.request(r)
            candles = r.response.get("candles", [])

            with db.session() as s:
                for c in candles:
                    if not c.get("complete", True):
                        continue
                    ts = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
                    mid = c["mid"]
                    s.merge(Bar(
                        symbol=instrument,
                        timeframe="1m",
                        open=float(mid["o"]),
                        high=float(mid["h"]),
                        low=float(mid["l"]),
                        close=float(mid["c"]),
                        volume=float(c.get("volume", 0)),
                        timestamp=ts,
                        source="backfill",
                    ))
            logger.info("  %s: %d candles saved", instrument, len(candles))
        except Exception as exc:
            logger.error("  Failed for %s: %s", instrument, exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    from storage.database import Database
    import yaml

    db = Database(os.environ.get("DB_PATH", "data/trading.db"))

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "instruments.yaml"
    )
    with open(config_path) as f:
        instruments = yaml.safe_load(f)

    backfill_stocks(instruments["stocks"], args.days, db)
    if instruments.get("forex"):
        backfill_forex(instruments["forex"], args.days, db)
    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
