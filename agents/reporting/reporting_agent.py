"""
Reporting Agent: sends the 12-hour performance report to Telegram.
Can be invoked directly (CLI) or called by the orchestrator.

Usage:
    python -m agents.reporting.reporting_agent --mode=send
"""
import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)


class ReportingAgent:
    def __init__(self, db, telegram_client):
        self._db = db
        self._telegram = telegram_client

    async def send_report(self):
        """Generate and send the full 12-hour report."""
        from agents.reporting.report_builder import build_report
        from agents.reporting.chart_generator import generate_equity_curve
        from storage.schema import BalanceSnapshot

        # Build text report
        text = build_report(self._db)

        # Build equity curve chart
        with self._db.session() as s:
            snaps = (
                s.query(BalanceSnapshot)
                .filter(BalanceSnapshot.source == "combined")
                .order_by(BalanceSnapshot.snapshot_at.asc())
                .all()
            )
            history = [(snap.snapshot_at, snap.balance) for snap in snaps]
            s.expunge_all()

        chart_path = generate_equity_curve(history)

        # Send to Telegram
        if chart_path:
            await self._telegram.send_photo(chart_path, caption="Equity Curve")

        await self._telegram.send_message(text)
        logger.info("Report sent successfully")

    async def send_alert(self, message: str):
        await self._telegram.send_alert(message)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["send"], default="send")
    args = parser.parse_args()

    # Load settings
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from config.settings import settings
    from storage.database import Database
    from agents.reporting.telegram_client import TelegramClient

    db = Database(settings.db_path)
    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    agent = ReportingAgent(db, telegram)

    asyncio.run(agent.send_report())


if __name__ == "__main__":
    main()
