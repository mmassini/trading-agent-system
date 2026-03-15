"""
Telegram Bot client for sending reports and alerts.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id
        self._bot = None

        if bot_token and chat_id:
            try:
                import telegram
                import asyncio
                self._bot = telegram.Bot(token=bot_token)
                logger.info("Telegram bot initialized")
            except ImportError:
                logger.warning("python-telegram-bot not installed")

    async def send_message(self, text: str) -> bool:
        if not self._bot:
            logger.warning("Telegram not configured — skipping message")
            return False
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
            )
            return True
        except Exception as exc:
            logger.error("Telegram send_message failed: %s", exc)
            return False

    async def send_photo(self, image_path: str, caption: str = "") -> bool:
        if not self._bot:
            return False
        try:
            with open(image_path, "rb") as f:
                await self._bot.send_photo(
                    chat_id=self._chat_id,
                    photo=f,
                    caption=caption,
                )
            return True
        except Exception as exc:
            logger.error("Telegram send_photo failed: %s", exc)
            return False

    async def send_alert(self, message: str) -> bool:
        """Send an urgent alert (same as message, different prefix for readability)."""
        return await self.send_message(f"⚠️ ALERT\n{message}")
