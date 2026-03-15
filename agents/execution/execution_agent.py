"""
Execution Agent: routes approved OrderSpecs to the correct broker executor.
Records all trades to SQLite for post-mortem analysis.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from agents.execution.alpaca_executor import AlpacaExecutor
from agents.execution.oanda_executor import OandaExecutor
from agents.risk.position_sizer import OrderSpec
from storage.database import Database
from storage.schema import Trade

logger = logging.getLogger(__name__)


class ExecutionAgent:
    def __init__(
        self,
        db: Database,
        alpaca: AlpacaExecutor,
        oanda: OandaExecutor,
    ):
        self._db = db
        self._alpaca = alpaca
        self._oanda = oanda

    def execute(
        self,
        spec: OrderSpec,
        signal_confidence: float = 0.0,
        sentiment_score: float = 0.0,
        model_version: str = "unknown",
    ) -> Optional[str]:
        """
        Execute an approved OrderSpec.

        Returns:
            Trade UUID if successful, None on failure.
        """
        # Route to correct broker
        if spec.broker == "alpaca":
            broker_order_id = self._alpaca.submit_bracket_order(spec)
        elif spec.broker == "oanda":
            broker_order_id = self._oanda.submit_order(spec)
        else:
            logger.error("Unknown broker: %s", spec.broker)
            return None

        if broker_order_id is None:
            logger.error("Order submission failed for %s", spec.symbol)
            return None

        # Record to database
        trade_id = str(uuid.uuid4())
        trade = Trade(
            id=trade_id,
            symbol=spec.symbol,
            broker=spec.broker,
            direction=spec.direction,
            entry_price=spec.entry_price,
            quantity=spec.quantity,
            stop_loss=spec.stop_loss,
            take_profit=spec.take_profit,
            status="open",
            entry_at=datetime.now(timezone.utc),
            signal_confidence=signal_confidence,
            sentiment_score=sentiment_score,
            model_version=model_version,
            broker_order_id=broker_order_id,
        )
        self._db.save_trade(trade)

        logger.info(
            "Trade opened: id=%s %s %s x%d @ %.4f",
            trade_id, spec.direction, spec.symbol, spec.quantity, spec.entry_price,
        )
        return trade_id
