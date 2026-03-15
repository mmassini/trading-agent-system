"""
Alpaca order execution: submits bracket orders (entry + SL + TP) for US stocks.
"""
import logging
import uuid
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from agents.risk.position_sizer import OrderSpec

logger = logging.getLogger(__name__)


class AlpacaExecutor:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._client = TradingClient(api_key, secret_key, paper=paper)

    def submit_bracket_order(self, spec: OrderSpec) -> Optional[str]:
        """
        Submit a bracket order (market entry with stop-loss and take-profit).

        Returns:
            Alpaca order ID string, or None on failure.
        """
        side = OrderSide.BUY if spec.direction == "LONG" else OrderSide.SELL

        # Alpaca requires TIF=DAY for bracket orders
        req = MarketOrderRequest(
            symbol=spec.symbol,
            qty=spec.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            take_profit=TakeProfitRequest(limit_price=spec.take_profit),
            stop_loss=StopLossRequest(stop_price=spec.stop_loss),
        )

        try:
            order = self._client.submit_order(req)
            logger.info(
                "Alpaca bracket order submitted: %s %s x%d | id:%s",
                spec.direction, spec.symbol, spec.quantity, order.id,
            )
            return str(order.id)
        except Exception as exc:
            logger.error("Alpaca order submission failed for %s: %s", spec.symbol, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Alpaca order cancelled: %s", order_id)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel Alpaca order %s: %s", order_id, exc)
            return False

    def get_account_balance(self) -> float:
        """Return current cash + equity balance."""
        try:
            account = self._client.get_account()
            return float(account.equity)
        except Exception as exc:
            logger.error("Failed to get Alpaca balance: %s", exc)
            return 0.0

    def get_open_positions(self) -> list[dict]:
        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": p.side.value,
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception as exc:
            logger.error("Failed to get Alpaca positions: %s", exc)
            return []
