"""
OANDA order execution: submits market orders with attached stop-loss and take-profit.
"""
import logging
from typing import Optional

import oandapyV20
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeCRCDO, OpenTrades
from oandapyV20.endpoints.accounts import AccountDetails

from agents.risk.position_sizer import OrderSpec

logger = logging.getLogger(__name__)


class OandaExecutor:
    def __init__(
        self,
        access_token: str,
        account_id: str,
        environment: str = "practice",
    ):
        self._api = oandapyV20.API(
            access_token=access_token, environment=environment
        )
        self._account_id = account_id

    def submit_order(self, spec: OrderSpec) -> Optional[str]:
        """
        Submit a market order with stop-loss and take-profit on OANDA.
        OANDA uses 'units' — positive = buy (LONG), negative = sell (SHORT).

        Returns:
            OANDA trade ID string, or None on failure.
        """
        units = spec.quantity if spec.direction == "LONG" else -spec.quantity

        order_body = {
            "order": {
                "type": "MARKET",
                "instrument": spec.symbol,
                "units": str(units),
                "stopLossOnFill": {
                    "price": f"{spec.stop_loss:.5f}",
                },
                "takeProfitOnFill": {
                    "price": f"{spec.take_profit:.5f}",
                },
                "timeInForce": "FOK",  # Fill or Kill for immediate execution
            }
        }

        try:
            r = OrderCreate(accountID=self._account_id, data=order_body)
            self._api.request(r)
            response = r.response

            # Extract trade ID from response
            trade_opened = response.get("orderFillTransaction", {}).get("tradeOpened")
            if trade_opened:
                trade_id = trade_opened.get("tradeID")
                logger.info(
                    "OANDA order filled: %s %s x%d | trade_id:%s",
                    spec.direction, spec.symbol, spec.quantity, trade_id,
                )
                return str(trade_id)
            else:
                logger.warning("OANDA order response had no tradeOpened: %s", response)
                return None

        except Exception as exc:
            logger.error("OANDA order submission failed for %s: %s", spec.symbol, exc)
            return None

    def close_trade(self, trade_id: str) -> bool:
        """Manually close an open trade (e.g., end-of-day cleanup)."""
        from oandapyV20.endpoints.trades import TradeClose
        try:
            r = TradeClose(accountID=self._account_id, tradeID=trade_id)
            self._api.request(r)
            logger.info("OANDA trade closed: %s", trade_id)
            return True
        except Exception as exc:
            logger.warning("Failed to close OANDA trade %s: %s", trade_id, exc)
            return False

    def get_account_balance(self) -> float:
        try:
            r = AccountDetails(accountID=self._account_id)
            self._api.request(r)
            return float(r.response["account"]["NAV"])
        except Exception as exc:
            logger.error("Failed to get OANDA balance: %s", exc)
            return 0.0

    def get_open_trades(self) -> list[dict]:
        try:
            r = OpenTrades(accountID=self._account_id)
            self._api.request(r)
            trades = r.response.get("trades", [])
            return [
                {
                    "trade_id": t["id"],
                    "symbol": t["instrument"],
                    "units": float(t["currentUnits"]),
                    "avg_entry": float(t["price"]),
                    "unrealized_pl": float(t["unrealizedPL"]),
                }
                for t in trades
            ]
        except Exception as exc:
            logger.error("Failed to get OANDA open trades: %s", exc)
            return []
