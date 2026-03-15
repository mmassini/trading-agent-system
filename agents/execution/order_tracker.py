"""
Order tracker: polls broker APIs for order/trade status updates
and syncs the SQLite trades table accordingly.
"""
import logging
import time
import uuid
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

from storage.database import Database
from storage.schema import Trade

logger = logging.getLogger(__name__)


class OrderTracker:
    """
    Polls Alpaca and OANDA at a configurable interval for trade updates.
    On fill: updates trade entry_price and status.
    On close (SL/TP hit or manual): calculates P&L and R-multiple.
    """

    def __init__(
        self,
        db: Database,
        alpaca_executor=None,
        oanda_executor=None,
        poll_interval: float = 10.0,
    ):
        self._db = db
        self._alpaca = alpaca_executor
        self._oanda = oanda_executor
        self._interval = poll_interval
        self._running = False

    def start(self):
        """Start polling in a background thread."""
        self._running = True
        Thread(target=self._poll_loop, daemon=True, name="OrderTracker").start()
        logger.info("OrderTracker started (interval=%.0fs)", self._interval)

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._sync_alpaca_positions()
                self._sync_oanda_trades()
                self._save_balance_snapshots()
            except Exception as exc:
                logger.error("OrderTracker poll error: %s", exc)
            time.sleep(self._interval)

    def _sync_alpaca_positions(self):
        if self._alpaca is None:
            return

        open_trades = self._db.get_open_trades()
        alpaca_positions = {p["symbol"]: p for p in self._alpaca.get_open_positions()}

        for trade in open_trades:
            if trade.broker != "alpaca":
                continue

            # If position no longer open on Alpaca → trade was closed (SL or TP hit)
            if trade.symbol not in alpaca_positions:
                self._mark_trade_closed_alpaca(trade)

    def _mark_trade_closed_alpaca(self, trade: Trade):
        """
        Called when Alpaca no longer has an open position for a tracked trade.
        Fetches the closed order to get fill price and compute P&L.
        """
        try:
            # Get recent closed orders for this symbol
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[trade.symbol],
                limit=5,
            )
            orders = self._alpaca._client.get_orders(req)

            exit_price = None
            for order in orders:
                # Find the closing order after our entry
                if order.filled_at and order.filled_at > trade.entry_at:
                    exit_price = float(order.filled_avg_price or 0)
                    break

            if exit_price is None or exit_price <= 0:
                logger.warning("Could not determine exit price for trade %s", trade.id)
                return

            self._record_closed_trade(trade, exit_price)

        except Exception as exc:
            logger.error("Error closing Alpaca trade %s: %s", trade.id, exc)

    def _sync_oanda_trades(self):
        if self._oanda is None:
            return

        open_db_trades = {
            t.broker_order_id: t
            for t in self._db.get_open_trades()
            if t.broker == "oanda"
        }

        oanda_open = {t["trade_id"]: t for t in self._oanda.get_open_trades()}

        for order_id, trade in open_db_trades.items():
            if order_id not in oanda_open:
                # Trade closed on OANDA (SL or TP hit)
                self._mark_trade_closed_oanda(trade)

    def _mark_trade_closed_oanda(self, trade: Trade):
        try:
            from oandapyV20.endpoints.trades import TradeDetails
            r = TradeDetails(
                accountID=self._oanda._account_id,
                tradeID=trade.broker_order_id,
            )
            self._oanda._api.request(r)
            t_data = r.response.get("trade", {})
            exit_price = float(t_data.get("averageClosePrice", 0))
            if exit_price > 0:
                self._record_closed_trade(trade, exit_price)
        except Exception as exc:
            logger.error("Error closing OANDA trade %s: %s", trade.id, exc)

    def _record_closed_trade(self, trade: Trade, exit_price: float):
        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity

        risk_per_share = abs(trade.entry_price - trade.stop_loss)
        risk_amount = risk_per_share * trade.quantity
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0

        self._db.close_trade(
            trade_id=trade.id,
            exit_price=exit_price,
            exit_at=datetime.now(timezone.utc),
            pnl=round(pnl, 4),
            r_multiple=round(r_multiple, 4),
        )
        logger.info(
            "Trade closed: %s %s | exit=%.4f | P&L=$%.2f | R=%.2f",
            trade.direction, trade.symbol, exit_price, pnl, r_multiple,
        )

    def _save_balance_snapshots(self):
        if self._alpaca:
            try:
                balance = self._alpaca.get_account_balance()
                self._db.save_balance_snapshot(balance, balance, "alpaca")
            except Exception:
                pass

        if self._oanda:
            try:
                balance = self._oanda.get_account_balance()
                self._db.save_balance_snapshot(balance, balance, "oanda")
            except Exception:
                pass

        # Combined balance (sum of both accounts)
        try:
            alpaca_bal = self._alpaca.get_account_balance() if self._alpaca else 0.0
            oanda_bal = self._oanda.get_account_balance() if self._oanda else 0.0
            combined = alpaca_bal + oanda_bal
            if combined > 0:
                self._db.save_balance_snapshot(combined, combined, "combined")
        except Exception:
            pass
