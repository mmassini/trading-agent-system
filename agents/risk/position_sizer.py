"""
Position sizing using fixed-fractional method with ATR-based stop distance.
All trade sizing flows through this module — correctness is critical.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderSpec:
    symbol: str
    direction: str      # "LONG" | "SHORT"
    entry_price: float
    quantity: float     # Shares (stocks) or units (forex)
    stop_loss: float
    take_profit: float
    risk_amount: float  # $ at risk on this trade
    stop_distance: float
    broker: str         # "alpaca" | "oanda"


def calculate_position_size(
    symbol: str,
    direction: str,
    entry_price: float,
    atr: float,
    account_balance: float,
    max_risk_pct: float = 0.02,
    atr_multiplier: float = 1.5,
    reward_risk_ratio: float = 2.0,
) -> OrderSpec:
    """
    Calculate order parameters using fixed-fractional position sizing.

    Formula:
        risk_amount   = account_balance × max_risk_pct
        stop_distance = atr × atr_multiplier
        quantity      = floor(risk_amount / stop_distance)
        stop_loss     = entry ∓ stop_distance
        take_profit   = entry ± (stop_distance × reward_risk_ratio)

    Args:
        symbol: Instrument symbol
        direction: "LONG" or "SHORT"
        entry_price: Expected fill price
        atr: ATR(14) at signal time
        account_balance: Current account balance in USD
        max_risk_pct: Max fraction of balance to risk (default 2%)
        atr_multiplier: Stop distance = ATR × this (default 1.5)
        reward_risk_ratio: TP = stop_distance × this (default 2.0)

    Returns:
        OrderSpec with all order parameters
    """
    if entry_price <= 0 or atr <= 0 or account_balance <= 0:
        raise ValueError(
            f"Invalid inputs: entry={entry_price}, atr={atr}, balance={account_balance}"
        )

    risk_amount = account_balance * max_risk_pct
    stop_distance = atr * atr_multiplier

    if stop_distance <= 0:
        raise ValueError(f"Stop distance is zero or negative: {stop_distance}")

    # Raw quantity (fractional), floor to whole shares/units
    raw_quantity = risk_amount / stop_distance
    quantity = int(raw_quantity)

    if quantity < 1:
        raise ValueError(
            f"Computed quantity={quantity} < 1. "
            f"Balance ${account_balance:.2f}, risk_amount=${risk_amount:.2f}, "
            f"stop_distance=${stop_distance:.4f}. Consider increasing balance or ATR."
        )

    if direction == "LONG":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (stop_distance * reward_risk_ratio)
    else:  # SHORT
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - (stop_distance * reward_risk_ratio)

    # Detect broker from symbol format
    broker = "oanda" if "_" in symbol else "alpaca"

    spec = OrderSpec(
        symbol=symbol,
        direction=direction,
        entry_price=round(entry_price, 6),
        quantity=quantity,
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_amount=round(risk_amount, 2),
        stop_distance=round(stop_distance, 6),
        broker=broker,
    )

    logger.info(
        "Position sized: %s %s x%d @ %.4f | SL:%.4f TP:%.4f | risk $%.2f",
        direction, symbol, quantity, entry_price, stop_loss, take_profit, risk_amount,
    )
    return spec
