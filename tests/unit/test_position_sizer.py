"""Unit tests for position sizing."""
import pytest
from agents.risk.position_sizer import calculate_position_size


def test_basic_long_position():
    spec = calculate_position_size(
        symbol="AAPL",
        direction="LONG",
        entry_price=150.0,
        atr=1.0,
        account_balance=10_000.0,
    )
    assert spec.direction == "LONG"
    assert spec.quantity >= 1
    # Risk amount should be ~2% of 10k = $200
    assert abs(spec.risk_amount - 200.0) < 1.0
    # Stop: entry - 1.5*ATR = 150 - 1.5 = 148.5
    assert abs(spec.stop_loss - 148.5) < 0.01
    # TP: entry + 2*1.5*ATR = 150 + 3.0 = 153.0
    assert abs(spec.take_profit - 153.0) < 0.01
    assert spec.broker == "alpaca"


def test_basic_short_position():
    spec = calculate_position_size(
        symbol="MSFT",
        direction="SHORT",
        entry_price=300.0,
        atr=2.0,
        account_balance=10_000.0,
    )
    assert spec.direction == "SHORT"
    # Stop above entry for SHORT
    assert spec.stop_loss > spec.entry_price
    # TP below entry for SHORT
    assert spec.take_profit < spec.entry_price


def test_forex_position():
    spec = calculate_position_size(
        symbol="EUR_USD",
        direction="LONG",
        entry_price=1.0900,
        atr=0.0010,
        account_balance=10_000.0,
    )
    assert spec.broker == "oanda"
    assert spec.quantity >= 1


def test_quantity_at_least_one():
    # Even with tiny balance relative to price, quantity >= 1 or raises ValueError
    try:
        spec = calculate_position_size(
            symbol="AMZN",
            direction="LONG",
            entry_price=3500.0,
            atr=50.0,
            account_balance=10_000.0,
        )
        assert spec.quantity >= 1
    except ValueError:
        pass  # Acceptable if quantity would be 0


def test_reward_risk_ratio():
    spec = calculate_position_size(
        symbol="SPY",
        direction="LONG",
        entry_price=500.0,
        atr=2.0,
        account_balance=10_000.0,
        reward_risk_ratio=2.0,
    )
    stop_dist = abs(spec.entry_price - spec.stop_loss)
    tp_dist = abs(spec.take_profit - spec.entry_price)
    assert abs(tp_dist / stop_dist - 2.0) < 0.01


def test_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_position_size("AAPL", "LONG", 0.0, 1.0, 10_000.0)  # entry=0
    with pytest.raises(ValueError):
        calculate_position_size("AAPL", "LONG", 150.0, 0.0, 10_000.0)  # atr=0
    with pytest.raises(ValueError):
        calculate_position_size("AAPL", "LONG", 150.0, 1.0, 0.0)    # balance=0
