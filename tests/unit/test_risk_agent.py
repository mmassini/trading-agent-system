"""Unit tests for the Risk Agent."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from agents.ml_analysis.signal import TradeSignal
from agents.risk.risk_agent import RiskAgent


def _make_signal(direction="LONG", confidence=0.75, atr=1.5) -> TradeSignal:
    return TradeSignal(
        symbol="AAPL",
        direction=direction,
        confidence=confidence,
        atr=atr,
        predicted_return=0.002,
        timestamp=datetime.now(timezone.utc),
        model_version="v_test",
    )


def _make_db(open_count=0, start_balance=10_000.0, current_balance=10_000.0):
    db = MagicMock()
    db.count_open_trades.return_value = open_count
    db.get_balance_at_session_open.return_value = start_balance
    db.get_latest_balance.return_value = current_balance
    return db


def test_approved_trade():
    db = _make_db()
    agent = RiskAgent(db)
    signal = _make_signal()
    decision = agent.evaluate(signal, current_price=150.0, account_balance=10_000.0)
    assert decision.approved is True
    assert decision.order_spec is not None
    assert decision.order_spec.symbol == "AAPL"


def test_rejected_low_confidence():
    db = _make_db()
    agent = RiskAgent(db, min_confidence=0.70)
    signal = _make_signal(confidence=0.60)
    decision = agent.evaluate(signal, current_price=150.0, account_balance=10_000.0)
    assert decision.approved is False
    assert "confidence" in decision.reason.lower()


def test_rejected_flat_signal():
    db = _make_db()
    agent = RiskAgent(db)
    signal = _make_signal(direction="FLAT", confidence=0.80)
    decision = agent.evaluate(signal, current_price=150.0, account_balance=10_000.0)
    assert decision.approved is False


def test_rejected_drawdown_halt():
    # -8% drawdown (above 6% limit)
    db = _make_db(start_balance=10_000.0, current_balance=9_200.0)
    agent = RiskAgent(db, max_daily_drawdown=0.06)
    signal = _make_signal()
    decision = agent.evaluate(signal, current_price=150.0, account_balance=9_200.0)
    assert decision.approved is False
    assert "drawdown" in decision.reason.lower()


def test_rejected_max_positions():
    db = _make_db(open_count=3)  # Already at max
    agent = RiskAgent(db, max_positions=3)
    signal = _make_signal()
    decision = agent.evaluate(signal, current_price=150.0, account_balance=10_000.0)
    assert decision.approved is False
    assert "position" in decision.reason.lower()


def test_order_spec_stop_above_entry_for_short():
    db = _make_db()
    agent = RiskAgent(db)
    signal = _make_signal(direction="SHORT")
    decision = agent.evaluate(signal, current_price=150.0, account_balance=10_000.0)
    assert decision.approved is True
    spec = decision.order_spec
    assert spec.stop_loss > spec.entry_price  # Stop is above entry for SHORT
    assert spec.take_profit < spec.entry_price  # TP is below entry for SHORT
