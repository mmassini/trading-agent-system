"""Unit tests for drawdown monitor."""
import pytest
from unittest.mock import MagicMock
from agents.risk.drawdown_monitor import DrawdownMonitor


def _make_db(start_balance: float, current_balance: float):
    db = MagicMock()
    db.get_balance_at_session_open.return_value = start_balance
    db.get_latest_balance.return_value = current_balance
    return db


def test_no_halt_when_profitable():
    db = _make_db(10_000.0, 10_200.0)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    assert monitor.is_halt_triggered() is False


def test_no_halt_below_threshold():
    # 5% drawdown, threshold is 6%
    db = _make_db(10_000.0, 9_500.0)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    assert monitor.is_halt_triggered() is False


def test_halt_at_threshold():
    # Exactly 6% drawdown
    db = _make_db(10_000.0, 9_400.0)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    assert monitor.is_halt_triggered() is True


def test_halt_above_threshold():
    # 10% drawdown, well above 6% limit
    db = _make_db(10_000.0, 9_000.0)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    assert monitor.is_halt_triggered() is True


def test_no_halt_when_no_data():
    db = _make_db(None, None)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    # No data → allow trading
    assert monitor.is_halt_triggered() is False


def test_current_drawdown_pct():
    db = _make_db(10_000.0, 9_500.0)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    pct = monitor.current_drawdown_pct()
    assert abs(pct - 0.05) < 0.001


def test_remaining_budget():
    db = _make_db(10_000.0, 9_700.0)
    monitor = DrawdownMonitor(db, max_drawdown=0.06)
    budget = monitor.remaining_risk_budget()
    # Max loss = 600, current loss = 300, remaining = 300
    assert abs(budget - 300.0) < 1.0
