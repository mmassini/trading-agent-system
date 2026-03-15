"""
SQLAlchemy session management and helper queries.
All database access goes through this module.
"""
import os
from contextlib import contextmanager
from datetime import datetime, date
from typing import Optional

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from storage.schema import Base, Trade, BalanceSnapshot, Heartbeat


def get_engine(db_path: str = "data/trading.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )


def create_all_tables(db_path: str = "data/trading.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


class Database:
    def __init__(self, db_path: str = "data/trading.db"):
        self.engine = get_engine(db_path)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    @contextmanager
    def session(self) -> Session:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── Trade helpers ─────────────────────────────────────────────────────────

    def count_open_trades(self) -> int:
        with self.session() as s:
            return s.query(Trade).filter(Trade.status == "open").count()

    def get_open_trades(self) -> list[Trade]:
        with self.session() as s:
            trades = s.query(Trade).filter(Trade.status == "open").all()
            s.expunge_all()
            return trades

    def get_trade_by_order_id(self, broker_order_id: str) -> Optional[Trade]:
        with self.session() as s:
            t = s.query(Trade).filter(
                Trade.broker_order_id == broker_order_id
            ).first()
            if t:
                s.expunge(t)
            return t

    def save_trade(self, trade: Trade):
        with self.session() as s:
            s.merge(trade)

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_at: datetime,
        pnl: float,
        r_multiple: float,
    ):
        with self.session() as s:
            t = s.query(Trade).filter(Trade.id == trade_id).first()
            if t:
                t.exit_price = exit_price
                t.exit_at = exit_at
                t.pnl = pnl
                t.r_multiple = r_multiple
                t.status = "closed"

    def count_closed_trades_today(self) -> int:
        today = date.today()
        with self.session() as s:
            return s.query(Trade).filter(
                Trade.status == "closed",
                func.date(Trade.exit_at) == today,
            ).count()

    def count_closed_trades_total(self) -> int:
        with self.session() as s:
            return s.query(Trade).filter(Trade.status == "closed").count()

    # ── Balance helpers ───────────────────────────────────────────────────────

    def save_balance_snapshot(self, balance: float, equity: float, source: str):
        with self.session() as s:
            s.add(BalanceSnapshot(balance=balance, equity=equity, source=source))

    def get_balance_at_session_open(self) -> Optional[float]:
        """Return the first balance snapshot of today."""
        today = date.today()
        with self.session() as s:
            snap = (
                s.query(BalanceSnapshot)
                .filter(func.date(BalanceSnapshot.snapshot_at) == today)
                .order_by(BalanceSnapshot.snapshot_at.asc())
                .first()
            )
            return snap.balance if snap else None

    def get_latest_balance(self, source: str = "combined") -> Optional[float]:
        with self.session() as s:
            snap = (
                s.query(BalanceSnapshot)
                .filter(BalanceSnapshot.source == source)
                .order_by(BalanceSnapshot.snapshot_at.desc())
                .first()
            )
            return snap.balance if snap else None

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def record_heartbeat(self, component: str, status: str = "ok", message: str = ""):
        with self.session() as s:
            s.add(Heartbeat(component=component, status=status, message=message))

    def get_last_heartbeat(self, component: str) -> Optional[Heartbeat]:
        with self.session() as s:
            h = (
                s.query(Heartbeat)
                .filter(Heartbeat.component == component)
                .order_by(Heartbeat.recorded_at.desc())
                .first()
            )
            if h:
                s.expunge(h)
            return h
