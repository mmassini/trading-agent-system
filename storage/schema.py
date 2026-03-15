"""
SQLAlchemy ORM models for all persistent data.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(String, primary_key=True)          # UUID
    symbol = Column(String, nullable=False)
    broker = Column(String, nullable=False)         # "alpaca" | "oanda"
    direction = Column(String, nullable=False)      # "LONG" | "SHORT"
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    status = Column(String, default="open")         # "open" | "closed" | "cancelled"
    entry_at = Column(DateTime, nullable=False)
    exit_at = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)       # P&L / initial risk amount
    signal_confidence = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    broker_order_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Bar(Base):
    __tablename__ = "bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)      # "1m" | "5m" | "15m"
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String, default="live")         # "live" | "backfill"


class SentimentRecord(Base):
    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    score = Column(Float, nullable=False)           # -1.0 to +1.0
    source = Column(String, nullable=False)         # "newsapi" | "twitter" | "reddit"
    headline = Column(Text, nullable=True)
    raw_text_hash = Column(String, nullable=True)   # SHA256 for deduplication
    computed_at = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String, nullable=False)     # "stock" | "forex"
    version = Column(String, nullable=False)
    sharpe_oos = Column(Float, nullable=True)       # Out-of-sample Sharpe
    precision_long = Column(Float, nullable=True)
    precision_short = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    promoted = Column(Boolean, default=False)
    promotion_reason = Column(Text, nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)


class BalanceSnapshot(Base):
    __tablename__ = "balance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=True)           # Balance + unrealized P&L
    source = Column(String, nullable=False)         # "alpaca" | "oanda" | "combined"
    snapshot_at = Column(DateTime, default=datetime.utcnow)


class Heartbeat(Base):
    __tablename__ = "heartbeats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    component = Column(String, nullable=False)      # "orchestrator" | "data_ingest" | etc.
    status = Column(String, default="ok")
    message = Column(Text, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)
