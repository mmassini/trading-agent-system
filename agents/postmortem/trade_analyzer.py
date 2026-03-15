"""
Trade analyzer: computes performance metrics for closed trades.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from storage.database import Database
from storage.schema import Trade

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_r_multiple: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    computed_at: datetime


def compute_metrics(trades: list[Trade]) -> Optional[TradeMetrics]:
    """
    Compute performance metrics from a list of closed trades.
    Returns None if not enough data.
    """
    closed = [t for t in trades if t.status == "closed" and t.pnl is not None]
    if len(closed) < 5:
        return None

    pnls = np.array([t.pnl for t in closed])
    r_multiples = np.array([t.r_multiple or 0.0 for t in closed])

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    avg_r = float(r_multiples.mean())

    # Sharpe ratio (annualized from per-trade returns)
    if pnls.std() > 0:
        sharpe = float((pnls.mean() / pnls.std()) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Maximum drawdown from cumulative P&L curve
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / (peak + 1e-10)
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    return TradeMetrics(
        total_trades=len(closed),
        win_count=int(len(wins)),
        loss_count=int(len(losses)),
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 4),
        avg_loss=round(avg_loss, 4),
        avg_r_multiple=round(avg_r, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        total_pnl=round(float(pnls.sum()), 4),
        best_trade=round(float(pnls.max()), 4),
        worst_trade=round(float(pnls.min()), 4),
        computed_at=datetime.utcnow(),
    )
