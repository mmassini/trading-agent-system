"""
Post-Mortem Agent: analyzes closed trades and triggers model retraining.
"""
import logging
from typing import Callable, Optional

from agents.ml_analysis.model_registry import ModelRegistry
from storage.database import Database

logger = logging.getLogger(__name__)


class PostMortemAgent:
    def __init__(
        self,
        db: Database,
        settings,
        registry: Optional[ModelRegistry] = None,
        notify_callback: Optional[Callable[[str], None]] = None,
    ):
        self._db = db
        self._settings = settings
        self._registry = registry or ModelRegistry()
        self._notify = notify_callback
        self._last_retrain_count = 0

    def check_retrain_threshold(self):
        """
        Called after each trade closes.
        Triggers retraining if N new trades have closed since last retrain.
        """
        total_closed = self._db.count_closed_trades_total()
        new_trades = total_closed - self._last_retrain_count

        if new_trades >= self._settings.retrain_after_n_trades:
            logger.info(
                "Retrain threshold reached: %d new closed trades since last retrain",
                new_trades,
            )
            self._last_retrain_count = total_closed
            self._run_retrain()

    def _run_retrain(self):
        from agents.postmortem.retrain_pipeline import run_retrain
        for model_type in ["stock", "forex"]:
            try:
                run_retrain(
                    db=self._db,
                    registry=self._registry,
                    model_type=model_type,
                    notify_callback=self._notify,
                )
            except Exception as exc:
                logger.error("Retrain failed for %s: %s", model_type, exc)

    def get_recent_performance(self) -> dict:
        """Return a dict of performance metrics for the reporting agent."""
        from agents.postmortem.trade_analyzer import compute_metrics
        with self._db.session() as s:
            from storage.schema import Trade
            trades = s.query(Trade).filter(Trade.status == "closed").all()
            s.expunge_all()

        metrics = compute_metrics(trades)
        if metrics is None:
            return {}

        return {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "sharpe_ratio": metrics.sharpe_ratio,
            "total_pnl": metrics.total_pnl,
            "best_trade": metrics.best_trade,
            "worst_trade": metrics.worst_trade,
            "max_drawdown": metrics.max_drawdown,
        }
