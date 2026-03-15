"""
Risk Agent: the final gatekeeper before any order is submitted.
Combines position sizing, drawdown monitoring, and exposure checks.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from agents.ml_analysis.signal import TradeSignal
from agents.risk.drawdown_monitor import DrawdownMonitor
from agents.risk.exposure_tracker import ExposureTracker
from agents.risk.position_sizer import OrderSpec, calculate_position_size
from storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    order_spec: Optional[OrderSpec] = None


class RiskAgent:
    """
    Evaluates a TradeSignal and returns a RiskDecision.
    All checks must pass before an order is submitted.
    """

    def __init__(
        self,
        db: Database,
        max_risk_pct: float = 0.02,
        atr_multiplier: float = 1.5,
        reward_risk_ratio: float = 2.0,
        min_confidence: float = 0.65,
        max_daily_drawdown: float = 0.06,
        max_positions: int = 3,
    ):
        self._db = db
        self._max_risk_pct = max_risk_pct
        self._atr_multiplier = atr_multiplier
        self._reward_risk_ratio = reward_risk_ratio
        self._min_confidence = min_confidence
        self._drawdown = DrawdownMonitor(db, max_drawdown=max_daily_drawdown)
        self._exposure = ExposureTracker(db, max_positions=max_positions)

    def evaluate(
        self,
        signal: TradeSignal,
        current_price: float,
        account_balance: float,
    ) -> RiskDecision:
        """
        Evaluate a signal and return a RiskDecision.

        Args:
            signal: Output from ML Analysis Agent
            current_price: Latest market price for sizing
            account_balance: Combined portfolio balance

        Returns:
            RiskDecision with approved flag, reason, and OrderSpec if approved
        """
        # ── Check 1: signal must be actionable ─────────────────────────────
        if not signal.is_actionable(self._min_confidence):
            return RiskDecision(
                approved=False,
                reason=f"Signal not actionable: direction={signal.direction}, "
                       f"confidence={signal.confidence:.2f} < {self._min_confidence}",
            )

        # ── Check 2: daily drawdown circuit breaker ─────────────────────────
        if self._drawdown.is_halt_triggered():
            drawdown_pct = (self._drawdown.current_drawdown_pct() or 0) * 100
            return RiskDecision(
                approved=False,
                reason=f"Daily drawdown halt triggered: {drawdown_pct:.1f}% daily loss",
            )

        # ── Check 3: position limit ─────────────────────────────────────────
        if not self._exposure.can_open_new_position():
            return RiskDecision(
                approved=False,
                reason=f"Max concurrent positions reached ({self._exposure.open_count()})",
            )

        # ── Check 4: compute position size ─────────────────────────────────
        try:
            spec = calculate_position_size(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=current_price,
                atr=signal.atr,
                account_balance=account_balance,
                max_risk_pct=self._max_risk_pct,
                atr_multiplier=self._atr_multiplier,
                reward_risk_ratio=self._reward_risk_ratio,
            )
        except ValueError as exc:
            return RiskDecision(
                approved=False,
                reason=f"Position sizing error: {exc}",
            )

        # ── Check 5: sanity check on stop/TP prices ─────────────────────────
        if spec.stop_loss <= 0:
            return RiskDecision(
                approved=False,
                reason=f"Invalid stop_loss price: {spec.stop_loss}",
            )
        if spec.take_profit <= 0:
            return RiskDecision(
                approved=False,
                reason=f"Invalid take_profit price: {spec.take_profit}",
            )

        logger.info(
            "Risk APPROVED: %s %s x%d | risk $%.2f | SL:%.4f TP:%.4f",
            spec.direction, spec.symbol, spec.quantity,
            spec.risk_amount, spec.stop_loss, spec.take_profit,
        )

        return RiskDecision(approved=True, reason="All checks passed", order_spec=spec)
