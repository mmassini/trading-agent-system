"""
TradeSignal: output contract of the ML Analysis Agent.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeSignal:
    symbol: str
    direction: str          # "LONG" | "SHORT" | "FLAT"
    confidence: float       # XGBoost predict_proba for the predicted class
    atr: float              # ATR(14) at signal time — used for stop distance
    predicted_return: float # Expected % move at horizon
    timestamp: datetime
    model_version: str = "unknown"

    def is_actionable(self, min_confidence: float = 0.65) -> bool:
        return self.direction != "FLAT" and self.confidence >= min_confidence
