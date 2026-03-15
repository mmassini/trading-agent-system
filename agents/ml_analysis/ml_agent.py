"""
ML Analysis Agent: builds features from live bars and produces a TradeSignal.
Called by the Orchestrator as a subagent.
"""
import logging
from datetime import datetime, timezone

import pandas as pd

from agents.data_ingest.data_buffer import DataBuffer
from agents.ml_analysis.feature_engineer import build_features
from agents.ml_analysis.model_registry import ModelRegistry
from agents.ml_analysis.signal import TradeSignal

logger = logging.getLogger(__name__)

MIN_BARS = 50  # Minimum bars needed to compute all features


def analyze_symbol(
    symbol: str,
    buffer: DataBuffer,
    registry: ModelRegistry,
    sentiment_score: float = 0.0,
) -> TradeSignal:
    """
    Produce a TradeSignal for a given symbol using live bar data.

    Args:
        symbol: Instrument to analyze (e.g., "AAPL" or "EUR_USD")
        buffer: DataBuffer containing live bars
        registry: ModelRegistry with champion models
        sentiment_score: Latest sentiment score from SentimentAgent

    Returns:
        TradeSignal with direction, confidence, and ATR
    """
    flat_signal = TradeSignal(
        symbol=symbol,
        direction="FLAT",
        confidence=0.0,
        atr=0.0,
        predicted_return=0.0,
        timestamp=datetime.now(timezone.utc),
    )

    # Get the model for this symbol
    model = registry.get_model(symbol)
    if model is None or not model.is_trained:
        logger.warning("No trained model available for %s", symbol)
        return flat_signal

    # Get bars from buffer
    bars = buffer.get_latest(symbol, "1m", n=200)
    if len(bars) < MIN_BARS:
        logger.debug("%s: not enough bars (%d/%d)", symbol, len(bars), MIN_BARS)
        return flat_signal

    # Convert to DataFrame
    df = pd.DataFrame([{
        "timestamp": b.timestamp,
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
    } for b in bars])

    # Build features
    try:
        features = build_features(df, sentiment_score=sentiment_score)
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", symbol, exc)
        return flat_signal

    # Drop rows with NaN (need at least one clean row)
    features_clean = features.dropna()
    if features_clean.empty:
        logger.warning("%s: all feature rows have NaN after engineering", symbol)
        return flat_signal

    # Get ATR from last valid row (used for position sizing)
    atr = float(features_clean["atr_14"].iloc[-1])
    if atr <= 0:
        logger.warning("%s: invalid ATR=%.6f", symbol, atr)
        return flat_signal

    # Predict
    direction, confidence = model.predict(features_clean)
    predicted_return = 0.0  # Placeholder — extend for regression model

    return TradeSignal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        atr=atr,
        predicted_return=predicted_return,
        timestamp=datetime.now(timezone.utc),
        model_version=model.version,
    )
