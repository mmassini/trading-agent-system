"""
FinBERT sentiment scorer.
Uses ProsusAI/finbert (finance-domain BERT) to score text → [-1.0, +1.0].
Model is pre-loaded at startup to avoid cold-start latency.
"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT model (ProsusAI/finbert)...")
            _pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,  # Return all class scores
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded.")
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s — falling back to 0.0 scores", exc)
            _pipeline = None
    return _pipeline


def score_text(text: str) -> float:
    """
    Score a single piece of financial text.

    Returns:
        float in [-1.0, +1.0]:
            +1.0 = very positive
             0.0 = neutral
            -1.0 = very negative
    """
    if not text or not text.strip():
        return 0.0

    pipe = _get_pipeline()
    if pipe is None:
        return 0.0

    try:
        results = pipe(text[:512])[0]  # list of {label, score} dicts
        score_map = {r["label"].lower(): r["score"] for r in results}
        # positive → +1, negative → -1, neutral → 0
        return (
            score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
        )
    except Exception as exc:
        logger.warning("FinBERT scoring error: %s", exc)
        return 0.0


def score_texts(texts: list[str]) -> list[float]:
    """Batch score a list of texts. Returns list of scores in [-1, +1]."""
    return [score_text(t) for t in texts]
