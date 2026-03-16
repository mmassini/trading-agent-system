"""
Model validator: champion challenge logic.
The challenger only replaces the champion if it beats it by MIN_IMPROVEMENT.
"""
import json
import logging
from pathlib import Path

from agents.ml_analysis.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MIN_IMPROVEMENT = 0.05  # 5% Sharpe improvement required


def challenge_champion(
    registry: ModelRegistry,
    challenger_model,
    model_type: str,
    challenger_metrics: dict,
) -> bool:
    """
    Compare challenger vs champion. Promote if challenger is significantly better.

    Returns:
        True if challenger was promoted, False if champion retained.
    """
    champion_metrics_path = Path("models/champion/metrics.json")
    champion_metrics = {}
    if champion_metrics_path.exists():
        with open(champion_metrics_path) as f:
            data = json.load(f)
        champion_metrics = data.get(f"{model_type}_metrics", {})

    champion_sharpe = champion_metrics.get("sharpe_oos", 0.0)
    challenger_sharpe = challenger_metrics.get("sharpe_oos", 0.0)
    champion_drawdown = champion_metrics.get("max_drawdown", 1.0)
    challenger_drawdown = challenger_metrics.get("max_drawdown", 1.0)

    logger.info(
        "Champion challenge [%s]: champion_sharpe=%.4f | challenger_sharpe=%.4f",
        model_type, champion_sharpe, challenger_sharpe,
    )

    # If no model file exists on disk, always promote (bootstrap case)
    from agents.ml_analysis.model_registry import CHAMPION_DIR, STOCK_MODEL_FILE, FOREX_MODEL_FILE
    model_file = STOCK_MODEL_FILE if model_type == "stock" else FOREX_MODEL_FILE
    model_exists = (Path(CHAMPION_DIR) / model_file).exists()
    if not model_exists:
        logger.info("No existing champion model file — auto-promoting challenger.")
        registry.save_champion(challenger_model, model_type, challenger_metrics)
        return True

    # Challenger must beat champion by MIN_IMPROVEMENT on Sharpe
    sharpe_threshold = champion_sharpe * (1 + MIN_IMPROVEMENT)
    beats_sharpe = challenger_sharpe >= sharpe_threshold

    # Challenger must not significantly worsen drawdown (allow 10% worse)
    drawdown_threshold = champion_drawdown * 1.10
    acceptable_drawdown = challenger_drawdown <= drawdown_threshold

    if beats_sharpe and acceptable_drawdown:
        logger.info(
            "CHALLENGER WINS [%s]: Sharpe %.4f > %.4f threshold (drawdown: %.4f)",
            model_type, challenger_sharpe, sharpe_threshold, challenger_drawdown,
        )
        registry.save_champion(challenger_model, model_type, challenger_metrics)
        return True
    else:
        reasons = []
        if not beats_sharpe:
            reasons.append(f"Sharpe {challenger_sharpe:.4f} < threshold {sharpe_threshold:.4f}")
        if not acceptable_drawdown:
            reasons.append(f"Drawdown {challenger_drawdown:.4f} > threshold {drawdown_threshold:.4f}")
        logger.info(
            "CHAMPION RETAINED [%s]: %s",
            model_type, "; ".join(reasons),
        )
        return False
