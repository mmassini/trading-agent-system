"""
Retraining pipeline: walk-forward XGBoost training + champion challenge.
Can be triggered automatically (after N trades) or by the weekly cron job.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from agents.ml_analysis.feature_engineer import build_features, build_labels, FEATURE_COLUMNS
from agents.ml_analysis.model_registry import ModelRegistry
from agents.ml_analysis.xgboost_model import TradingModel
from agents.postmortem.model_validator import challenge_champion
from storage.database import Database

logger = logging.getLogger(__name__)

TRAIN_WINDOW_DAYS = 60
VALIDATION_WINDOW_DAYS = 10
HOLDOUT_DAYS = 20


def run_retrain(
    db: Database,
    registry: ModelRegistry,
    model_type: str = "stock",
    notify_callback=None,
):
    """
    Full retraining pipeline for stock or forex models.

    Args:
        db: Database instance
        registry: ModelRegistry for loading/saving champion
        model_type: "stock" or "forex"
        notify_callback: Optional callable(message: str) for Telegram notifications
    """
    logger.info("Starting retrain pipeline for: %s", model_type)

    # ── 1. Load historical bars from SQLite ──────────────────────────────────
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=TRAIN_WINDOW_DAYS + VALIDATION_WINDOW_DAYS + HOLDOUT_DAYS + 5)

    # Filter by instrument type
    if model_type == "stock":
        symbol_filter = "symbol NOT LIKE '%\\_%' ESCAPE '\\'"
    else:
        symbol_filter = "symbol LIKE '%\\_%' ESCAPE '\\'"

    with db.session() as s:
        result = s.execute(text(
            f"SELECT symbol, open, high, low, close, volume, timestamp "
            f"FROM bars "
            f"WHERE {symbol_filter} AND timeframe='1m' "
            f"AND timestamp >= :start ORDER BY symbol, timestamp"
        ), {"start": start_date.isoformat()})
        rows = result.fetchall()

    if len(rows) < 500:
        raise RuntimeError(f"Not enough data for retraining: {len(rows)} rows (need 500+). Run backfill first.")

    # ── 2. Group by symbol and build features ────────────────────────────────
    df_all = pd.DataFrame(rows, columns=["symbol", "open", "high", "low", "close", "volume", "timestamp"])
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)

    all_features = []
    all_labels = []

    for symbol, group in df_all.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 100:
            continue

        try:
            feats = build_features(group, sentiment_score=0.0)
            labels = build_labels(group, horizon=5, threshold=0.001)

            # Align: drop NaN rows
            valid_mask = feats.notna().all(axis=1) & labels.notna()
            feats = feats[valid_mask]
            labels = labels[valid_mask]

            if len(feats) > 50:
                all_features.append(feats)
                all_labels.append(labels)
        except Exception as exc:
            logger.warning("Feature engineering failed for %s: %s", symbol, exc)

    if not all_features:
        logger.warning("No valid feature data for retraining")
        return

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    logger.info("Combined dataset: %d samples, %d features", len(X), len(X.columns))

    # ── 3. Time-based split ───────────────────────────────────────────────────
    # Use index-based split (assumes chronological order after concat)
    n = len(X)
    holdout_start = int(n * (1 - HOLDOUT_DAYS / (TRAIN_WINDOW_DAYS + VALIDATION_WINDOW_DAYS + HOLDOUT_DAYS)))
    val_start = int(holdout_start * (TRAIN_WINDOW_DAYS / (TRAIN_WINDOW_DAYS + VALIDATION_WINDOW_DAYS)))

    X_train, y_train = X.iloc[:val_start], y.iloc[:val_start]
    X_val, y_val = X.iloc[val_start:holdout_start], y.iloc[val_start:holdout_start]
    X_holdout, y_holdout = X.iloc[holdout_start:], y.iloc[holdout_start:]

    logger.info("Split: train=%d, val=%d, holdout=%d", len(X_train), len(X_val), len(X_holdout))

    # ── 4. Train challenger ───────────────────────────────────────────────────
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}"
    challenger = TradingModel()

    try:
        challenger.train(X_train, y_train, X_val, y_val, version=version)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return

    # ── 5. Evaluate on holdout ────────────────────────────────────────────────
    predictions = challenger.predict_batch(X_holdout)
    holdout_pnl = _simulate_pnl(predictions, y_holdout)

    if holdout_pnl.std() > 0:
        sharpe = float((holdout_pnl.mean() / holdout_pnl.std()) * np.sqrt(252))
    else:
        sharpe = 0.0

    cumulative = np.cumsum(holdout_pnl.values)
    peak = np.maximum.accumulate(cumulative)
    max_dd = float(((peak - cumulative) / (peak + 1e-10)).max())

    challenger_metrics = {
        "sharpe_oos": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "holdout_samples": len(X_holdout),
        "version": version,
    }
    logger.info("Challenger metrics [%s]: %s", model_type, challenger_metrics)

    # ── 6. Champion challenge ─────────────────────────────────────────────────
    promoted = challenge_champion(registry, challenger, model_type, challenger_metrics)

    # ── 7. Log to DB ──────────────────────────────────────────────────────────
    from storage.schema import ModelRun
    with db.session() as s:
        s.add(ModelRun(
            model_type=model_type,
            version=version,
            sharpe_oos=sharpe,
            max_drawdown=max_dd,
            promoted=promoted,
            promotion_reason="Champion challenge" if promoted else "Did not beat champion",
        ))

    # ── 8. Notify ─────────────────────────────────────────────────────────────
    if notify_callback:
        status = "PROMOTED" if promoted else "retained (champion kept)"
        msg = (
            f"Model retrain complete [{model_type}]:\n"
            f"  Version: {version}\n"
            f"  Sharpe OOS: {sharpe:.4f}\n"
            f"  Max Drawdown: {max_dd:.2%}\n"
            f"  Result: {status}"
        )
        notify_callback(msg)


def _simulate_pnl(predictions: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Simulate simple P&L: +1 for correct directional prediction, -1 for wrong.
    Ignores FLAT predictions (they generate 0 P&L).
    """
    pnl = pd.Series(0.0, index=predictions.index)
    for i in predictions.index:
        pred_dir = predictions.loc[i, "direction"]
        true_label = labels.loc[i]  # 0=SHORT, 1=FLAT, 2=LONG

        if pred_dir == "FLAT" or true_label == 1:
            continue

        pred_label = 2 if pred_dir == "LONG" else 0
        pnl.loc[i] = 1.0 if pred_label == true_label else -1.0

    return pnl
