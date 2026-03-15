"""
XGBoost model wrapper for trade signal classification.
Handles training, inference, save, and load.
"""
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from agents.ml_analysis.feature_engineer import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class TradingModel:
    """
    Wraps XGBClassifier for 3-class signal prediction:
      0 = SHORT, 1 = FLAT, 2 = LONG
    """

    VERSION_PREFIX = "v"

    def __init__(self):
        self._model: Optional[XGBClassifier] = None
        self._version: str = "untrained"
        self._trained_at: Optional[datetime] = None

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        version: str | None = None,
    ):
        """
        Train the XGBoost model with early stopping on validation set.
        """
        assert list(X_train.columns) == FEATURE_COLUMNS, (
            "Feature columns mismatch — ensure feature_engineer.py output matches FEATURE_COLUMNS"
        )

        self._model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            num_class=3,
            objective="multi:softprob",
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self._trained_at = datetime.utcnow()
        self._version = version or f"{self.VERSION_PREFIX}{int(self._trained_at.timestamp())}"
        logger.info("Model trained: %s | best iteration: %d",
                    self._version, self._model.best_iteration)

    def predict(self, X: pd.DataFrame) -> tuple[str, float]:
        """
        Predict signal for the last row of X.

        Returns:
            (direction, confidence) where direction is "LONG", "SHORT", or "FLAT"
        """
        assert self._model is not None, "Model not trained"
        row = X[FEATURE_COLUMNS].iloc[[-1]]
        proba = self._model.predict_proba(row)[0]  # [short_p, flat_p, long_p]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])

        direction_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        return direction_map[class_idx], confidence

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with columns [direction, confidence] for each row."""
        assert self._model is not None, "Model not trained"
        X_feat = X[FEATURE_COLUMNS].copy()
        proba = self._model.predict_proba(X_feat)
        class_idx = np.argmax(proba, axis=1)
        direction_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        return pd.DataFrame({
            "direction": [direction_map[i] for i in class_idx],
            "confidence": proba[np.arange(len(class_idx)), class_idx],
            "short_proba": proba[:, 0],
            "flat_proba": proba[:, 1],
            "long_proba": proba[:, 2],
        }, index=X.index)

    def save(self, path: str):
        """Save model to disk (XGBoost JSON format)."""
        assert self._model is not None, "No model to save"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._model.save_model(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str, version: str = "loaded"):
        """Load model from disk."""
        self._model = XGBClassifier()
        self._model.load_model(path)
        self._version = version
        logger.info("Model loaded from %s (version: %s)", path, version)

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self._model.feature_importances_,
            index=FEATURE_COLUMNS,
        ).sort_values(ascending=False)
