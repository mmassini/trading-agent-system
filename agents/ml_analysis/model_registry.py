"""
Model registry: loads the champion model from disk with file-lock hot-reload.
The orchestrator calls check_and_reload() before each decision cycle.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from filelock import FileLock

from agents.ml_analysis.xgboost_model import TradingModel

logger = logging.getLogger(__name__)

CHAMPION_DIR = "models/champion"
STOCK_MODEL_FILE = "xgb_stock_model.json"
FOREX_MODEL_FILE = "xgb_forex_model.json"
METRICS_FILE = "metrics.json"
LOCK_TIMEOUT = 10  # seconds


class ModelRegistry:
    """
    Manages champion models for stocks and forex.
    Supports atomic hot-reload when retrain_pipeline promotes a new model.
    """

    def __init__(self, champion_dir: str = CHAMPION_DIR):
        self._dir = Path(champion_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._stock_model: Optional[TradingModel] = None
        self._forex_model: Optional[TradingModel] = None
        self._stock_mtime: float = 0.0
        self._forex_mtime: float = 0.0

        # Load on startup
        self._load_stock()
        self._load_forex()

    def _load_stock(self):
        path = self._dir / STOCK_MODEL_FILE
        lock = FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT)
        if path.exists():
            with lock:
                m = TradingModel()
                m.load(str(path), version=self._read_version("stock"))
                self._stock_model = m
                self._stock_mtime = path.stat().st_mtime
                logger.info("Stock model loaded: %s", m.version)

    def _load_forex(self):
        path = self._dir / FOREX_MODEL_FILE
        lock = FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT)
        if path.exists():
            with lock:
                m = TradingModel()
                m.load(str(path), version=self._read_version("forex"))
                self._forex_model = m
                self._forex_mtime = path.stat().st_mtime
                logger.info("Forex model loaded: %s", m.version)

    def _read_version(self, model_type: str) -> str:
        metrics_path = self._dir / METRICS_FILE
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            return metrics.get(f"{model_type}_version", "unknown")
        return "unknown"

    def check_and_reload(self):
        """
        Call before each decision cycle.
        If the model file on disk was updated (by retrain_pipeline), reload it.
        """
        stock_path = self._dir / STOCK_MODEL_FILE
        forex_path = self._dir / FOREX_MODEL_FILE

        if stock_path.exists():
            mtime = stock_path.stat().st_mtime
            if mtime > self._stock_mtime:
                logger.info("Stock model updated on disk — hot reloading")
                self._load_stock()

        if forex_path.exists():
            mtime = forex_path.stat().st_mtime
            if mtime > self._forex_mtime:
                logger.info("Forex model updated on disk — hot reloading")
                self._load_forex()

    def get_stock_model(self) -> Optional[TradingModel]:
        return self._stock_model

    def get_forex_model(self) -> Optional[TradingModel]:
        return self._forex_model

    def get_model(self, symbol: str) -> Optional[TradingModel]:
        """Return the appropriate model based on symbol format."""
        # Forex symbols use underscore: EUR_USD, GBP_USD, etc.
        if "_" in symbol:
            return self._forex_model
        return self._stock_model

    def save_champion(self, model: TradingModel, model_type: str, metrics: dict):
        """
        Atomically replace the champion model file.
        Called by model_validator.py after a successful challenge.
        """
        filename = STOCK_MODEL_FILE if model_type == "stock" else FOREX_MODEL_FILE
        path = self._dir / filename
        lock = FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT)

        with lock:
            # Write to temp file first, then atomic rename.
            # Must keep .json extension so XGBoost saves in JSON (not binary) format.
            tmp_path = path.with_name(path.stem + "_tmp.json")
            model.save(str(tmp_path))
            tmp_path.replace(path)

        # Update metrics.json
        metrics_path = self._dir / METRICS_FILE
        existing = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)

        existing[f"{model_type}_version"] = model.version
        existing[f"{model_type}_metrics"] = metrics

        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        logger.info("Champion %s model saved: %s", model_type, model.version)
