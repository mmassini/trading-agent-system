"""
Feature engineering: converts OHLCV bars into the ~50-feature matrix used
for both training and live inference. Must be IDENTICAL in both paths.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav
import ta.volume as tavol

logger = logging.getLogger(__name__)

# ── Feature names (in order) ─────────────────────────────────────────────────
# Changing this list invalidates any trained model. Update model version if modified.
FEATURE_COLUMNS = [
    # Returns
    "close_pct_1m", "close_pct_5m", "close_pct_15m",
    "log_return_1m", "log_return_5m",
    # Volume
    "volume_ratio", "volume_spike",
    # Momentum
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "cci_14", "roc_5", "mom_10",
    # Trend
    "ema_9", "ema_21", "ema_cross",
    "macd", "macd_signal", "macd_hist",
    "adx_14", "psar_dist",
    # Volatility
    "atr_14", "natr_14", "bb_pct_b", "bb_width",
    # Volume-based
    "vwap_dist", "obv_slope_5", "cmf_14",
    # Sentiment (external — filled in by ML agent from SentimentAgent output)
    "sentiment_score",
    # Time
    "minutes_since_open", "hour_of_day", "day_of_week",
    # Lag features (lag-1)
    "rsi_14_lag1", "macd_hist_lag1", "atr_14_lag1",
    "ema_cross_lag1", "volume_ratio_lag1",
    # Lag features (lag-2)
    "rsi_14_lag2", "macd_hist_lag2", "atr_14_lag2",
]


def build_features(
    df_1m: pd.DataFrame,
    sentiment_score: float = 0.0,
    market_open_utc: str = "13:30",
) -> pd.DataFrame:
    """
    Build the feature matrix from a DataFrame of 1-minute OHLCV bars.

    Args:
        df_1m: DataFrame with columns [open, high, low, close, volume, timestamp]
               sorted ascending by timestamp.
        sentiment_score: Latest FinBERT sentiment for the symbol (-1 to +1).
        market_open_utc: Market open time string (HH:MM UTC) for time features.

    Returns:
        DataFrame with FEATURE_COLUMNS, NaN rows dropped.
        Last row is the most recent (live) feature vector.
    """
    df = df_1m.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # ── Returns ───────────────────────────────────────────────────────────────
    df["close_pct_1m"] = c.pct_change(1)
    df["close_pct_5m"] = c.pct_change(5)
    df["close_pct_15m"] = c.pct_change(15)
    df["log_return_1m"] = np.log(c / c.shift(1))
    df["log_return_5m"] = np.log(c / c.shift(5))

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ma = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_ma.replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(float)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"] = tam.RSIIndicator(c, window=14).rsi()
    df["rsi_7"] = tam.RSIIndicator(c, window=7).rsi()

    stoch = tam.StochasticOscillator(h, lo, c, window=5, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["cci_14"] = tat.CCIIndicator(h, lo, c, window=14).cci()
    df["roc_5"] = tam.ROCIndicator(c, window=5).roc()
    df["mom_10"] = c.diff(10)  # Momentum = price difference over N periods

    # ── Trend ─────────────────────────────────────────────────────────────────
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    df["ema_9"] = ema9
    df["ema_21"] = ema21
    df["ema_cross"] = (ema9 - ema21) / c

    macd_ind = tat.MACD(c, window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    df["adx_14"] = tat.ADXIndicator(h, lo, c, window=14).adx()

    psar = tat.PSARIndicator(h, lo, c)
    psar_vals = psar.psar()
    df["psar_dist"] = (c - psar_vals) / c

    # ── Volatility ────────────────────────────────────────────────────────────
    df["atr_14"] = tav.AverageTrueRange(h, lo, c, window=14).average_true_range()
    df["natr_14"] = df["atr_14"] / c

    bb = tav.BollingerBands(c, window=20, window_dev=2)
    df["bb_pct_b"] = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / c

    # ── Volume-based ──────────────────────────────────────────────────────────
    # VWAP: approximate intraday VWAP using cumulative method
    typical_price = (h + lo + c) / 3
    cum_tp_vol = (typical_price * v).cumsum()
    cum_vol = v.cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_dist"] = (c - vwap) / c

    obv = tavol.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_slope_5"] = obv.diff(5) / (obv.abs().rolling(5).mean().replace(0, np.nan))

    df["cmf_14"] = tavol.ChaikinMoneyFlowIndicator(h, lo, c, v, window=14).chaikin_money_flow()

    # ── Sentiment (injected externally) ──────────────────────────────────────
    df["sentiment_score"] = sentiment_score

    # ── Time features ─────────────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts = df["timestamp"]
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True)

    open_h, open_m = [int(x) for x in market_open_utc.split(":")]
    open_minutes = open_h * 60 + open_m
    df["minutes_since_open"] = ts.apply(
        lambda t: (t.hour * 60 + t.minute) - open_minutes
    ).clip(lower=0)
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek

    # ── Lag features ──────────────────────────────────────────────────────────
    for col, lag in [
        ("rsi_14", 1), ("macd_hist", 1), ("atr_14", 1),
        ("ema_cross", 1), ("volume_ratio", 1),
        ("rsi_14", 2), ("macd_hist", 2), ("atr_14", 2),
    ]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ── Finalize ──────────────────────────────────────────────────────────────
    result = df[FEATURE_COLUMNS].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def build_labels(df_1m: pd.DataFrame, horizon: int = 5, threshold: float = 0.001) -> pd.Series:
    """
    Generate classification labels for training.

    Returns:
        Series with values:
            0 = SHORT  (future return < -threshold)
            1 = FLAT   (-threshold <= return <= threshold)
            2 = LONG   (future return > threshold)
    """
    c = df_1m["close"]
    future_return = c.pct_change(horizon).shift(-horizon)

    labels = pd.Series(index=df_1m.index, dtype=int)
    labels[:] = 1  # FLAT default
    labels[future_return < -threshold] = 0   # SHORT
    labels[future_return > threshold] = 2    # LONG
    return labels
