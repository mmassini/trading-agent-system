"""Unit tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from agents.ml_analysis.feature_engineer import (
    build_features,
    build_labels,
    FEATURE_COLUMNS,
)


def _make_bars(n: int = 150) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing."""
    np.random.seed(42)
    base = 100.0
    dates = [
        datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
        for i in range(n)
    ]
    closes = base + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    opens = closes + np.random.randn(n) * 0.1
    volumes = np.random.randint(1000, 100000, n).astype(float)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_feature_columns_match():
    df = _make_bars(150)
    features = build_features(df)
    assert list(features.columns) == FEATURE_COLUMNS


def test_no_look_ahead_bias():
    """Features at time T should not use data from T+1 or beyond."""
    df = _make_bars(150)
    features = build_features(df)
    # If all feature values at a given row are based on <= that row's data,
    # changing future rows should not affect current row features.
    features_original = build_features(df.copy())

    # Modify the last 10 rows
    df_modified = df.copy()
    df_modified.loc[140:, "close"] = 999.0
    features_modified = build_features(df_modified)

    # Row 130 features should be identical
    row_idx = 130
    orig_row = features_original.iloc[row_idx].dropna()
    mod_row = features_modified.iloc[row_idx].dropna()
    # Check that the same columns are present
    common_cols = orig_row.index.intersection(mod_row.index)
    pd.testing.assert_series_equal(orig_row[common_cols], mod_row[common_cols], check_names=False)


def test_output_shape():
    df = _make_bars(150)
    features = build_features(df)
    assert features.shape[0] == 150
    assert features.shape[1] == len(FEATURE_COLUMNS)


def test_labels_three_classes():
    df = _make_bars(150)
    labels = build_labels(df, horizon=5)
    unique_labels = set(labels.dropna().unique())
    assert unique_labels.issubset({0, 1, 2})


def test_atr_positive():
    df = _make_bars(150)
    features = build_features(df)
    atr_values = features["atr_14"].dropna()
    assert (atr_values > 0).all(), "ATR must be positive"


def test_features_no_infinite():
    df = _make_bars(200)
    features = build_features(df)
    assert not np.isinf(features.values).any(), "No infinite values in features"
