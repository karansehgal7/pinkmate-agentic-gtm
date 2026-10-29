"""
Unit tests for pinkmate/ml/dataset.py — Taiwan Bankruptcy loader + preprocessing.

Uses the synthetic Taiwan-shaped dataset so the test suite stays hermetic
and does not require UCI network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.dataset import (
    TAIWAN_MINORITY_RATE,
    TAIWAN_N_FEATURES,
    TAIWAN_N_ROWS,
    TAIWAN_RANDOM_SEED,
    make_synthetic_taiwan_like,
    preprocess,
)


class TestMakeSyntheticTaiwanLike:
    def test_shape_matches_paper_declaration(self):
        ds = make_synthetic_taiwan_like()
        assert len(ds.X) == TAIWAN_N_ROWS
        assert ds.X.shape[1] == TAIWAN_N_FEATURES
        assert ds.source == "synthetic"

    def test_minority_rate_matches_target_within_rounding(self):
        ds = make_synthetic_taiwan_like()
        actual_rate = float(ds.y.mean())
        assert actual_rate == pytest.approx(TAIWAN_MINORITY_RATE, abs=1e-4)

    def test_feature_names_populated_and_unique(self):
        ds = make_synthetic_taiwan_like()
        assert len(ds.feature_names) == TAIWAN_N_FEATURES
        assert len(set(ds.feature_names)) == TAIWAN_N_FEATURES

    def test_deterministic_under_fixed_seed(self):
        a = make_synthetic_taiwan_like(seed=42)
        b = make_synthetic_taiwan_like(seed=42)
        assert a.sha256 == b.sha256
        np.testing.assert_array_equal(a.X.values, b.X.values)
        np.testing.assert_array_equal(a.y.values, b.y.values)


class TestPreprocess:
    def test_stratified_split_preserves_minority_rate(self):
        ds = make_synthetic_taiwan_like()
        split = preprocess(ds.X, ds.y, seed=TAIWAN_RANDOM_SEED)
        rate_train = split.y_train.mean()
        rate_test = split.y_test.mean()
        full_rate = ds.y.mean()
        # Stratification should preserve minority rate within 0.5 pp of full rate.
        assert abs(rate_train - full_rate) < 0.005
        assert abs(rate_test - full_rate) < 0.005

    def test_eighty_twenty_split_sizes(self):
        ds = make_synthetic_taiwan_like()
        split = preprocess(ds.X, ds.y, seed=TAIWAN_RANDOM_SEED)
        total = len(ds.X)
        expected_train = int(round(total * 0.80))
        # Allow 1-row off-by-one for the stratified split's row-level rounding.
        assert abs(len(split.X_train) - expected_train) <= 1

    def test_feature_names_preserved(self):
        ds = make_synthetic_taiwan_like()
        split = preprocess(ds.X, ds.y, seed=TAIWAN_RANDOM_SEED)
        assert split.feature_names == ds.feature_names

    def test_scaler_fit_on_train_only(self):
        """Standardisation params must come from the train partition only."""
        ds = make_synthetic_taiwan_like()
        split = preprocess(ds.X, ds.y, seed=TAIWAN_RANDOM_SEED)
        # Train partition should be ~0 mean after scaling (within tolerance).
        assert abs(split.X_train.mean()) < 0.01
        assert abs(split.X_train.std() - 1.0) < 0.05
        # Test partition is NOT guaranteed exactly 0-mean because scaler was fit
        # on train only, but should be close given the stratified draw.
        assert abs(split.X_test.mean()) < 0.1

    def test_deterministic_under_fixed_seed(self):
        ds = make_synthetic_taiwan_like()
        a = preprocess(ds.X, ds.y, seed=42)
        b = preprocess(ds.X, ds.y, seed=42)
        np.testing.assert_array_equal(a.X_train, b.X_train)
        np.testing.assert_array_equal(a.X_test, b.X_test)
        np.testing.assert_array_equal(a.y_train, b.y_train)
        np.testing.assert_array_equal(a.y_test, b.y_test)
