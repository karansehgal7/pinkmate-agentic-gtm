"""
Unit tests for pinkmate/ml/imbalance.py — SMOTE wrapper (paper §III.C).
"""

from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.imbalance import SMOTE_K_NEIGHBORS, apply_smote


def _imbalanced_toy(seed: int = 42, n_majority: int = 500, n_minority: int = 30):
    rng = np.random.default_rng(seed)
    X_maj = rng.normal(0.0, 1.0, size=(n_majority, 10))
    X_min = rng.normal(1.5, 0.8, size=(n_minority, 10))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority, dtype=int), np.ones(n_minority, dtype=int)])
    return X, y


class TestApplySmote:
    def test_minority_class_matches_majority_after_smote(self):
        X, y = _imbalanced_toy()
        X_r, y_r = apply_smote(X, y, seed=42)
        counts = {c: int((y_r == c).sum()) for c in np.unique(y_r)}
        assert counts[0] == counts[1], (
            f"Post-SMOTE classes should match in size; got {counts}"
        )

    def test_training_rows_increase(self):
        X, y = _imbalanced_toy(n_majority=500, n_minority=30)
        X_r, y_r = apply_smote(X, y, seed=42)
        assert len(X_r) > len(X)
        assert len(X_r) == 500 + 500  # oversampled to majority count

    def test_feature_dimensionality_preserved(self):
        X, y = _imbalanced_toy()
        X_r, _ = apply_smote(X, y, seed=42)
        assert X_r.shape[1] == X.shape[1]

    def test_deterministic_under_fixed_seed(self):
        X, y = _imbalanced_toy()
        X_a, y_a = apply_smote(X, y, seed=42)
        X_b, y_b = apply_smote(X, y, seed=42)
        np.testing.assert_array_equal(X_a, X_b)
        np.testing.assert_array_equal(y_a, y_b)

    def test_different_seeds_give_different_synthetic_samples(self):
        X, y = _imbalanced_toy()
        X_a, _ = apply_smote(X, y, seed=42)
        X_b, _ = apply_smote(X, y, seed=100)
        # Original rows at the top of the array should still match; synthetic
        # minority rows at the bottom should differ.
        np.testing.assert_array_equal(X_a[:len(X)], X_b[:len(X)])
        assert not np.allclose(X_a[len(X):], X_b[len(X):])

    def test_rejects_single_class(self):
        X = np.random.default_rng(42).normal(0, 1, size=(100, 5))
        y = np.zeros(100, dtype=int)
        with pytest.raises(ValueError, match="at least 2 classes"):
            apply_smote(X, y)

    def test_rejects_minority_smaller_than_k_neighbors(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 5))
        y = np.array([0] * 97 + [1] * 3, dtype=int)  # minority count 3 < 5
        with pytest.raises(ValueError, match="requires at least"):
            apply_smote(X, y, k_neighbors=SMOTE_K_NEIGHBORS)
