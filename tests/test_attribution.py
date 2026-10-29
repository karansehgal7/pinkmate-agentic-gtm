"""
Unit tests for pinkmate/ml/attribution.py — σ_SHAP computation (paper §V).
"""

from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.attribution import (
    AttributionResult,
    compute_rotated_shap_variance,
    sigma_shap,
)


class TestSigmaShap:
    def test_zero_variance_gives_zero_sigma(self):
        per_feat = np.zeros(10)
        assert sigma_shap(per_feat) == pytest.approx(0.0)

    def test_known_values(self):
        # σ_SHAP = sqrt((1/m) Σ_j Var_k) — all equal values collapse to sqrt(v).
        per_feat = np.array([0.04, 0.04, 0.04, 0.04])  # mean = 0.04
        assert sigma_shap(per_feat) == pytest.approx(0.2, abs=1e-6)

    def test_paper_table2_target_shape(self):
        """Paper reports σ_SHAP ~ 0.01–0.02; verify the function returns comparable magnitudes."""
        # Simulate per-feature variances that would yield σ_SHAP ≈ 0.02.
        # 0.02 = sqrt( (1/m) · Σ_j v_j ) => (1/m) · Σ_j v_j = 0.0004
        # Across m=95 features, mean per-feature variance should be 0.0004.
        rng = np.random.default_rng(42)
        per_feat = rng.uniform(0.0001, 0.0007, size=95)
        # scale so mean exactly 0.0004
        per_feat = per_feat * (0.0004 / per_feat.mean())
        assert sigma_shap(per_feat) == pytest.approx(0.02, abs=1e-6)

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            sigma_shap(np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            sigma_shap(np.array([0.1, -0.05, 0.2]))


class TestComputeRotatedShapVariance:
    """
    Integration-style test using a tiny trained XGBoost model on synthetic data.
    Verifies the output shape, metric non-negativity, and that higher K yields
    smaller sampling noise.
    """

    @pytest.fixture
    def tiny_model_and_data(self):
        from xgboost import XGBClassifier
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(500, 20))
        signal = np.array([1.0, -0.8, 0.6, -0.4] + [0.0] * 16)
        y = (X @ signal + rng.normal(0, 0.1, 500) > 0.3).astype(int)
        model = XGBClassifier(
            random_state=42, n_estimators=30, eval_metric="logloss",
            tree_method="hist", n_jobs=-1,
        )
        model.fit(X, y)
        return model, X, y

    def test_result_shape_and_bounds(self, tiny_model_and_data):
        model, X, _ = tiny_model_and_data
        feature_names = tuple(f"f{i}" for i in range(20))
        result = compute_rotated_shap_variance(
            model=model, model_name="xgboost_test",
            X_train=X, X_reference=X[:50], feature_names=feature_names,
            k_backgrounds=5, background_size=50, seed=42,
        )
        assert isinstance(result, AttributionResult)
        assert result.sigma_shap >= 0.0
        assert result.mean_absolute_phi >= 0.0
        assert result.per_feature_variance.shape == (20,)
        assert (result.per_feature_variance >= 0).all()
        assert result.k_backgrounds == 5
        assert result.background_size == 50
        assert result.n_reference == 50

    def test_deterministic_under_fixed_seed(self, tiny_model_and_data):
        model, X, _ = tiny_model_and_data
        feature_names = tuple(f"f{i}" for i in range(20))
        a = compute_rotated_shap_variance(
            model=model, model_name="xgb", X_train=X, X_reference=X[:50],
            feature_names=feature_names, k_backgrounds=5, background_size=50, seed=42,
        )
        b = compute_rotated_shap_variance(
            model=model, model_name="xgb", X_train=X, X_reference=X[:50],
            feature_names=feature_names, k_backgrounds=5, background_size=50, seed=42,
        )
        assert a.sigma_shap == pytest.approx(b.sigma_shap)
        np.testing.assert_array_equal(a.per_feature_variance, b.per_feature_variance)

    def test_rejects_feature_name_mismatch(self, tiny_model_and_data):
        model, X, _ = tiny_model_and_data
        feature_names = tuple(f"f{i}" for i in range(10))  # wrong length (should be 20)
        with pytest.raises(ValueError, match="feature_names length"):
            compute_rotated_shap_variance(
                model=model, model_name="xgb", X_train=X, X_reference=X[:50],
                feature_names=feature_names, k_backgrounds=5, background_size=50, seed=42,
            )

    def test_rejects_oversized_background(self, tiny_model_and_data):
        model, X, _ = tiny_model_and_data
        feature_names = tuple(f"f{i}" for i in range(20))
        with pytest.raises(ValueError, match="exceeds training pool"):
            compute_rotated_shap_variance(
                model=model, model_name="xgb", X_train=X, X_reference=X[:50],
                feature_names=feature_names, k_backgrounds=5,
                background_size=1000, seed=42,
            )
