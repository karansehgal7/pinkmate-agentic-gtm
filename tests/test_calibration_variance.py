"""
Tests for pinkmate.ml.calibration_variance — σ_calibration governance signal.

Enforces the three properties σ_calibration MUST satisfy:
    1. Bounded: σ_ECE ∈ [0, 0.5], σ_Brier ∈ [0, 0.5]
    2. K-stable: σ at K=50 close to σ at K=10 on well-behaved synthetic data
    3. Reproducible: same seed → identical σ
"""

from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.calibration_variance import (
    BOUND_UPPER,
    CalibrationResult,
    K_DEFAULT,
    compute_brier,
    compute_ece,
    compute_rotated_calibration_variance,
    sigma_calibration,
    sweep_k_stability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictor():
    """A perfectly calibrated predictor on a Bernoulli(0.5) task."""
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.5).astype(int)

    def predict_proba(X):
        # Return the true probability for every row of X (which we treat
        # as a proxy for the same underlying distribution)
        return np.full(X.shape[0], 0.5)

    X = np.zeros((n, 3))
    return X, y, predict_proba


@pytest.fixture
def synthetic_logistic_data():
    """Synthetic logistic data + a reasonably calibrated predictor."""
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.normal(size=(n, 5))
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < y_prob).astype(int)

    def predict_proba(X_in):
        logits_in = 0.8 * X_in[:, 0] - 0.5 * X_in[:, 1] + 0.3 * X_in[:, 2]
        return 1.0 / (1.0 + np.exp(-logits_in))

    return X, y, predict_proba


# ===========================================================================
# TestECE — single-shot Expected Calibration Error
# ===========================================================================

class TestECE:
    def test_perfect_calibration_returns_near_zero(self):
        """Predictor matching empirical frequency → ECE ≈ 0."""
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        p = np.array([0.5] * 10)
        ece = compute_ece(y, p, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01), (
            f"ECE on perfectly-calibrated 0.5/0.5 data should be ~0; got {ece}"
        )

    def test_worst_case_calibration_near_one(self):
        """Predicting 1.0 for all negatives → ECE close to 1.0."""
        y = np.zeros(100)
        p = np.ones(100)
        ece = compute_ece(y, p, n_bins=10)
        assert ece == pytest.approx(1.0, abs=0.01), (
            f"ECE when always predicting 1.0 for negatives should be ~1.0; got {ece}"
        )

    def test_ece_is_bounded_zero_one(self):
        """ECE ∈ [0, 1] for any inputs."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            y = (rng.random(200) < 0.3).astype(int)
            p = rng.random(200)
            ece = compute_ece(y, p, n_bins=10)
            assert 0.0 <= ece <= 1.0, f"ECE {ece} outside [0, 1]"

    def test_empty_input_returns_zero(self):
        """Empty arrays → ECE = 0.0 (no crash)."""
        assert compute_ece(np.array([]), np.array([])) == 0.0

    def test_length_mismatch_raises(self):
        """y and p of different lengths → ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            compute_ece(np.array([0, 1]), np.array([0.5]))


# ===========================================================================
# TestBrier — single-shot Brier score
# ===========================================================================

class TestBrier:
    def test_perfect_prediction_zero(self):
        """Predicting label exactly → Brier = 0."""
        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        assert compute_brier(y, p) == pytest.approx(0.0)

    def test_worst_prediction_one(self):
        """Predicting opposite of label → Brier = 1."""
        y = np.array([0, 1, 0, 1])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier(y, p) == pytest.approx(1.0)

    def test_half_probability_quarter_brier(self):
        """Always predicting 0.5 on 50/50 data → Brier = 0.25."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        p = np.full(8, 0.5)
        assert compute_brier(y, p) == pytest.approx(0.25)

    def test_brier_bounded_zero_one(self):
        """Brier ∈ [0, 1] for any inputs in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            y = (rng.random(100) < 0.5).astype(int)
            p = rng.random(100)
            brier = compute_brier(y, p)
            assert 0.0 <= brier <= 1.0, f"Brier {brier} outside [0, 1]"


# ===========================================================================
# TestSigmaCalibration — the primitive σ operator
# ===========================================================================

class TestSigmaCalibration:
    def test_identical_rotations_give_zero(self):
        """All K metrics equal → σ = 0."""
        assert sigma_calibration(np.array([0.1] * 20)) == pytest.approx(0.0)

    def test_single_rotation_returns_zero(self):
        """K < 2 → σ = 0 (no variance defined)."""
        assert sigma_calibration(np.array([0.1])) == 0.0
        assert sigma_calibration(np.array([])) == 0.0

    def test_matches_numpy_std_ddof_one(self):
        """σ_calibration IS sample std with ddof=1."""
        vals = np.array([0.1, 0.15, 0.12, 0.18, 0.11])
        assert sigma_calibration(vals) == pytest.approx(
            float(np.std(vals, ddof=1))
        )

    def test_negative_input_raises(self):
        """Calibration metrics are non-negative; reject negatives."""
        with pytest.raises(ValueError, match="non-negative"):
            sigma_calibration(np.array([0.1, -0.05, 0.2]))

    def test_property_1_bounded_by_half(self):
        """
        PROPERTY 1 (Bounded): σ of values in [0, 1] is at most 0.5.

        Proven: max sample std of values in [0, 1] is 0.5 (achieved by
        the degenerate point mass at 0 and 1 in equal parts).
        """
        rng = np.random.default_rng(0)
        for _ in range(20):
            vals = rng.random(50)  # in [0, 1]
            s = sigma_calibration(vals)
            assert s <= BOUND_UPPER + 1e-9, (
                f"σ_calibration = {s} exceeds theoretical bound {BOUND_UPPER}"
            )


# ===========================================================================
# TestRotatedCalibrationVariance — the full measurement harness
# ===========================================================================

class TestRotatedCalibrationVariance:
    def test_returns_calibration_result(self, synthetic_logistic_data):
        """Entry point returns a well-formed CalibrationResult."""
        X, y, pf = synthetic_logistic_data
        res = compute_rotated_calibration_variance(X, y, pf, k_rotations=10)
        assert isinstance(res, CalibrationResult)
        assert res.k_rotations == 10
        assert res.n_samples == len(y)
        assert res.per_rotation_ece.shape == (10,)
        assert res.per_rotation_brier.shape == (10,)
        assert 0.0 <= res.sigma_ece <= BOUND_UPPER
        assert 0.0 <= res.sigma_brier <= BOUND_UPPER

    def test_property_1_bounded_on_real_pipeline(self, synthetic_logistic_data):
        """PROPERTY 1 (Bounded) holds end-to-end through the pipeline."""
        X, y, pf = synthetic_logistic_data
        res = compute_rotated_calibration_variance(X, y, pf, k_rotations=50)
        assert 0.0 <= res.sigma_ece <= BOUND_UPPER
        assert 0.0 <= res.sigma_brier <= BOUND_UPPER

    def test_property_3_reproducibility_same_seed(self, synthetic_logistic_data):
        """PROPERTY 3 (Reproducibility): same seed → identical σ."""
        X, y, pf = synthetic_logistic_data
        r1 = compute_rotated_calibration_variance(X, y, pf, k_rotations=20, seed=7)
        r2 = compute_rotated_calibration_variance(X, y, pf, k_rotations=20, seed=7)
        assert r1.sigma_ece == pytest.approx(r2.sigma_ece)
        assert r1.sigma_brier == pytest.approx(r2.sigma_brier)
        np.testing.assert_array_equal(r1.per_rotation_ece, r2.per_rotation_ece)
        np.testing.assert_array_equal(r1.per_rotation_brier, r2.per_rotation_brier)

    def test_different_seed_different_realization(self, synthetic_logistic_data):
        """Sanity: different seed → different per-rotation trajectory (σ may still agree in expectation)."""
        X, y, pf = synthetic_logistic_data
        r1 = compute_rotated_calibration_variance(X, y, pf, k_rotations=20, seed=1)
        r2 = compute_rotated_calibration_variance(X, y, pf, k_rotations=20, seed=99)
        # Per-rotation trajectories must differ (otherwise RNG seeding broken)
        assert not np.array_equal(r1.per_rotation_ece, r2.per_rotation_ece)

    def test_accepts_sklearn_shape_proba(self, synthetic_logistic_data):
        """predict_proba returning (n, 2) sklearn-style is handled."""
        X, y, pf = synthetic_logistic_data

        def pf_sklearn(X_in):
            p1 = pf(X_in)
            return np.column_stack([1.0 - p1, p1])

        res = compute_rotated_calibration_variance(X, y, pf_sklearn, k_rotations=10)
        assert 0.0 <= res.sigma_ece <= BOUND_UPPER

    def test_empty_test_set_raises(self):
        """Empty test set → ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_rotated_calibration_variance(
                X_test=np.zeros((0, 3)),
                y_test=np.array([]),
                predict_proba_fn=lambda X: np.array([]),
            )

    def test_invalid_k_raises(self, synthetic_logistic_data):
        """K < 1 → ValueError."""
        X, y, pf = synthetic_logistic_data
        with pytest.raises(ValueError, match="k_rotations must be"):
            compute_rotated_calibration_variance(X, y, pf, k_rotations=0)


# ===========================================================================
# TestKStability — PROPERTY 2
# ===========================================================================

class TestKStability:
    def test_property_2_k_sweep_bounded(self, synthetic_logistic_data):
        """σ at every K value in the sweep stays within [0, 0.5]."""
        X, y, pf = synthetic_logistic_data
        results = sweep_k_stability(X, y, pf, k_values=(5, 10, 20, 50))
        for k, res in results.items():
            assert 0.0 <= res.sigma_ece <= BOUND_UPPER, (
                f"K={k}: σ_ECE={res.sigma_ece} outside [0, {BOUND_UPPER}]"
            )
            assert 0.0 <= res.sigma_brier <= BOUND_UPPER

    def test_property_2_convergence_k10_to_k50(self, synthetic_logistic_data):
        """
        PROPERTY 2 (K-stability): σ at K=50 close to σ at K=10 on
        well-behaved synthetic data. Tolerance is empirical — bootstrap
        std at K=10 has inherent K^(-0.5) noise.
        """
        X, y, pf = synthetic_logistic_data
        results = sweep_k_stability(X, y, pf, k_values=(10, 50))
        s10 = results[10].sigma_ece
        s50 = results[50].sigma_ece
        diff = abs(s10 - s50)
        # Both should be small (< 0.05) and within 0.05 of each other
        assert diff < 0.05, (
            f"K-stability failed: |σ(K=10) - σ(K=50)| = {diff:.4f} "
            f"on well-behaved synthetic data (σ_ECE at K=10={s10:.4f}, "
            f"K=50={s50:.4f})"
        )

    def test_property_2_default_k_is_50(self):
        """K_DEFAULT matches paper's canonical σ_SHAP protocol."""
        assert K_DEFAULT == 50


# ===========================================================================
# TestConstants — smoke tests on module-level constants
# ===========================================================================

class TestConstants:
    def test_bound_upper_is_half(self):
        """BOUND_UPPER equals the theoretical σ cap of 0.5."""
        assert BOUND_UPPER == 0.5

    def test_k_default_matches_paper(self):
        """K_DEFAULT = 50, matching paper §V."""
        assert K_DEFAULT == 50
