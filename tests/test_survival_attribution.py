"""
Tests for pinkmate.ml.survival_attribution — SurvSHAP + σ_SHAP on survival models.

Enforces:
    - σ_SHAP primitive bounded/non-negative/reproducible
    - End-to-end rotated-background wrapper returns a SurvivalAttributionResult
    - Falls back to finite-difference attribution when survshap is unavailable
    - Spearman transfer correlation helper is stable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pinkmate.ml.survival import build_survival_model
from pinkmate.ml.survival_attribution import (
    DEFAULT_K_ROTATED_BACKGROUNDS,
    DEFAULT_TIME_GRID,
    SurvivalAttributionResult,
    compute_rotated_survshap_variance,
    sigma_shap_hazard,
    spearman_transfer_correlation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_survival_setup():
    """Fitted Cox PH on synthetic Weibull data."""
    rng = np.random.default_rng(0)
    n = 200
    p = 3
    X = rng.normal(size=(n, p))
    beta = np.array([0.4, -0.2, 0.1])
    linpred = X @ beta
    T = np.exp(linpred) * (-np.log(1 - rng.uniform(size=n))) ** (1.0 / 1.5)
    C = 90.0
    time = np.minimum(T, C)
    event = (T < C).astype(int)
    columns = tuple(f"x{i}" for i in range(p))
    X_df = pd.DataFrame(X, columns=columns)
    model = build_survival_model("cox_ph").fit(X_df, time, event)
    return model, X_df, columns


# ===========================================================================
# TestSigmaSHAPHazardPrimitive
# ===========================================================================

class TestSigmaSHAPHazardPrimitive:
    def test_zero_variance_zero_sigma(self):
        assert sigma_shap_hazard(np.zeros(5)) == 0.0

    def test_matches_sqrt_mean_variance(self):
        v = np.array([0.1, 0.4, 0.2, 0.05])
        expected = float(np.sqrt(v.mean()))
        assert sigma_shap_hazard(v) == pytest.approx(expected)

    def test_negative_input_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            sigma_shap_hazard(np.array([0.1, -0.05]))

    def test_non_1d_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            sigma_shap_hazard(np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_sigma_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            v = rng.uniform(0, 1, size=7)
            assert sigma_shap_hazard(v) >= 0.0


# ===========================================================================
# TestRotatedSurvSHAPVariance
# ===========================================================================

class TestRotatedSurvSHAPVariance:
    def test_returns_survival_attribution_result(self, small_survival_setup):
        model, X, cols = small_survival_setup
        res = compute_rotated_survshap_variance(
            model=model,
            model_key="cox_ph",
            X_train=X,
            X_reference=X.iloc[:20],
            feature_names=cols,
            k_backgrounds=3,
            background_size=50,
        )
        assert isinstance(res, SurvivalAttributionResult)
        assert res.model_key == "cox_ph"
        assert res.k_rotations == 3
        assert res.per_feature_variance.shape == (len(cols),)

    def test_sigma_non_negative(self, small_survival_setup):
        model, X, cols = small_survival_setup
        res = compute_rotated_survshap_variance(
            model=model,
            model_key="cox_ph",
            X_train=X,
            X_reference=X.iloc[:20],
            feature_names=cols,
            k_backgrounds=3,
            background_size=50,
        )
        assert res.sigma_shap_hazard >= 0

    def test_reproducibility_same_seed(self, small_survival_setup):
        model, X, cols = small_survival_setup
        res1 = compute_rotated_survshap_variance(
            model=model, model_key="cox_ph", X_train=X, X_reference=X.iloc[:15],
            feature_names=cols, k_backgrounds=3, background_size=40, seed=7,
            use_survshap_package=False,  # force deterministic finite-diff path
        )
        res2 = compute_rotated_survshap_variance(
            model=model, model_key="cox_ph", X_train=X, X_reference=X.iloc[:15],
            feature_names=cols, k_backgrounds=3, background_size=40, seed=7,
            use_survshap_package=False,
        )
        assert res1.sigma_shap_hazard == pytest.approx(res2.sigma_shap_hazard)

    def test_invalid_k_raises(self, small_survival_setup):
        model, X, cols = small_survival_setup
        with pytest.raises(ValueError, match="k_backgrounds"):
            compute_rotated_survshap_variance(
                model=model, model_key="cox_ph", X_train=X, X_reference=X.iloc[:10],
                feature_names=cols, k_backgrounds=0,
            )

    def test_empty_reference_raises(self, small_survival_setup):
        model, X, cols = small_survival_setup
        with pytest.raises(ValueError, match="empty"):
            compute_rotated_survshap_variance(
                model=model, model_key="cox_ph", X_train=X, X_reference=X.iloc[:0],
                feature_names=cols,
            )

    def test_default_k_matches_paper_canonical(self):
        assert DEFAULT_K_ROTATED_BACKGROUNDS == 50

    def test_default_time_grid_is_monthly_in_90_day_window(self):
        assert DEFAULT_TIME_GRID == (30.0, 60.0, 90.0)


# ===========================================================================
# TestSpearmanTransferCorrelation
# ===========================================================================

class TestSpearmanTransferCorrelation:
    def test_perfect_positive_correlation(self):
        """Matched σ_SHAP values across 4 pairs → ρ close to 1."""
        sigmac = {"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.7}
        sigmas = {"A": 0.2, "B": 0.4, "C": 0.6, "D": 0.8}
        pairs = [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]
        rho, lo, hi, n = spearman_transfer_correlation(sigmac, sigmas, pairs)
        assert rho == pytest.approx(1.0)
        assert n == 4

    def test_fewer_than_three_pairs_returns_nan(self):
        rho, lo, hi, n = spearman_transfer_correlation(
            {"a": 0.1, "b": 0.2}, {"A": 0.3, "B": 0.4},
            [("a", "A"), ("b", "B")],
        )
        assert np.isnan(rho)
        assert n == 2

    def test_missing_key_pair_skipped(self):
        sigmac = {"a": 0.1, "c": 0.5}
        sigmas = {"A": 0.2, "C": 0.6}
        pairs = [("a", "A"), ("b_missing", "B_missing"), ("c", "C")]
        rho, lo, hi, n = spearman_transfer_correlation(sigmac, sigmas, pairs)
        # Only 2 valid pairs → nan
        assert np.isnan(rho)
        assert n == 2
