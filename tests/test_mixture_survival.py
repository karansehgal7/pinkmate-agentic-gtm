"""Tests for pinkmate.ml.mixture_survival — Weibull mixture via EM."""
from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.mixture_survival import (
    DEFAULT_K_CANDIDATES,
    MixtureFitResult,
    WeibullComponent,
    fit_weibull_mixture,
    select_k_by_bic,
)


@pytest.fixture
def two_regime_data():
    """Synthetic two-regime data: fast converters (scale=5) + slow (scale=30)."""
    rng = np.random.default_rng(0)
    n_per = 150
    t_fast = rng.weibull(1.8, n_per) * 5.0
    t_slow = rng.weibull(1.8, n_per) * 30.0
    t = np.concatenate([t_fast, t_slow])
    # All events observed (no censoring in this fixture)
    e = np.ones(2 * n_per, dtype=int)
    return t, e


class TestWeibullComponent:
    def test_pdf_non_negative(self):
        c = WeibullComponent(shape=2.0, scale=5.0)
        t = np.linspace(0.1, 20.0, 50)
        assert (c.pdf(t) >= 0).all()

    def test_survival_decreasing(self):
        c = WeibullComponent(shape=2.0, scale=5.0)
        t = np.linspace(0.1, 20.0, 50)
        S = c.survival(t)
        assert (np.diff(S) <= 1e-9).all()

    def test_survival_at_zero_is_one(self):
        c = WeibullComponent(shape=2.0, scale=5.0)
        # S(0+) should be very close to 1
        assert c.survival(np.array([0.001]))[0] > 0.99


class TestFitWeibullMixture:
    def test_k_eq_2_returns_two_components(self, two_regime_data):
        t, e = two_regime_data
        res = fit_weibull_mixture(t, e, k=2)
        assert res.k == 2
        assert len(res.components) == 2
        assert res.mixing_weights.shape == (2,)
        assert res.mixing_weights.sum() == pytest.approx(1.0)

    def test_two_components_recover_distinct_scales(self, two_regime_data):
        """EM should identify two components with distinct scales."""
        t, e = two_regime_data
        res = fit_weibull_mixture(t, e, k=2)
        scales = sorted(c.scale for c in res.components)
        # True scales: 5 and 30. Ratio should be substantial (>3x)
        assert scales[1] / scales[0] > 3.0

    def test_log_likelihood_finite(self, two_regime_data):
        t, e = two_regime_data
        res = fit_weibull_mixture(t, e, k=2)
        assert np.isfinite(res.log_likelihood)

    def test_bic_and_aic_finite(self, two_regime_data):
        t, e = two_regime_data
        res = fit_weibull_mixture(t, e, k=2)
        assert np.isfinite(res.bic)
        assert np.isfinite(res.aic)

    def test_mixing_weights_non_negative(self, two_regime_data):
        t, e = two_regime_data
        res = fit_weibull_mixture(t, e, k=3)
        assert (res.mixing_weights >= -1e-9).all()

    def test_invalid_k_raises(self, two_regime_data):
        t, e = two_regime_data
        with pytest.raises(ValueError, match="k must be"):
            fit_weibull_mixture(t, e, k=0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_weibull_mixture(np.array([1.0, 2.0]), np.array([1]), k=2)


class TestBICSelection:
    def test_select_k_returns_tuple(self, two_regime_data):
        t, e = two_regime_data
        best_k, all_results = select_k_by_bic(t, e)
        assert best_k in DEFAULT_K_CANDIDATES
        assert isinstance(all_results, dict)
        for k in DEFAULT_K_CANDIDATES:
            if k in all_results:
                assert isinstance(all_results[k], MixtureFitResult)
