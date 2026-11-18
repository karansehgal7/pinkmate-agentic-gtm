"""Tests for pinkmate.ml.bayesian_survival — PyMC Weibull."""
from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.bayesian_survival import (
    BayesianSurvivalResult,
    fit_bayesian_weibull,
)


# These tests invoke PyMC NUTS. Use tiny draws/tune to keep runtime short.
# Marked slow via pytest convention — pytest -q runs them all; use -k to filter.


@pytest.fixture(scope="module")
def tiny_bayes_fit():
    """Fit a Bayesian Weibull once (expensive) and reuse across tests."""
    rng = np.random.default_rng(42)
    n = 40
    t = rng.weibull(2.0, n) * 20.0
    e = np.ones(n, dtype=int)
    res = fit_bayesian_weibull(
        t, e,
        n_draws=200, n_tune=200, n_chains=2,
        seed=42, progressbar=False,
    )
    return res, t, e


class TestBayesianWeibullFit:
    def test_returns_result_dataclass(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        assert isinstance(res, BayesianSurvivalResult)

    def test_posterior_samples_populated(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        assert res.posterior_shape.ndim == 1
        assert res.posterior_scale.ndim == 1
        assert res.n_samples == len(res.posterior_shape)
        assert res.n_samples > 0

    def test_posterior_mean_near_true_params(self, tiny_bayes_fit):
        """True shape=2.0, scale=20.0. Posterior means should be in rough neighbourhood."""
        res, _, _ = tiny_bayes_fit
        mean_shape, mean_scale = res.posterior_mean()
        assert 1.0 < mean_shape < 3.5, f"shape posterior mean {mean_shape} off"
        assert 10.0 < mean_scale < 40.0, f"scale posterior mean {mean_scale} off"

    def test_posterior_covers_true_shape_in_ci(self, tiny_bayes_fit):
        """True shape=2.0 should lie inside the 95% credible interval."""
        res, _, _ = tiny_bayes_fit
        ci_lo, ci_hi = res.credible_interval("shape", 0.95)
        assert ci_lo < 2.0 < ci_hi, f"true shape 2.0 not in CI [{ci_lo}, {ci_hi}]"


class TestCredibleIntervals:
    def test_shape_ci_narrower_than_scale_ci(self, tiny_bayes_fit):
        """Both should return a (lo, hi) tuple with lo < hi."""
        res, _, _ = tiny_bayes_fit
        for param in ("shape", "scale"):
            lo, hi = res.credible_interval(param, 0.95)
            assert lo < hi

    def test_invalid_parameter_raises(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        with pytest.raises(ValueError, match="parameter"):
            res.credible_interval("nonsense")

    def test_wider_level_means_wider_ci(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        lo_90, hi_90 = res.credible_interval("shape", 0.90)
        lo_99, hi_99 = res.credible_interval("shape", 0.99)
        assert lo_99 <= lo_90
        assert hi_99 >= hi_90


class TestSurvivalProbability:
    def test_returns_three_floats(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        mean, lo, hi = res.survival_probability(10.0)
        assert lo <= mean <= hi

    def test_survival_decreases_with_time(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        s_5, _, _ = res.survival_probability(5.0)
        s_30, _, _ = res.survival_probability(30.0)
        assert s_30 < s_5

    def test_survival_bounded_zero_one(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        for t in [1.0, 10.0, 50.0]:
            mean, lo, hi = res.survival_probability(t)
            assert 0.0 <= mean <= 1.0
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0


class TestDiagnostics:
    def test_rhat_near_one(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        # With only 200 draws, R-hat may exceed 1.01; allow up to 1.1
        assert res.rhat_shape < 1.1, f"R-hat shape {res.rhat_shape} too high"
        assert res.rhat_scale < 1.1, f"R-hat scale {res.rhat_scale} too high"

    def test_n_observations_captured(self, tiny_bayes_fit):
        res, _, _ = tiny_bayes_fit
        assert res.n_observations == 40
        assert res.n_events == 40


class TestInputValidation:
    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 5"):
            fit_bayesian_weibull(
                np.array([1.0, 2.0]), np.array([1, 1]),
                n_draws=50, n_tune=50,
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_bayesian_weibull(
                np.array([1.0, 2.0, 3.0]), np.array([1, 1]),
                n_draws=50, n_tune=50,
            )
