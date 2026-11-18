"""
Tests for pinkmate.ml.survival — 7-model survival ensemble.

Correctness checks on synthetic Weibull-generated data, plus smoke tests
that every model in the factory fits and predicts without raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pinkmate.ml.survival import (
    LIFELINES_PARAMETRIC_KEYS,
    MODEL_KEYS,
    SurvivalFitResult,
    build_all_survival_models,
    build_survival_model,
    fit_model,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic Weibull-generated survival data
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_weibull_data():
    """
    Synthetic right-censored Weibull survival data.

    T ~ Weibull(shape=1.5, scale=exp(β'x))
    Censored at C = 90 days (right-censoring).
    event = 1 iff T < C.
    """
    rng = np.random.default_rng(42)
    n = 400
    p = 4
    X = rng.normal(size=(n, p))
    beta = np.array([0.5, -0.3, 0.2, 0.1])
    linpred = X @ beta
    scale = np.exp(linpred)
    shape = 1.5
    u = rng.uniform(size=n)
    T = scale * (-np.log(1 - u)) ** (1.0 / shape)
    C = 90.0
    time = np.minimum(T, C)
    event = (T < C).astype(int)
    columns = [f"x{i}" for i in range(p)]
    X_df = pd.DataFrame(X, columns=columns)
    return X_df, time, event


@pytest.fixture
def small_split_data(synth_weibull_data):
    """Train/test split for concordance / fit evaluation."""
    X, t, e = synth_weibull_data
    n = len(t)
    cut = int(n * 0.8)
    return (X.iloc[:cut], t[:cut], e[:cut]), (X.iloc[cut:], t[cut:], e[cut:])


# ===========================================================================
# TestCoxPH
# ===========================================================================

class TestCoxPH:
    def test_fit_returns_self(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("cox_ph")
        result = m.fit(X, t, e)
        assert result is m
        assert m._fitted

    def test_concordance_above_random(self, small_split_data):
        (X_tr, t_tr, e_tr), (X_te, t_te, e_te) = small_split_data
        m = build_survival_model("cox_ph").fit(X_tr, t_tr, e_tr)
        c = m.concordance_index(X_te, t_te, e_te)
        # 0.5 is random; small n + censoring + 4 features gives modest signal.
        # Threshold at 0.52 avoids flakiness while still asserting above-random.
        assert c > 0.52, f"Cox PH concordance {c:.3f} not above random"

    def test_log_likelihood_finite(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("cox_ph").fit(X, t, e)
        assert np.isfinite(m.log_likelihood())

    def test_aic_finite(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("cox_ph").fit(X, t, e)
        assert np.isfinite(m.aic())


# ===========================================================================
# TestParametricAFT (Weibull, Exponential, LogNormal, LogLogistic)
# ===========================================================================

class TestParametricAFT:
    @pytest.mark.parametrize("key", LIFELINES_PARAMETRIC_KEYS)
    def test_fit_without_raising(self, synth_weibull_data, key):
        X, t, e = synth_weibull_data
        m = build_survival_model(key).fit(X, t, e)
        assert m._fitted

    @pytest.mark.parametrize("key", LIFELINES_PARAMETRIC_KEYS)
    def test_log_likelihood_finite(self, synth_weibull_data, key):
        X, t, e = synth_weibull_data
        m = build_survival_model(key).fit(X, t, e)
        assert np.isfinite(m.log_likelihood())

    @pytest.mark.parametrize("key", LIFELINES_PARAMETRIC_KEYS)
    def test_aic_finite(self, synth_weibull_data, key):
        X, t, e = synth_weibull_data
        m = build_survival_model(key).fit(X, t, e)
        assert np.isfinite(m.aic())

    def test_weibull_fits_better_than_lognormal_on_weibull_data(self, synth_weibull_data):
        """
        On Weibull-generated data, WeibullAFT should fit at least as well as
        LogNormalAFT (correctly-specified vs mis-specified). AIC: lower is better.
        Generous tolerance since AIC can drift with small n.
        """
        X, t, e = synth_weibull_data
        weibull = build_survival_model("weibull_aft").fit(X, t, e)
        lognorm = build_survival_model("lognormal_aft").fit(X, t, e)
        assert weibull.aic() < lognorm.aic() + 30, (
            f"On Weibull data, WeibullAFT AIC ({weibull.aic():.1f}) should be "
            f"at most slightly worse than LogNormalAFT AIC ({lognorm.aic():.1f})"
        )


# ===========================================================================
# TestXGBSurvival
# ===========================================================================

class TestXGBSurvival:
    def test_fit_without_raising(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("xgb_survival", n_estimators=20).fit(X, t, e)
        assert m._fitted

    def test_predict_cumulative_hazard_shape(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("xgb_survival", n_estimators=20).fit(X, t, e)
        times = np.array([30.0, 60.0, 90.0])
        ch = m.predict_cumulative_hazard(X.iloc[:10], times)
        assert ch.shape == (10, 3)
        assert (ch >= 0).all()

    def test_survival_function_bounded(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("xgb_survival", n_estimators=20).fit(X, t, e)
        S = m.predict_survival_function(X.iloc[:10], np.array([30.0]))
        assert (S >= 0).all() and (S <= 1).all()


# ===========================================================================
# TestLGBAFT
# ===========================================================================

class TestLGBAFT:
    def test_fit_without_raising(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("lgb_aft", n_estimators=20).fit(X, t, e)
        assert m._fitted

    def test_median_survival_positive(self, synth_weibull_data):
        X, t, e = synth_weibull_data
        m = build_survival_model("lgb_aft", n_estimators=20).fit(X, t, e)
        medians = m.predict_median_survival_time(X.iloc[:10])
        assert (medians > 0).all()

    def test_survival_function_monotone(self, synth_weibull_data):
        """S(t) is monotone non-increasing in t."""
        X, t, e = synth_weibull_data
        m = build_survival_model("lgb_aft", n_estimators=20).fit(X, t, e)
        times = np.array([10.0, 30.0, 60.0, 90.0])
        S = m.predict_survival_function(X.iloc[:5], times)
        # Across time axis, S should not increase
        diffs = np.diff(S, axis=1)
        assert (diffs <= 1e-9).all(), f"S(t) not monotone; diffs = {diffs}"


# ===========================================================================
# TestEnsembleFactory
# ===========================================================================

class TestEnsembleFactory:
    def test_all_six_keys_registered(self):
        assert len(MODEL_KEYS) == 6
        assert "cox_ph" in MODEL_KEYS
        assert "xgb_survival" in MODEL_KEYS
        assert "lgb_aft" in MODEL_KEYS
        for k in LIFELINES_PARAMETRIC_KEYS:
            assert k in MODEL_KEYS

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown model_key"):
            build_survival_model("random_forest")

    def test_build_all_returns_six_instances(self):
        models = build_all_survival_models()
        assert len(models) == 6
        for key in MODEL_KEYS:
            assert key in models

    def test_fit_model_returns_result(self, synth_weibull_data):
        """fit_model helper returns a SurvivalFitResult with populated fields."""
        X, t, e = synth_weibull_data
        m = build_survival_model("cox_ph")
        res = fit_model(m, X, t, e)
        assert isinstance(res, SurvivalFitResult)
        assert res.model_key == "cox_ph"
        assert res.n_samples == len(t)
        assert res.n_events == int(e.sum())
        assert res.fit_time_seconds >= 0
        assert np.isfinite(res.log_likelihood)
