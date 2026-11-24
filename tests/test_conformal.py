"""Tests for pinkmate.ml.conformal — Slice 7.5a."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pinkmate.ml.conformal import (
    ConformalScore,
    ConformalScoringWrapper,
    commit_at_lower_bound,
    interval_is_informative,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CANONICAL_SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(CANONICAL_SEED)


@pytest.fixture
def synthetic_split(rng):
    """A train/calibration/test split with recoverable signal."""
    X = rng.normal(size=(600, 5))
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(scale=0.5, size=600) > 0.5).astype(int)
    return {
        "X_train": X[:400],
        "y_train": y[:400],
        "X_cal": X[400:500],
        "y_cal": y[400:500],
        "X_test": X[500:],
        "y_test": y[500:],
    }


@pytest.fixture
def prefit_ensemble(synthetic_split):
    """A tiny prefit ensemble good enough for 90% coverage."""
    X_train = synthetic_split["X_train"]
    y_train = synthetic_split["y_train"]
    return {
        "lr": LogisticRegression().fit(X_train, y_train),
        "rf": RandomForestClassifier(n_estimators=30, random_state=42).fit(X_train, y_train),
    }


# ---------------------------------------------------------------------------
# Constructor-level tests
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_alpha(prefit_ensemble):
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalScoringWrapper(prefit_ensemble, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalScoringWrapper(prefit_ensemble, alpha=1.0)


def test_constructor_rejects_invalid_class_index(prefit_ensemble):
    with pytest.raises(ValueError, match="class_index must be 0 or 1"):
        ConformalScoringWrapper(prefit_ensemble, class_index=2)


def test_constructor_rejects_empty_ensemble():
    with pytest.raises(ValueError, match="non-empty dict"):
        ConformalScoringWrapper({}, alpha=0.10)


def test_constructor_rejects_non_dict_ensemble():
    with pytest.raises(ValueError, match="non-empty dict"):
        ConformalScoringWrapper([LogisticRegression()], alpha=0.10)


# ---------------------------------------------------------------------------
# Conformalisation tests
# ---------------------------------------------------------------------------


def test_conformalize_rejects_small_calibration(prefit_ensemble, synthetic_split):
    w = ConformalScoringWrapper(prefit_ensemble, alpha=0.10)
    with pytest.raises(ValueError, match="too small"):
        w.conformalize(synthetic_split["X_cal"][:5], synthetic_split["y_cal"][:5])


def test_conformalize_rejects_mismatched_lengths(prefit_ensemble, synthetic_split):
    w = ConformalScoringWrapper(prefit_ensemble, alpha=0.10)
    with pytest.raises(ValueError, match="rows"):
        w.conformalize(
            synthetic_split["X_cal"], synthetic_split["y_cal"][: -5]
        )


def test_conformalize_returns_self(prefit_ensemble, synthetic_split):
    w = ConformalScoringWrapper(prefit_ensemble, alpha=0.10)
    returned = w.conformalize(synthetic_split["X_cal"], synthetic_split["y_cal"])
    assert returned is w
    assert w._is_conformalised


def test_predict_batch_before_conformalise_raises(prefit_ensemble, synthetic_split):
    w = ConformalScoringWrapper(prefit_ensemble, alpha=0.10)
    with pytest.raises(RuntimeError, match="not yet conformalised"):
        w.predict_interval_batch(synthetic_split["X_test"])


# ---------------------------------------------------------------------------
# Prediction and coverage tests
# ---------------------------------------------------------------------------


def _fitted_wrapper(ensemble, split, alpha=0.10):
    w = ConformalScoringWrapper(ensemble, alpha=alpha)
    w.conformalize(split["X_cal"], split["y_cal"])
    return w


def test_predict_interval_batch_shape(prefit_ensemble, synthetic_split):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split)
    pts, los, ups = w.predict_interval_batch(synthetic_split["X_test"])
    n = synthetic_split["X_test"].shape[0]
    assert pts.shape == (n,)
    assert los.shape == (n,)
    assert ups.shape == (n,)


def test_predict_interval_batch_bounds_within_simplex(prefit_ensemble, synthetic_split):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split)
    pts, los, ups = w.predict_interval_batch(synthetic_split["X_test"])
    assert np.all(los >= 0.0) and np.all(los <= 1.0)
    assert np.all(ups >= 0.0) and np.all(ups <= 1.0)
    assert np.all(ups >= los)


def test_coverage_near_target_at_alpha_0_10(prefit_ensemble, synthetic_split):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split, alpha=0.10)
    cov = w.empirical_coverage(synthetic_split["X_test"], synthetic_split["y_test"])
    # Allow a reasonably wide tolerance band given n_test=100
    assert 0.75 <= cov <= 1.0


def test_coverage_scales_with_alpha(prefit_ensemble, synthetic_split):
    # With smaller alpha the interval should widen -> higher coverage
    w_10 = _fitted_wrapper(prefit_ensemble, synthetic_split, alpha=0.10)
    w_05 = _fitted_wrapper(prefit_ensemble, synthetic_split, alpha=0.05)
    cov_10 = w_10.empirical_coverage(
        synthetic_split["X_test"], synthetic_split["y_test"]
    )
    cov_05 = w_05.empirical_coverage(
        synthetic_split["X_test"], synthetic_split["y_test"]
    )
    # Strictly: cov_05 >= cov_10 up to sampling noise
    assert cov_05 + 0.05 >= cov_10


# ---------------------------------------------------------------------------
# Single-lead API
# ---------------------------------------------------------------------------


def test_predict_interval_single_returns_conformal_score(
    prefit_ensemble, synthetic_split
):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split)
    score = w.predict_interval_single(synthetic_split["X_test"][0], theta=0.75)
    assert isinstance(score, ConformalScore)
    assert 0.0 <= score.point_score <= 1.0
    assert 0.0 <= score.lower_bound <= 1.0
    assert 0.0 <= score.upper_bound <= 1.0
    assert score.upper_bound >= score.lower_bound
    assert score.alpha == pytest.approx(0.10)
    assert score.theta == pytest.approx(0.75)


def test_predict_interval_single_rejects_multi_row(prefit_ensemble, synthetic_split):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split)
    with pytest.raises(ValueError, match="exactly one lead"):
        w.predict_interval_single(synthetic_split["X_test"][:3], theta=0.5)


def test_predict_interval_single_rejects_bad_theta(prefit_ensemble, synthetic_split):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split)
    with pytest.raises(ValueError, match="theta must be in"):
        w.predict_interval_single(synthetic_split["X_test"][0], theta=1.5)


def test_commit_rule_is_conservative(prefit_ensemble, synthetic_split):
    """Lower-bound commit should fire strictly less often than point-score commit."""
    w = _fitted_wrapper(prefit_ensemble, synthetic_split, alpha=0.10)
    pts, los, _ups = w.predict_interval_batch(synthetic_split["X_test"])
    theta = 0.5
    conservative_commits = int(np.sum(los >= theta))
    point_commits = int(np.sum(pts >= theta))
    assert conservative_commits <= point_commits


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------


def test_conformal_score_as_dict_json_serialisable():
    import json

    s = ConformalScore(
        point_score=0.72,
        lower_bound=0.55,
        upper_bound=0.88,
        is_committed_at_theta=False,
        alpha=0.10,
        theta=0.75,
    )
    serialised = json.dumps(s.as_dict())
    recovered = json.loads(serialised)
    assert recovered["point_score"] == pytest.approx(0.72)
    assert recovered["interval_width"] == pytest.approx(0.33)
    assert recovered["is_committed_at_theta"] is False


def test_conformal_score_interval_width():
    s = ConformalScore(0.5, 0.2, 0.9, False, 0.10, 0.5)
    assert s.interval_width() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_commit_at_lower_bound_uses_stored_theta_by_default():
    s = ConformalScore(0.7, 0.60, 0.8, True, 0.10, 0.55)
    assert commit_at_lower_bound(s) is True

    s2 = ConformalScore(0.5, 0.50, 0.6, False, 0.10, 0.60)
    assert commit_at_lower_bound(s2) is False


def test_commit_at_lower_bound_override_theta():
    s = ConformalScore(0.7, 0.60, 0.8, True, 0.10, 0.55)
    assert commit_at_lower_bound(s, theta=0.7) is False
    assert commit_at_lower_bound(s, theta=0.5) is True


def test_commit_at_lower_bound_rejects_bad_theta():
    s = ConformalScore(0.7, 0.60, 0.8, True, 0.10, 0.55)
    with pytest.raises(ValueError, match="theta must be in"):
        commit_at_lower_bound(s, theta=1.5)


def test_interval_is_informative_true_for_narrow_interval():
    s = ConformalScore(0.7, 0.65, 0.75, True, 0.10, 0.5)
    assert interval_is_informative(s, max_width=0.5) is True


def test_interval_is_informative_false_for_wide_interval():
    s = ConformalScore(0.5, 0.0, 1.0, False, 0.10, 0.5)
    assert interval_is_informative(s, max_width=0.5) is False


def test_interval_is_informative_rejects_bad_max_width():
    s = ConformalScore(0.5, 0.3, 0.7, False, 0.10, 0.5)
    with pytest.raises(ValueError, match="max_width must be"):
        interval_is_informative(s, max_width=1.5)


# ---------------------------------------------------------------------------
# Width-summary diagnostic
# ---------------------------------------------------------------------------


def test_interval_width_summary_structure(prefit_ensemble, synthetic_split):
    w = _fitted_wrapper(prefit_ensemble, synthetic_split)
    summary = w.interval_width_summary(synthetic_split["X_test"])
    assert set(summary.keys()) == {"mean", "median", "min", "max", "std"}
    assert 0.0 <= summary["min"] <= summary["mean"] <= summary["max"] <= 1.0
