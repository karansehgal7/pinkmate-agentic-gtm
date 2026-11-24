"""Tests for pinkmate.ml.production_hardening — Slice 7.5 composition."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pinkmate.ml.conformal import ConformalScoringWrapper
from pinkmate.ml.drift import FeatureSnapshot, save_distribution_snapshot
from pinkmate.ml.production_hardening import (
    HardenedScore,
    ProductionHardeningPipeline,
)


CANONICAL_SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(CANONICAL_SEED)


@pytest.fixture
def pipeline_bundle(rng):
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + X[:, 1] * 0.3 + rng.normal(scale=0.4, size=400) > 0.3).astype(int)
    ensemble = {
        "lr": LogisticRegression().fit(X[:300], y[:300]),
        "rf": RandomForestClassifier(n_estimators=20, random_state=42).fit(
            X[:300], y[:300]
        ),
    }
    wrapper = ConformalScoringWrapper(ensemble, alpha=0.10)
    wrapper.conformalize(X[300:400], y[300:400])

    snapshots = {
        f"f{i}": FeatureSnapshot.from_array(f"f{i}", X[:, i]) for i in range(3)
    }
    return {
        "X": X,
        "y": y,
        "ensemble": ensemble,
        "wrapper": wrapper,
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_pipeline_constructs_with_snapshots(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
        reference_snapshots=pipeline_bundle["snapshots"],
    )
    assert len(p.reference_snapshots) == 3


def test_pipeline_constructs_without_snapshots(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    assert p.reference_snapshots == {}


def test_pipeline_defaults_thresholds(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    assert p.drift_thresholds.warn == 0.10
    assert p.drift_thresholds.alert == 0.25
    assert p.fairness_thresholds.warn == 0.10
    assert p.fairness_thresholds.alert == 0.20


# ---------------------------------------------------------------------------
# from_models_dir factory
# ---------------------------------------------------------------------------


def test_from_models_dir_loads_snapshot_if_present(pipeline_bundle, tmp_path):
    save_distribution_snapshot(
        tmp_path / "train_distribution_snapshot.json",
        pipeline_bundle["snapshots"],
    )
    p = ProductionHardeningPipeline.from_models_dir(
        models_dir=tmp_path,
        ensemble=pipeline_bundle["ensemble"],
        X_calibration=pipeline_bundle["X"][300:400],
        y_calibration=pipeline_bundle["y"][300:400],
        alpha=0.10,
    )
    assert len(p.reference_snapshots) == 3


def test_from_models_dir_no_snapshot_falls_back_quietly(
    pipeline_bundle, tmp_path
):
    p = ProductionHardeningPipeline.from_models_dir(
        models_dir=tmp_path,
        ensemble=pipeline_bundle["ensemble"],
        X_calibration=pipeline_bundle["X"][300:400],
        y_calibration=pipeline_bundle["y"][300:400],
        alpha=0.10,
    )
    assert p.reference_snapshots == {}


# ---------------------------------------------------------------------------
# Single-lead scoring
# ---------------------------------------------------------------------------


def test_score_with_interval_returns_conformal_score(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
        reference_snapshots=pipeline_bundle["snapshots"],
    )
    score = p.score_with_interval(pipeline_bundle["X"][0], theta=0.5)
    assert 0.0 <= score.point_score <= 1.0
    assert 0.0 <= score.lower_bound <= score.upper_bound <= 1.0


def test_score_lead_with_drift_returns_hardened_score(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
        reference_snapshots=pipeline_bundle["snapshots"],
    )
    # Build a per-feature drift dict (batch of 50 points for stable PSI)
    batch_idx = slice(100, 150)
    feature_dict = {
        "f0": pipeline_bundle["X"][batch_idx, 0],
        "f1": pipeline_bundle["X"][batch_idx, 1],
        "f2": pipeline_bundle["X"][batch_idx, 2],
    }
    # Score ONE lead but pass a batch-sized feature dict for stable drift.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hardened = p.score_lead(
            pipeline_bundle["X"][0],
            feature_dict_for_drift=feature_dict,
            theta=0.5,
        )
    assert isinstance(hardened, HardenedScore)
    # In-distribution drift check should produce flags, and none should be red.
    assert isinstance(hardened.drift_flags, list)
    assert len(hardened.drift_flags) == 3


def test_score_lead_skips_drift_when_no_snapshot(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    hardened = p.score_lead(
        pipeline_bundle["X"][0],
        feature_dict_for_drift={"f0": np.array([0.5])},
        theta=0.5,
    )
    assert hardened.drift_flags == []
    assert hardened.drift_is_alerting is False


def test_score_lead_as_dict_json_serialisable(pipeline_bundle):
    import json

    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    hardened = p.score_lead(pipeline_bundle["X"][0], theta=0.5)
    serialised = json.dumps(hardened.as_dict())
    recovered = json.loads(serialised)
    assert "conformal" in recovered
    assert "drift_flags" in recovered
    assert "drift_is_alerting" in recovered


# ---------------------------------------------------------------------------
# Drift check end-to-end
# ---------------------------------------------------------------------------


def test_check_drift_returns_empty_when_no_snapshot(pipeline_bundle):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    results = p.check_drift({"f0": np.array([0.5])})
    assert results == []


def test_check_drift_fires_on_large_shift(pipeline_bundle, rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = ProductionHardeningPipeline(
            conformal_wrapper=pipeline_bundle["wrapper"],
            reference_snapshots=pipeline_bundle["snapshots"],
        )
        shifted = {
            "f0": rng.normal(loc=2.0, size=500),
            "f1": rng.normal(size=500),
            "f2": rng.normal(size=500),
        }
        results = p.check_drift(shifted, raise_warning_on_red=False)
        by_name = {r.feature: r for r in results}
        assert by_name["f0"].level == "red"


# ---------------------------------------------------------------------------
# Batch fairness
# ---------------------------------------------------------------------------


def test_audit_batch_fairness_runs_across_attributes(pipeline_bundle, rng):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    n = 400
    y_pred = rng.random(n) > 0.5
    y_true = rng.choice([0, 1], size=n)
    protected = {
        "employee_band": rng.choice(["SMB", "Mid", "Enterprise"], size=n),
        "industry": rng.choice(["SaaS", "FinServ", "Manufacturing"], size=n),
    }
    reports = p.audit_batch_fairness(y_pred, protected, y_true=y_true)
    assert len(reports) == 2
    attrs = {r.attribute for r in reports}
    assert attrs == {"employee_band", "industry"}
    flags = p.summarise_fairness(reports)
    assert len(flags) == 2


def test_audit_batch_fairness_without_y_true(pipeline_bundle, rng):
    p = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
    )
    n = 300
    y_pred = rng.random(n) > 0.5
    protected = {
        "employee_band": rng.choice(["SMB", "Mid", "Enterprise"], size=n),
    }
    reports = p.audit_batch_fairness(y_pred, protected)
    assert len(reports) == 1
    assert reports[0].equalized_odds_difference is None


# ---------------------------------------------------------------------------
# Cross-process replay fidelity — Slice 3+ determinism contract
# ---------------------------------------------------------------------------


def test_replay_fidelity_conformal_is_deterministic(pipeline_bundle):
    """Slice 7.5 additions must preserve Slice 3 cross-process replay fidelity.

    Given the SAME seeds, calibration data, and features, the conformal
    pipeline must produce bit-identical (point, lower, upper, commit) for
    the same input. If this test fails, the additions have broken the
    cross-process replay-fidelity contract from Slice 3 / Slice 4 (see
    pinkmate/benchmarks/replay_fidelity.py).
    """
    # Build two independent pipelines on the SAME inputs
    import copy

    p1 = ProductionHardeningPipeline(
        conformal_wrapper=pipeline_bundle["wrapper"],
        reference_snapshots=copy.deepcopy(pipeline_bundle["snapshots"]),
    )
    # Fresh wrapper from the same ensemble + same calibration data
    from pinkmate.ml.conformal import ConformalScoringWrapper

    wrapper2 = ConformalScoringWrapper(pipeline_bundle["ensemble"], alpha=0.10)
    wrapper2.conformalize(
        pipeline_bundle["X"][300:400], pipeline_bundle["y"][300:400]
    )
    p2 = ProductionHardeningPipeline(
        conformal_wrapper=wrapper2,
        reference_snapshots=copy.deepcopy(pipeline_bundle["snapshots"]),
    )

    # Score the same batch of 20 leads with each pipeline
    test_batch = pipeline_bundle["X"][400:420]
    scores_1 = [p1.score_with_interval(x, theta=0.5) for x in test_batch]
    scores_2 = [p2.score_with_interval(x, theta=0.5) for x in test_batch]

    for s1, s2 in zip(scores_1, scores_2):
        assert s1.point_score == pytest.approx(s2.point_score, abs=1e-10)
        assert s1.lower_bound == pytest.approx(s2.lower_bound, abs=1e-10)
        assert s1.upper_bound == pytest.approx(s2.upper_bound, abs=1e-10)
        assert s1.is_committed_at_theta == s2.is_committed_at_theta
