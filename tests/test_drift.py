"""Tests for pinkmate.ml.drift — Slice 7.5b."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from pinkmate.ml.drift import (
    DriftThresholds,
    FeatureDriftResult,
    FeatureSnapshot,
    compute_feature_drift,
    compute_kl_divergence,
    compute_ks_test,
    compute_psi,
    load_distribution_snapshot,
    psi_sentinel,
    save_distribution_snapshot,
    summarise_drift_flags,
)


CANONICAL_SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(CANONICAL_SEED)


# ---------------------------------------------------------------------------
# PSI correctness
# ---------------------------------------------------------------------------


def test_psi_zero_for_identical_distributions(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    psi = compute_psi(ref, cur)
    assert psi < 0.05  # sampling noise on 5000 vs 5000


def test_psi_grows_with_mean_shift(rng):
    ref = rng.normal(size=5000)
    psi_small = compute_psi(ref, rng.normal(loc=0.3, size=5000))
    psi_large = compute_psi(ref, rng.normal(loc=1.5, size=5000))
    assert psi_small < psi_large
    assert psi_large > 0.25  # triggers red


def test_psi_symmetric_direction_is_both_positive(rng):
    """PSI is positive in both directions (not symmetric in value, but both>0)."""
    ref = rng.normal(size=5000)
    cur = rng.normal(loc=0.5, size=5000)
    assert compute_psi(ref, cur) > 0
    assert compute_psi(cur, ref) > 0


def test_psi_handles_binary_feature(rng):
    """Binary features should produce a finite PSI, not NaN/inf."""
    ref = rng.integers(low=0, high=2, size=5000)
    cur = rng.integers(low=0, high=2, size=5000)
    psi = compute_psi(ref, cur)
    assert np.isfinite(psi)


def test_psi_rejects_invalid_n_bins(rng):
    ref = rng.normal(size=100)
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        compute_psi(ref, ref, n_bins=1)


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-NaN value"):
        compute_psi(np.array([]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def test_kl_zero_for_identical_distributions(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    kl = compute_kl_divergence(ref, cur)
    assert kl < 0.05


def test_kl_grows_with_shift(rng):
    ref = rng.normal(size=5000)
    kl_small = compute_kl_divergence(ref, rng.normal(loc=0.3, size=5000))
    kl_large = compute_kl_divergence(ref, rng.normal(loc=1.5, size=5000))
    assert kl_small < kl_large


def test_kl_is_asymmetric(rng):
    """KL(P||Q) != KL(Q||P) for shifted distributions."""
    ref = rng.normal(size=5000)
    cur = rng.normal(loc=1.0, size=5000)
    kl_pq = compute_kl_divergence(ref, cur)
    kl_qp = compute_kl_divergence(cur, ref)
    # They are both positive but typically unequal
    assert abs(kl_pq - kl_qp) > 1e-4


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------


def test_ks_test_finds_no_shift(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    stat, pval = compute_ks_test(ref, cur)
    assert pval > 0.01  # no shift -> non-tiny p-value


def test_ks_test_finds_mean_shift(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(loc=1.0, size=5000)
    stat, pval = compute_ks_test(ref, cur)
    assert pval < 0.001
    assert stat > 0.2


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_thresholds_classify():
    th = DriftThresholds()
    assert th.classify(0.00) == "green"
    assert th.classify(0.05) == "green"
    assert th.classify(0.10) == "green"  # inclusive at warn
    assert th.classify(0.15) == "yellow"
    assert th.classify(0.25) == "yellow"  # inclusive at alert
    assert th.classify(0.30) == "red"


def test_thresholds_rejects_invalid():
    with pytest.raises(ValueError, match="warn.*alert"):
        DriftThresholds(warn=0.3, alert=0.1)


def test_psi_sentinel_raises_on_red(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(loc=2.0, size=5000)
    psi = compute_psi(ref, cur)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        level = psi_sentinel(psi, "test_feat")
        assert level == "red"
        assert any("Significant feature drift" in str(x.message) for x in w)


def test_psi_sentinel_quiet_on_green(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    psi = compute_psi(ref, cur)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        level = psi_sentinel(psi, "test_feat")
        assert level == "green"
        assert len([x for x in w if "Significant feature drift" in str(x.message)]) == 0


def test_psi_sentinel_can_suppress_warning(rng):
    ref = rng.normal(size=5000)
    cur = rng.normal(loc=2.0, size=5000)
    psi = compute_psi(ref, cur)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        level = psi_sentinel(psi, "test_feat", raise_warning_on_red=False)
        assert level == "red"
        assert len([x for x in w if "Significant feature drift" in str(x.message)]) == 0


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------


def test_feature_snapshot_subsamples(rng):
    arr = rng.normal(size=10_000)
    snap = FeatureSnapshot.from_array("revenue", arr, max_sample_size=1000)
    assert snap.n_training == 10_000
    assert len(snap.values) == 1000


def test_feature_snapshot_roundtrip(rng, tmp_path):
    arr = rng.normal(size=5000)
    snap = FeatureSnapshot.from_array("rev", arr)

    path = tmp_path / "snap.json"
    save_distribution_snapshot(path, {"rev": snap})
    loaded = load_distribution_snapshot(path)
    assert "rev" in loaded
    assert loaded["rev"].mean == pytest.approx(snap.mean, abs=1e-6)
    assert loaded["rev"].n_training == snap.n_training
    assert len(loaded["rev"].values) == len(snap.values)


def test_feature_snapshot_rejects_all_nan():
    arr = np.full(100, np.nan)
    with pytest.raises(ValueError, match="no non-NaN values"):
        FeatureSnapshot.from_array("bad", arr)


def test_load_missing_snapshot_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="No training distribution snapshot"):
        load_distribution_snapshot(missing)


def test_load_malformed_snapshot_raises(tmp_path):
    path = tmp_path / "bad.json"
    with open(path, "w") as fh:
        json.dump({"features": "not a dict"}, fh)
    with pytest.raises(ValueError, match="features.*not a dict"):
        load_distribution_snapshot(path)


# ---------------------------------------------------------------------------
# Batch drift computation
# ---------------------------------------------------------------------------


def test_compute_feature_drift_marks_missing(rng):
    snaps = {
        "a": FeatureSnapshot.from_array("a", rng.normal(size=1000)),
        "b": FeatureSnapshot.from_array("b", rng.normal(size=1000)),
    }
    results = compute_feature_drift(snaps, {"a": rng.normal(size=500)})
    by_name = {r.feature: r for r in results}
    assert by_name["a"].level in ("green", "yellow", "red")
    assert by_name["b"].level == "missing"
    assert by_name["b"].psi is None


def test_compute_feature_drift_mixed_levels(rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snaps = {
            "stable": FeatureSnapshot.from_array("stable", rng.normal(size=1000)),
            "shifted": FeatureSnapshot.from_array("shifted", rng.normal(size=1000)),
        }
        batch = {
            "stable": rng.normal(size=500),
            "shifted": rng.normal(loc=2.5, size=500),
        }
        results = compute_feature_drift(snaps, batch)
        by_name = {r.feature: r for r in results}
        assert by_name["stable"].level == "green"
        assert by_name["shifted"].level == "red"
        assert by_name["shifted"].psi > 0.25


def test_summarise_drift_flags(rng):
    results = [
        FeatureDriftResult("a", 0.05, 0.02, 0.01, 0.9, "green"),
        FeatureDriftResult("b", 0.30, 0.15, 0.3, 0.001, "red"),
        FeatureDriftResult("c", None, None, None, None, "missing"),
    ]
    flags = summarise_drift_flags(results)
    assert flags == ["green:a", "missing:c", "red:b"]


def test_drift_result_as_dict_json_serialisable():
    r = FeatureDriftResult("a", 0.05, 0.02, 0.01, 0.9, "green")
    serialised = json.dumps(r.as_dict())
    recovered = json.loads(serialised)
    assert recovered["feature"] == "a"
    assert recovered["psi"] == pytest.approx(0.05)
    assert recovered["level"] == "green"
