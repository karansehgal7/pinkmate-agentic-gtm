"""Tests for ``scripts/measure_conformal_coverage_cross_domain.py``.

Pre-camera-ready micro-slice measurement script correctness contract.

These tests use synthetic fixtures — they verify the script's STRUCTURE
and MEASUREMENT CORRECTNESS without needing real trained models or
calibration data. The actual numerical paper table is produced by
running the script locally against real trained models.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Dynamic import of the script module
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_conformal_coverage_cross_domain.py"


@pytest.fixture(scope="module")
def script_module():
    """Load the script as a module for direct function testing."""
    spec = importlib.util.spec_from_file_location(
        "measure_conformal_coverage_cross_domain", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_conformal_coverage_cross_domain"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


CANONICAL_SEED: int = 42


@pytest.fixture
def rng():
    return np.random.default_rng(CANONICAL_SEED)


@pytest.fixture
def synthetic_dataset_dir(rng, tmp_path):
    """Build a complete models/<key>/ directory with pickles + calibration.npz."""
    dataset_dir = tmp_path / "models" / "synth"
    dataset_dir.mkdir(parents=True)

    # Generate separable synthetic data
    n_train, n_cal = 400, 200
    n_total = n_train + n_cal
    X = rng.normal(size=(n_total, 5))
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(scale=0.4, size=n_total) > 0.3).astype(int)

    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:], y[n_train:]

    # Train two tiny models (we only need a couple to exercise the ensemble path)
    models = {
        "logistic_regression": LogisticRegression().fit(X_train, y_train),
        "random_forest": RandomForestClassifier(
            n_estimators=15, random_state=CANONICAL_SEED
        ).fit(X_train, y_train),
    }
    for name, model in models.items():
        with open(dataset_dir / f"{name}.pkl", "wb") as fh:
            pickle.dump(model, fh)

    # Write calibration.npz in the Slice-7.5 training-script format
    np.savez(
        dataset_dir / "calibration.npz",
        X_calibration=X_cal,
        y_calibration=y_cal,
        feature_names=np.array([f"f{i}" for i in range(5)], dtype=object),
    )
    return dataset_dir


@pytest.fixture
def minimal_dataset_cfg(synthetic_dataset_dir):
    return {
        "key": "synth",
        "label": "Synthetic",
        "models_dir": str(synthetic_dataset_dir.relative_to(synthetic_dataset_dir.parent.parent)),
        "minority_rate_pct": 25.0,
        "domain": "Synthetic test",
        "source": "pytest fixture",
        "slice_shipped": "pre-camera-ready test",
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def test_load_ensemble_returns_dict_of_models(script_module, synthetic_dataset_dir):
    ensemble = script_module.load_ensemble(synthetic_dataset_dir)
    assert set(ensemble.keys()) == {"logistic_regression", "random_forest"}
    for model in ensemble.values():
        assert hasattr(model, "predict_proba")


def test_load_ensemble_empty_when_models_dir_empty(script_module, tmp_path):
    empty_dir = tmp_path / "models" / "empty"
    empty_dir.mkdir(parents=True)
    assert script_module.load_ensemble(empty_dir) == {}


def test_load_calibration_returns_tuple(script_module, synthetic_dataset_dir):
    pair = script_module.load_calibration(synthetic_dataset_dir / "calibration.npz")
    assert pair is not None
    X, y = pair
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]


def test_load_calibration_returns_none_when_missing(script_module, tmp_path):
    assert script_module.load_calibration(tmp_path / "no_such_file.npz") is None


# ---------------------------------------------------------------------------
# Split + bootstrap
# ---------------------------------------------------------------------------


def test_split_calibration_final_eval_proportions(script_module, rng):
    X = rng.normal(size=(100, 3))
    y = rng.integers(0, 2, size=100)
    X_c, y_c, X_e, y_e = script_module.split_calibration_final_eval(
        X, y, frac_calibration=0.8, seed=CANONICAL_SEED
    )
    assert X_c.shape[0] == 80
    assert X_e.shape[0] == 20
    assert y_c.shape[0] == 80
    assert y_e.shape[0] == 20


def test_split_is_deterministic_under_same_seed(script_module, rng):
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, size=50)
    Xc1, _, Xe1, _ = script_module.split_calibration_final_eval(X, y, seed=7)
    Xc2, _, Xe2, _ = script_module.split_calibration_final_eval(X, y, seed=7)
    np.testing.assert_array_equal(Xc1, Xc2)
    np.testing.assert_array_equal(Xe1, Xe2)


def test_bootstrap_ci_contains_point_estimate(script_module, rng):
    # On all-True array, coverage is exactly 1.0; CI should bracket it
    in_interval = np.ones(100, dtype=bool)
    lo, hi = script_module.bootstrap_coverage_ci(
        in_interval, n_bootstrap=100, confidence=0.95, seed=CANONICAL_SEED
    )
    assert lo == pytest.approx(1.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_ci_brackets_empirical_coverage(script_module, rng):
    # Coverage ~0.9 on 500 trials — CI should bracket in most cases
    in_interval = rng.random(500) < 0.9
    point = float(np.mean(in_interval))
    lo, hi = script_module.bootstrap_coverage_ci(
        in_interval, n_bootstrap=200, seed=CANONICAL_SEED
    )
    assert lo <= point <= hi


def test_bootstrap_ci_handles_empty_input(script_module):
    lo, hi = script_module.bootstrap_coverage_ci(
        np.array([], dtype=bool), n_bootstrap=50, seed=CANONICAL_SEED
    )
    assert lo == 0.0
    assert hi == 0.0


# ---------------------------------------------------------------------------
# End-to-end per-dataset measurement
# ---------------------------------------------------------------------------


def test_measure_dataset_coverage_runs_on_synthetic(
    script_module, synthetic_dataset_dir, tmp_path
):
    cfg = {
        "key": "synth",
        "label": "Synthetic",
        "models_dir": str(synthetic_dataset_dir.relative_to(tmp_path)),
        "minority_rate_pct": 25.0,
        "domain": "Synthetic test",
        "source": "pytest fixture",
        "slice_shipped": "pre-camera-ready test",
    }
    result = script_module.measure_dataset_coverage(
        dataset_cfg=cfg,
        repo_root=tmp_path,
        alpha=0.10,
        n_bootstrap=50,
        tolerance=0.10,
        theta=0.75,
        seed=CANONICAL_SEED,
    )
    assert result.status == "ok", result.note
    assert result.empirical_coverage is not None
    assert 0.0 <= result.empirical_coverage <= 1.0
    assert result.coverage_ci_low <= result.empirical_coverage <= result.coverage_ci_high
    assert result.conservative_commit_rate <= result.point_commit_rate


def test_measure_dataset_coverage_skipped_when_models_missing(
    script_module, tmp_path
):
    cfg = {
        "key": "missing",
        "label": "Missing",
        "models_dir": "models/missing",
        "minority_rate_pct": 10.0,
        "domain": "none",
        "source": "none",
        "slice_shipped": "none",
    }
    result = script_module.measure_dataset_coverage(
        dataset_cfg=cfg,
        repo_root=tmp_path,
        alpha=0.10,
        n_bootstrap=10,
        tolerance=0.05,
        theta=0.75,
        seed=CANONICAL_SEED,
    )
    assert result.status == "skipped"
    assert result.empirical_coverage is None
    assert "no model pickles" in result.note.lower()


def test_measure_dataset_coverage_skipped_when_calibration_missing(
    script_module, tmp_path
):
    # Put pickles but no calibration.npz
    dataset_dir = tmp_path / "models" / "no_cal"
    dataset_dir.mkdir(parents=True)
    with open(dataset_dir / "logistic_regression.pkl", "wb") as fh:
        pickle.dump(LogisticRegression().fit([[0, 0], [1, 1]], [0, 1]), fh)
    cfg = {
        "key": "no_cal",
        "label": "NoCal",
        "models_dir": "models/no_cal",
        "minority_rate_pct": 10.0,
        "domain": "none",
        "source": "none",
        "slice_shipped": "none",
    }
    result = script_module.measure_dataset_coverage(
        dataset_cfg=cfg,
        repo_root=tmp_path,
        alpha=0.10,
        n_bootstrap=10,
        tolerance=0.05,
        theta=0.75,
        seed=CANONICAL_SEED,
    )
    assert result.status == "skipped"
    assert "no calibration.npz" in result.note.lower()


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_report_renders_all_expected_sections(script_module):
    DCR = script_module.DatasetCoverageResult
    results = [
        DCR(
            key="synth",
            label="Synthetic",
            minority_rate_pct=25.0,
            domain="synthetic",
            source="fixture",
            slice_shipped="n/a",
            status="ok",
            n_calibration=80,
            n_final_eval=20,
            empirical_coverage=0.90,
            coverage_ci_low=0.85,
            coverage_ci_high=0.95,
            mean_interval_width=0.40,
            median_interval_width=0.38,
            point_commit_rate=0.30,
            conservative_commit_rate=0.15,
            within_tolerance=True,
        ),
    ]
    report = script_module.render_markdown_report(
        results, alpha=0.10, n_bootstrap=200, tolerance=0.05,
        theta=0.75, seed=42,
    )
    assert "# Conformal coverage across four cross-domain datasets" in report
    assert "## Protocol" in report
    assert "## Primary — empirical coverage per dataset" in report
    assert "## Secondary — interval widths" in report
    assert "## Tertiary — commit rates" in report
    assert "## Reproducibility" in report
    # Table values should be present
    assert "Synthetic" in report
    assert "0.900" in report


def test_report_includes_skipped_notes_when_present(script_module):
    DCR = script_module.DatasetCoverageResult
    results = [
        DCR(
            key="missing",
            label="Missing",
            minority_rate_pct=10.0,
            domain="n/a",
            source="n/a",
            slice_shipped="n/a",
            status="skipped",
            note="no calibration.npz in models/missing",
        ),
    ]
    report = script_module.render_markdown_report(
        results, alpha=0.10, n_bootstrap=200, tolerance=0.05,
        theta=0.75, seed=42,
    )
    assert "## Notes on skipped / errored datasets" in report
    assert "Missing" in report
    assert "no calibration.npz" in report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def test_main_writes_report_file_when_no_datasets_available(
    script_module, tmp_path, monkeypatch
):
    # Run main against an empty repo_root — no datasets available, but the
    # script should still write a well-formed report (with all datasets
    # marked skipped) and exit cleanly with code 0.
    report_path = tmp_path / "docs" / "conformal_coverage_report.md"
    exit_code = script_module.main(
        [
            "--repo-root", str(tmp_path),
            "--report-path", "docs/conformal_coverage_report.md",
            "--n-bootstrap", "20",
        ]
    )
    assert exit_code == 0
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "## Primary" in content
    # All four real datasets should be marked skipped
    assert content.count("skipped") >= 4
