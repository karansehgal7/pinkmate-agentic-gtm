"""
Slice 5.1 — X Education training pipeline integration tests.

Scope and non-scope:
    This suite verifies the Slice 5.1 training pipeline runs end-to-end
    on the synthetic X Education-shaped fixture AND enforces the
    methodology-integrity AUC guard.

    Methodology-integrity guard:
        All model AUCs < 0.95 after leakage columns are excluded. If any
        model ≥ 0.95, a leakage column likely slipped through and the
        test fails LOUDLY pointing at the loader's dropped-columns list.

    Qualitative-direction tests:
        Model-stability ordering (LightGBM σ_SHAP < XGBoost σ_SHAP) is
        architectural, dataset-agnostic — asserted PASS.

        SMOTE direction is methodologically EXPECTED to weaken at X
        Education's 38.5% minority. We do NOT assert direction here —
        we only assert the measurements are well-formed (bounded, no
        NaN). A DRIFT is a legitimate scientific outcome reported by
        the training script's verdict, not a test failure.

    Not asserted:
        - Magnitude of σ_SHAP (would require real Zenodo data, not synthetic)
        - Specific classification metric values (brittle on synthetic data)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper: load train_xeducation_ensemble.py as a module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xedu_module():
    """Import train_xeducation_ensemble.py for direct function access."""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "train_xeducation_ensemble_mod",
        repo_root / "scripts" / "train_xeducation_ensemble.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_xeducation_ensemble_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def artefacts(xedu_module):
    """
    Run the X Education pipeline once on synthetic data with K=10 for
    stable qualitative-direction tests. K=10 noise floor rationale is
    documented in test_gtm_training.py.
    """
    art = xedu_module.run_xeducation_pipeline(
        source="synthetic",
        seed=42,
        k_backgrounds=10,
        background_size=50,
        n_reference=50,
        include_xgb_no_smote_baseline=True,
        skip_attribution=False,
    )
    return art


# ---------------------------------------------------------------------------
# Pipeline completeness
# ---------------------------------------------------------------------------

class TestPipelineCompleteness:
    def test_runs_end_to_end(self, artefacts):
        assert artefacts is not None

    def test_six_models_trained(self, xedu_module, artefacts):
        from pinkmate.ml.model_zoo import MODEL_NAMES
        assert set(artefacts.zoo.models.keys()) == set(MODEL_NAMES)

    def test_source_recorded_as_synthetic(self, artefacts):
        assert artefacts.source == "synthetic"

    def test_minority_rate_near_target(self, artefacts):
        """Minority rate should be in the 37-40% band (around 38.5%)."""
        assert 0.37 <= artefacts.minority_rate <= 0.40


# ---------------------------------------------------------------------------
# Classification metrics — methodology-integrity AUC guard
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def test_metrics_populated_for_all_models(self, artefacts):
        from pinkmate.ml.model_zoo import MODEL_NAMES
        model_names = {m.model_name for m in artefacts.metrics_with_smote}
        assert model_names == set(MODEL_NAMES)

    def test_all_metrics_bounded_in_unit_interval(self, artefacts):
        for m in artefacts.metrics_with_smote:
            d = m.as_dict()
            assert 0.0 <= d["precision"] <= 1.0
            assert 0.0 <= d["recall"] <= 1.0
            assert 0.0 <= d["f1"] <= 1.0
            assert 0.0 <= d["roc_auc"] <= 1.0

    def test_no_nan_in_metrics(self, artefacts):
        import math
        for m in artefacts.metrics_with_smote:
            d = m.as_dict()
            assert not math.isnan(d["precision"])
            assert not math.isnan(d["recall"])
            assert not math.isnan(d["f1"])
            assert not math.isnan(d["roc_auc"])

    def test_methodology_integrity_auc_guard(self, artefacts):
        """
        THE methodology-integrity test. If any model achieves ROC-AUC
        >= 0.95, we likely haven't excluded a leakage column — the
        synthetic fixture is engineered to produce honest AUCs in the
        0.70-0.85 range.

        A failure here points at the X Education loader's LEAKAGE_COLUMNS
        list — something leaky slipped through and inflated the metric.
        """
        for m in artefacts.metrics_with_smote:
            d = m.as_dict()
            assert d["roc_auc"] < 0.95, (
                f"Methodology-integrity failure: {d['model']} ROC-AUC = "
                f"{d['roc_auc']:.4f} >= 0.95. This almost always means a "
                f"leakage column slipped through. Review "
                f"pinkmate/ml/x_education_loader.py LEAKAGE_COLUMNS and "
                f"NEAR_ZERO_VARIANCE_COLUMNS."
            )


# ---------------------------------------------------------------------------
# σ_SHAP qualitative directions
# ---------------------------------------------------------------------------

class TestQualitativeDirection:
    """
    See module-level docstring — model-stability ordering is architectural
    and asserted PASS; SMOTE direction is NOT asserted because it
    methodologically may DRIFT on X Education's 38.5% minority and the
    verdict framing documents DRIFT as a nuanced finding, not a failure.
    """

    def test_lightgbm_more_stable_than_xgboost_with_smote(self, artefacts):
        """
        Paper §VI.C Observation 2 — LightGBM exhibits lower σ_SHAP than
        XGBoost under SMOTE. Architectural property, should hold regardless
        of imbalance or domain.
        """
        xgb_s = artefacts.attributions_with_smote["xgboost"].sigma_shap
        lgb_s = artefacts.attributions_with_smote["lightgbm"].sigma_shap
        assert lgb_s < xgb_s, (
            f"Paper §VI.C Observation 2 (LightGBM MORE stable than XGBoost "
            f"under SMOTE) failed to transfer to synthetic X Education: "
            f"lgb_smote={lgb_s:.4f}, xgb_smote={xgb_s:.4f}. This is an "
            f"architectural property and should hold — investigate "
            f"hyperparameters."
        )

    def test_smote_direction_recorded_and_bounded(self, artefacts):
        """
        SMOTE-direction MEASUREMENT must be well-formed. We do NOT assert
        the direction itself — on X Education at 38.5% minority a DRIFT
        is a legitimate methodological-boundary finding, not a failure.
        """
        assert artefacts.xgb_no_smote_attribution is not None
        xgb_no = artefacts.xgb_no_smote_attribution.sigma_shap
        xgb_yes = artefacts.attributions_with_smote["xgboost"].sigma_shap
        # Both measurements must be in [0, 1] and non-NaN
        assert 0.0 < xgb_no < 1.0, f"xgb_no_smote σ_SHAP out of band: {xgb_no}"
        assert 0.0 < xgb_yes < 1.0, f"xgb_smote σ_SHAP out of band: {xgb_yes}"


# ---------------------------------------------------------------------------
# Artefact persistence
# ---------------------------------------------------------------------------

class TestArtefactPersistence:
    def test_save_creates_pickles_and_summaries(self, artefacts, tmp_path):
        models_dir = tmp_path / "xedu_models"
        artefacts.save(models_dir)

        from pinkmate.ml.model_zoo import MODEL_NAMES
        for name in MODEL_NAMES:
            assert (models_dir / f"{name}.pkl").exists(), f"{name}.pkl missing"

        assert (models_dir / "metrics.json").exists()
        assert (models_dir / "attribution.json").exists()

    def test_attribution_json_structure(self, artefacts, tmp_path):
        models_dir = tmp_path / "xedu_models_attr"
        artefacts.save(models_dir)

        with open(models_dir / "attribution.json") as f:
            payload = json.load(f)

        assert payload["dataset"] == "x_education"
        assert "source" in payload
        assert "minority_rate" in payload
        assert "with_smote" in payload
        assert "xgboost_no_smote_baseline" in payload


# ---------------------------------------------------------------------------
# Verdict computation (string-level)
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_verdict_has_both_direction_keys(self, xedu_module, artefacts):
        v = xedu_module.compute_verdict(artefacts)
        assert "smote_direction" in v
        assert "model_stability_ordering" in v

    def test_verdict_strings_include_pass_drift_or_not_run(
        self, xedu_module, artefacts,
    ):
        v = xedu_module.compute_verdict(artefacts)
        for key, val in v.items():
            assert any(val.startswith(prefix) for prefix in ("PASS", "DRIFT", "NOT_RUN")), (
                f"Unexpected verdict format for {key}: {val}"
            )
