"""
End-to-end Slice 2 integration tests on synthetic Taiwan-shaped data.

Runs the full pipeline (preprocess → SMOTE → six-model train → classification
metrics → σ_SHAP measurement) and asserts:
    - Pipeline completes without error on synthetic data (hermetic test).
    - Qualitative findings direction matches paper §VI.C:
        * SMOTE INCREASES σ_SHAP vs no-SMOTE on XGBoost
        * LightGBM σ_SHAP < XGBoost σ_SHAP (both with SMOTE)
    - All models produce valid classification metrics.
    - Trained artefacts can be persisted and reloaded.

Numerical σ_SHAP values are not asserted against Table II here — the
synthetic data does not share Taiwan's underlying distribution, so the
exact numbers legitimately differ. The paper's Table II values are
reproduced only against real UCI 572 data (see scripts/reproduce_table2.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from pinkmate.ml.pipeline import run_slice_two_pipeline


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("slice2_cache")
    art = run_slice_two_pipeline(
        source="synthetic",
        seed=42,
        k_backgrounds=5,
        background_size=50,
        n_reference=50,
        include_xgb_no_smote_baseline=True,
        cache_dir=tmp_dir,
    )
    return art


class TestSliceTwoPipelineRuns:
    def test_source_recorded(self, artefacts):
        assert artefacts.source == "synthetic"

    def test_six_models_trained(self, artefacts):
        from pinkmate.ml.model_zoo import MODEL_NAMES
        assert set(artefacts.zoo.models.keys()) == set(MODEL_NAMES)

    def test_classification_metrics_populated(self, artefacts):
        assert len(artefacts.metrics_with_smote) == 6
        for m in artefacts.metrics_with_smote:
            d = m.as_dict()
            assert 0.0 <= d["precision"] <= 1.0
            assert 0.0 <= d["recall"] <= 1.0
            assert 0.0 <= d["f1"] <= 1.0
            assert 0.0 <= d["roc_auc"] <= 1.0


class TestQualitativeFindings:
    """
    Paper §VI.C qualitative observations — these should hold on any
    sufficiently imbalanced dataset where SMOTE increases minority support,
    not just Taiwan's exact numbers.
    """

    def test_smote_increases_xgboost_sigma_shap(self, artefacts):
        assert artefacts.xgb_no_smote_attribution is not None
        xgb_no_smote = artefacts.xgb_no_smote_attribution.sigma_shap
        xgb_smote = artefacts.attributions_with_smote["xgboost"].sigma_shap
        assert xgb_smote > xgb_no_smote, (
            f"Paper §VI.C Observation 1 (SMOTE INCREASES σ_SHAP) must hold: "
            f"xgb_no_smote={xgb_no_smote:.4f}, xgb_smote={xgb_smote:.4f}"
        )

    def test_lightgbm_more_stable_than_xgboost_with_smote(self, artefacts):
        """
        Paper §VI.C Observation 2: at comparable recall, LightGBM exhibits
        lower attribution variance than XGBoost under the same SMOTE training.
        """
        xgb_smote = artefacts.attributions_with_smote["xgboost"].sigma_shap
        lgb_smote = artefacts.attributions_with_smote["lightgbm"].sigma_shap
        assert lgb_smote < xgb_smote, (
            f"Paper §VI.C Observation 2 (LightGBM MORE stable than XGBoost under SMOTE) "
            f"must hold: lgb_smote={lgb_smote:.4f}, xgb_smote={xgb_smote:.4f}"
        )


class TestArtefactPersistence:
    def test_save_creates_pickled_models_and_summaries(self, artefacts, tmp_path):
        models_dir = tmp_path / "models_out"
        artefacts.save(models_dir)

        # Each model pickled individually.
        for name in artefacts.zoo.models.keys():
            assert (models_dir / f"{name}.pkl").exists(), f"{name}.pkl not persisted"

        # Summaries written.
        assert (models_dir / "metrics.json").exists()
        assert (models_dir / "attribution.json").exists()

    def test_pickled_model_round_trip(self, artefacts, tmp_path):
        """Can we reload a pickled model and get consistent predictions?"""
        models_dir = tmp_path / "models_rt"
        artefacts.save(models_dir)

        with open(models_dir / "xgboost.pkl", "rb") as f:
            reloaded = pickle.load(f)

        import numpy as np
        X_sample = artefacts.split.X_test[:5]
        original_proba = artefacts.zoo.models["xgboost"].predict_proba(X_sample)
        reloaded_proba = reloaded.predict_proba(X_sample)
        np.testing.assert_allclose(original_proba, reloaded_proba, atol=1e-6)
