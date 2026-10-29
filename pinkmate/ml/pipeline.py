"""
End-to-end training + attribution-measurement pipeline, per paper §VI.A.

Produces:
    - Fitted six-model zoo (pickled to disk under ``models/``)
    - Classification metrics on the held-out test partition (Table I)
    - σ_SHAP attribution-stability measurements on tree models (Table II)

The pipeline is designed to be callable from both the scripts/
reproduction entry point and from the ScoringAgent's lazy initialiser.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

from pinkmate.ml.attribution import (
    DEFAULT_K_ROTATED_BACKGROUNDS,
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_N_REFERENCE_INPUTS,
    AttributionResult,
    compute_rotated_shap_variance,
)
from pinkmate.ml.dataset import (
    PreprocessedSplit,
    TAIWAN_RANDOM_SEED,
    load_taiwan_bankruptcy,
    preprocess,
)
from pinkmate.ml.imbalance import apply_smote
from pinkmate.ml.model_zoo import MODEL_NAMES, TrainedZoo, train_zoo

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric structures
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Per-model classification metrics on the test partition (Table I)."""

    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "model": self.model_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
        }


def classification_metrics(
    zoo: TrainedZoo,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> List[ModelMetrics]:
    """Compute per-model Precision/Recall/F1/ROC-AUC at a decision threshold."""
    proba = zoo.predict_proba_all(X_test)
    out: List[ModelMetrics] = []
    for name in MODEL_NAMES:
        p = proba[name]
        y_pred = (p >= threshold).astype(int)
        out.append(ModelMetrics(
            model_name=name,
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, p)),
        ))
    return out


# ---------------------------------------------------------------------------
# Attribution-stability measurement (Table II)
# ---------------------------------------------------------------------------

TREE_MODELS_FOR_ATTRIBUTION: Tuple[str, ...] = ("xgboost", "lightgbm")
"""
Tree models for which the paper's §V attribution-stability measurement is
reported in Table II. CatBoost and RandomForest are also tree-based and
supported by TreeSHAP, but Table II focuses on XGBoost/LightGBM.
"""


def measure_attribution_stability(
    zoo: TrainedZoo,
    X_train_post_smote: np.ndarray,
    X_test: np.ndarray,
    feature_names: Tuple[str, ...],
    k_backgrounds: int = DEFAULT_K_ROTATED_BACKGROUNDS,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    n_reference: int = DEFAULT_N_REFERENCE_INPUTS,
    seed: int = TAIWAN_RANDOM_SEED,
) -> Dict[str, AttributionResult]:
    """
    Run the §V attribution-stability measurement on tree models for which
    the paper reports σ_SHAP.

    Args:
        zoo: Trained six-model zoo.
        X_train_post_smote: Training features AFTER SMOTE was applied —
            TreeSHAP backgrounds are drawn from the SMOTE-oversampled pool
            per the paper's reproducibility protocol.
        X_test: Test partition (full).
        feature_names: Column names (preserved order).
        k_backgrounds / background_size / n_reference: Paper defaults 50/100/200.
        seed: Base seed for reference-input selection and background rotations.

    Returns:
        Dict mapping each measured model name to its AttributionResult.
    """
    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(len(X_test), size=min(n_reference, len(X_test)), replace=False)
    X_reference = X_test[ref_idx]

    results: Dict[str, AttributionResult] = {}
    for name in TREE_MODELS_FOR_ATTRIBUTION:
        LOGGER.info(
            "Computing σ_SHAP for %s (K=%d backgrounds × size=%d, n_ref=%d)",
            name, k_backgrounds, background_size, len(X_reference),
        )
        results[name] = compute_rotated_shap_variance(
            model=zoo.models[name],
            model_name=name,
            X_train=X_train_post_smote,
            X_reference=X_reference,
            feature_names=feature_names,
            k_backgrounds=k_backgrounds,
            background_size=background_size,
            seed=seed,
        )
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@dataclass
class SliceTwoArtefacts:
    """Container for a full Slice 2 training + measurement run."""

    zoo: TrainedZoo
    metrics_with_smote: List[ModelMetrics]
    xgb_no_smote_attribution: Optional[AttributionResult]
    attributions_with_smote: Dict[str, AttributionResult]
    split: PreprocessedSplit
    smote_X_train: np.ndarray
    smote_y_train: np.ndarray
    source: str
    seed: int

    def save(self, models_dir: Path) -> None:
        """
        Persist the trained models + attribution summary to disk.

        Models land in ``models_dir/<model_name>.pkl``; attribution summary
        in ``models_dir/attribution.json``; metrics summary in
        ``models_dir/metrics.json``.
        """
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.zoo.models.items():
            with open(models_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
        with open(models_dir / "metrics.json", "w") as f:
            json.dump([m.as_dict() for m in self.metrics_with_smote], f, indent=2)

        attribution_payload = {
            "with_smote": {
                name: {
                    "sigma_shap": r.sigma_shap,
                    "mean_absolute_phi": r.mean_absolute_phi,
                    "k_backgrounds": r.k_backgrounds,
                    "background_size": r.background_size,
                    "n_reference": r.n_reference,
                }
                for name, r in self.attributions_with_smote.items()
            },
        }
        if self.xgb_no_smote_attribution is not None:
            attribution_payload["xgboost_no_smote"] = {
                "sigma_shap": self.xgb_no_smote_attribution.sigma_shap,
                "mean_absolute_phi": self.xgb_no_smote_attribution.mean_absolute_phi,
                "k_backgrounds": self.xgb_no_smote_attribution.k_backgrounds,
                "background_size": self.xgb_no_smote_attribution.background_size,
                "n_reference": self.xgb_no_smote_attribution.n_reference,
            }
        with open(models_dir / "attribution.json", "w") as f:
            json.dump(attribution_payload, f, indent=2)

        LOGGER.info("Saved Slice 2 artefacts to %s", models_dir)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_slice_two_pipeline(
    source: str = "uci",
    seed: int = TAIWAN_RANDOM_SEED,
    k_backgrounds: int = DEFAULT_K_ROTATED_BACKGROUNDS,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    n_reference: int = DEFAULT_N_REFERENCE_INPUTS,
    include_xgb_no_smote_baseline: bool = True,
    cache_dir: Optional[Path] = None,
) -> SliceTwoArtefacts:
    """
    Execute the end-to-end Slice 2 pipeline:
        1. Load Taiwan Bankruptcy UCI 572 (or synthetic for test envs)
        2. Preprocess (median impute → standard scale → stratified 80/20 split)
        3. SMOTE on training partition only
        4. Train six-model zoo
        5. Measure classification metrics (Table I)
        6. Measure σ_SHAP for XGBoost+LightGBM with SMOTE
        7. If baseline requested, also measure σ_SHAP for XGBoost WITHOUT SMOTE
           (Table II row 1 baseline)

    Args:
        source: 'uci' (default) fetches real UCI 572; 'synthetic' uses
            :func:`pinkmate.ml.dataset.make_synthetic_taiwan_like` — intended
            for CI / sandboxed environments without UCI access.
        seed: Paper default 42.
        k_backgrounds / background_size / n_reference: Paper defaults 50/100/200.
        include_xgb_no_smote_baseline: Train and measure XGBoost without
            SMOTE for the Table II baseline. Adds ~1 minute to runtime.
        cache_dir: Dataset cache directory.

    Returns:
        SliceTwoArtefacts containing trained zoo, metrics, and attributions.
    """
    LOGGER.info("Starting Slice 2 pipeline (source=%s, seed=%d)", source, seed)

    if source == "uci":
        dataset = load_taiwan_bankruptcy(cache_dir=cache_dir, allow_cache=True)
    elif source == "synthetic":
        from pinkmate.ml.dataset import make_synthetic_taiwan_like
        dataset = make_synthetic_taiwan_like(seed=seed)
    else:
        raise ValueError(f"Unknown source '{source}'; expected 'uci' or 'synthetic'.")

    LOGGER.info(
        "Dataset loaded: source=%s, rows=%d, features=%d, minority_rate=%.4f",
        dataset.source, len(dataset.X), dataset.X.shape[1], dataset.y.mean(),
    )

    split = preprocess(dataset.X, dataset.y, seed=seed)
    LOGGER.info(
        "Preprocessed: train=%d, test=%d, minority_train=%.4f, minority_test=%.4f",
        len(split.X_train), len(split.X_test),
        split.y_train.mean(), split.y_test.mean(),
    )

    X_train_sm, y_train_sm = apply_smote(split.X_train, split.y_train, seed=seed)
    LOGGER.info("After SMOTE: train rows=%d, minority_rate=%.4f",
                len(X_train_sm), y_train_sm.mean())

    zoo = train_zoo(X_train_sm, y_train_sm, seed=seed)
    LOGGER.info("Trained %d models", len(zoo.models))

    metrics = classification_metrics(zoo, split.X_test, split.y_test)

    attributions = measure_attribution_stability(
        zoo=zoo,
        X_train_post_smote=X_train_sm,
        X_test=split.X_test,
        feature_names=split.feature_names,
        k_backgrounds=k_backgrounds,
        background_size=background_size,
        n_reference=n_reference,
        seed=seed,
    )

    xgb_no_smote_attr: Optional[AttributionResult] = None
    if include_xgb_no_smote_baseline:
        LOGGER.info("Training XGBoost baseline WITHOUT SMOTE for Table II row 1 baseline")
        baseline_zoo = train_zoo(split.X_train, split.y_train, seed=seed)
        rng = np.random.default_rng(seed)
        ref_idx = rng.choice(len(split.X_test), size=min(n_reference, len(split.X_test)), replace=False)
        X_reference = split.X_test[ref_idx]
        xgb_no_smote_attr = compute_rotated_shap_variance(
            model=baseline_zoo.models["xgboost"],
            model_name="xgboost_no_smote",
            X_train=split.X_train,
            X_reference=X_reference,
            feature_names=split.feature_names,
            k_backgrounds=k_backgrounds,
            background_size=background_size,
            seed=seed,
        )

    return SliceTwoArtefacts(
        zoo=zoo,
        metrics_with_smote=metrics,
        xgb_no_smote_attribution=xgb_no_smote_attr,
        attributions_with_smote=attributions,
        split=split,
        smote_X_train=X_train_sm,
        smote_y_train=y_train_sm,
        source=dataset.source,
        seed=seed,
    )
