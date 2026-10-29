"""
Six-model ensemble framework, per paper §III.D and §VI.A.

Models (in paper evaluation order):
    1. LogisticRegression
    2. RandomForestClassifier
    3. AdaBoostClassifier
    4. XGBoostClassifier
    5. LightGBMClassifier
    6. CatBoostClassifier

Protocol per §VI.A:
    - Fixed random_state (default 42) across all models
    - Identical pipeline configuration
    - Trained on SMOTE-oversampled training partition (§III.C)
    - Evaluated on the stratified held-out test partition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES: Tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "adaboost",
    "xgboost",
    "lightgbm",
    "catboost",
)


# ---------------------------------------------------------------------------
# Lazy imports — the heavy ensembles are imported at build time so the
# ml package module import stays cheap for environments that only need
# the formal classes.
# ---------------------------------------------------------------------------

def _import_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier


def _import_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier


def _import_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model_zoo(
    seed: int = 42,
    n_estimators_boosting: int = 100,
) -> Dict[str, object]:
    """
    Construct the six-model zoo with identical seed and comparable
    hyper-parameter choices across ensembles.

    Args:
        seed: Random state propagated to every model. Paper default: 42.
        n_estimators_boosting: Number of boosting iterations for the
            gradient-boosting ensembles. Default 100 matches the companion
            reproducibility artefact for the measured Table I/II values.

    Returns:
        Dict mapping MODEL_NAMES to configured (untrained) estimators.
    """
    XGBClassifier = _import_xgboost()
    LGBMClassifier = _import_lightgbm()
    CatBoostClassifier = _import_catboost()

    zoo: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            random_state=seed,
            max_iter=1000,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            random_state=seed,
            n_estimators=n_estimators_boosting,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(
            random_state=seed,
            n_estimators=n_estimators_boosting,
        ),
        "xgboost": XGBClassifier(
            random_state=seed,
            n_estimators=n_estimators_boosting,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            random_state=seed,
            n_estimators=n_estimators_boosting,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            random_seed=seed,
            iterations=n_estimators_boosting,
            verbose=False,
            allow_writing_files=False,
        ),
    }
    return zoo


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainedZoo:
    """Container for the six trained estimators."""

    models: Dict[str, object]
    seed: int

    def predict_proba_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return per-model class-1 probability on X."""
        out: Dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            # Column 1 is always the positive (minority / bankrupt) class
            # by sklearn convention — fit() orders classes_ ascending.
            out[name] = proba[:, 1]
        return out


def train_zoo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_estimators_boosting: int = 100,
) -> TrainedZoo:
    """
    Train the six-model zoo on a (SMOTE-oversampled) training partition.

    Args:
        X_train: Training features (post-SMOTE if imbalance mitigation applied).
        y_train: Training target.
        seed: Random seed.
        n_estimators_boosting: Boosting iteration count.

    Returns:
        TrainedZoo with every estimator fitted.
    """
    zoo = build_model_zoo(seed=seed, n_estimators_boosting=n_estimators_boosting)
    for name, model in zoo.items():
        model.fit(X_train, y_train)
    return TrainedZoo(models=zoo, seed=seed)


def predict_proba_zoo(zoo: TrainedZoo, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience wrapper around :meth:`TrainedZoo.predict_proba_all`."""
    return zoo.predict_proba_all(X)
