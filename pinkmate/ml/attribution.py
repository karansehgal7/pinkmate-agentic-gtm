"""
TreeSHAP attribution + σ_SHAP measurement, per paper §V.

σ_SHAP definition (§V, equation):

    σ_SHAP = sqrt( (1/m) · Σ_{j=1}^{m} Var_k( |φ_{j,k}| ) )

    where:
        - m is the number of features
        - K is the number of rotated SHAP background samples (paper uses K=50)
        - φ_{j,k} is the mean-absolute SHAP value for feature j on the
          reference input set, computed under background sample k
        - Var_k(·) is the variance across the K rotated backgrounds

Operational protocol per §V:
    - K = 50 rotated background samples (each 100 training points drawn
      under seeded subsampling)
    - 200 reference inputs drawn from the test partition

Lower σ_SHAP indicates more stable attribution across audit cycles — the
governance-relevant notion of stability per §V.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import shap

DEFAULT_K_ROTATED_BACKGROUNDS: int = 50
DEFAULT_BACKGROUND_SIZE: int = 100
DEFAULT_N_REFERENCE_INPUTS: int = 200


@dataclass
class AttributionResult:
    """
    Output of a σ_SHAP measurement run on a single trained model.

    Attributes:
        model_name: Which model the attribution was computed for.
        sigma_shap: The σ_SHAP metric (lower = more stable attribution).
        mean_absolute_phi: Mean |φ| across features (reference magnitude
                           level, paper Table II column 3).
        per_feature_variance: 1-D array (shape: n_features,) of across-K
                              variance of mean-abs attribution per feature.
        feature_names: Tuple of feature names (preserved column order).
        k_backgrounds: Number of rotated backgrounds used.
        background_size: Size of each background sample.
        n_reference: Number of reference inputs the attribution was averaged over.
    """

    model_name: str
    sigma_shap: float
    mean_absolute_phi: float
    per_feature_variance: np.ndarray
    feature_names: Tuple[str, ...]
    k_backgrounds: int
    background_size: int
    n_reference: int


# ---------------------------------------------------------------------------
# σ_SHAP computation
# ---------------------------------------------------------------------------

def sigma_shap(per_feature_variance: np.ndarray) -> float:
    """
    Compute σ_SHAP from the per-feature across-K variance of mean-abs
    attribution magnitudes.

    σ_SHAP = sqrt( (1/m) · Σ_j Var_k( |φ_{j,k}| ) )

    Args:
        per_feature_variance: Shape (n_features,), the Var_k( |φ_{j,k}| )
            for each feature j.

    Returns:
        σ_SHAP as a Python float.
    """
    per_feature_variance = np.asarray(per_feature_variance, dtype=np.float64)
    if per_feature_variance.ndim != 1:
        raise ValueError(
            f"per_feature_variance must be 1-D (features,); got shape {per_feature_variance.shape}"
        )
    if (per_feature_variance < 0).any():
        raise ValueError("per_feature_variance must be non-negative")
    return float(np.sqrt(per_feature_variance.mean()))


# ---------------------------------------------------------------------------
# TreeSHAP measurement harness
# ---------------------------------------------------------------------------

def compute_rotated_shap_variance(
    model: object,
    model_name: str,
    X_train: np.ndarray,
    X_reference: np.ndarray,
    feature_names: Tuple[str, ...],
    k_backgrounds: int = DEFAULT_K_ROTATED_BACKGROUNDS,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    seed: int = 42,
) -> AttributionResult:
    """
    Compute σ_SHAP for a trained tree-based model using K rotated SHAP
    background samples.

    Procedure (§V):
        For k = 1..K:
            1. Subsample `background_size` training points as the SHAP
               background (seeded, k-indexed for reproducibility).
            2. Run TreeSHAP on `X_reference` with that background.
            3. Compute per-feature mean |φ| across the reference inputs:
                   φ_{j,k} = mean_i |SHAP[i, j]|
        Compute Var_k( |φ_{j,k}| ) per feature j (unbiased, ddof=1).
        σ_SHAP = sqrt( (1/m) Σ_j Var_k( |φ_{j,k}| ) ).

    Args:
        model: A trained tree-based classifier (XGBoost / LightGBM /
            CatBoost / RandomForest). TreeSHAP requires a tree model.
        model_name: Label used in the returned result.
        X_train: Training feature matrix (used as the background pool).
        X_reference: Reference input set (e.g. 200 test observations).
        feature_names: Tuple of feature names, length n_features.
        k_backgrounds: Number of rotated backgrounds (K). Paper default: 50.
        background_size: Size of each background sample. Paper default: 100.
        seed: Random seed for background subsampling.

    Returns:
        AttributionResult with σ_SHAP, mean |φ|, per-feature variance.

    Raises:
        ValueError: If feature-name count does not match X_reference's columns,
            or if background_size exceeds the training pool size.
    """
    n_features = X_reference.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X_reference columns ({n_features})."
        )
    if background_size > len(X_train):
        raise ValueError(
            f"background_size={background_size} exceeds training pool size {len(X_train)}."
        )

    rng = np.random.default_rng(seed)

    # Per-K per-feature mean-abs attribution matrix: shape (K, n_features)
    mean_abs_phi = np.zeros((k_backgrounds, n_features), dtype=np.float64)

    for k in range(k_backgrounds):
        bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
        background = X_train[bg_idx]

        explainer = shap.TreeExplainer(
            model,
            background,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X_reference, check_additivity=False)

        # Normalise shape: shap_values may be a list (multi-class) or ndarray.
        # For binary classification, shap_values is often a 2-D ndarray of
        # shape (n_reference, n_features) corresponding to the positive class.
        # Some library versions return shape (n_reference, n_features, n_classes)
        # — we take the positive-class slice.
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            # (n_reference, n_features, n_classes) → take class 1
            sv = sv[:, :, 1]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            sv = np.asarray(shap_values[1])
        # Else assume (n_reference, n_features) 2-D for the positive class.

        # mean_i |SHAP[i, j]|
        mean_abs_phi[k, :] = np.mean(np.abs(sv), axis=0)

    # Var_k across K rotated backgrounds, per feature. Use unbiased (ddof=1)
    # to match the paper's reported variance semantics for small K.
    per_feature_variance = mean_abs_phi.var(axis=0, ddof=1)

    sigma = sigma_shap(per_feature_variance)
    mean_abs_phi_overall = float(mean_abs_phi.mean())

    return AttributionResult(
        model_name=model_name,
        sigma_shap=sigma,
        mean_absolute_phi=mean_abs_phi_overall,
        per_feature_variance=per_feature_variance,
        feature_names=feature_names,
        k_backgrounds=k_backgrounds,
        background_size=background_size,
        n_reference=X_reference.shape[0],
    )
