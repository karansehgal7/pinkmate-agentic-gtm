"""
Calibration-variance measurement — σ_calibration governance signal.

Sibling to pinkmate/ml/attribution.py's σ_SHAP:

    σ_SHAP measures ATTRIBUTION-stability under K-rotated SHAP backgrounds.
    σ_calibration measures CALIBRATION-stability under K-rotated test
    resamples.

Research question (Slice 5.3):
    Does σ_calibration correlate with σ_SHAP?
        YES → calibration-stability may serve as a cheaper proxy for
              attribution-stability (useful when SHAP is expensive).
        NO  → both are complementary governance signals and must be
              reported together; attribution-stable models may still
              be calibration-unstable and vice versa.

Three properties the measurement MUST satisfy (enforced by
tests/test_calibration_variance.py):
    1. Bounded: σ_ECE ∈ [0, 0.5] and σ_Brier ∈ [0, 0.5]
    2. K-stable: converges as K increases; σ at K=50 is within noise
       of σ at K=10 on well-behaved data.
    3. Reproducible: same seed produces identical σ.

σ_calibration definition (Slice 5.3, sibling to paper §V σ_SHAP):

    σ_ECE   = sqrt( Var_k( ECE_k ) )     (sample std over K rotations)
    σ_Brier = sqrt( Var_k( Brier_k ) )   (sample std over K rotations)

    where ECE_k and Brier_k are computed on the k-th rotated resample
    of the test set.

Operational protocol (matches σ_SHAP for direct comparability):
    - K = 50 rotations (matches paper's canonical K)
    - Seeded resampling with replacement (bootstrap) of the test set
    - ECE n_bins = 10 (standard binning scheme)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants (match paper's canonical σ_SHAP protocol)
# ---------------------------------------------------------------------------

K_DEFAULT: int = 50              # matches paper §V canonical K
ECE_N_BINS_DEFAULT: int = 10     # standard binning for ECE
BOUND_UPPER: float = 0.5         # theoretical upper bound on σ_ECE / σ_Brier


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Container for one model's σ_calibration measurement."""

    sigma_ece: float
    sigma_brier: float
    mean_ece: float
    mean_brier: float
    per_rotation_ece: np.ndarray
    per_rotation_brier: np.ndarray
    k_rotations: int
    n_samples: int
    model_name: str = ""


# ---------------------------------------------------------------------------
# Single-shot calibration metrics
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = ECE_N_BINS_DEFAULT,
) -> float:
    """
    Expected Calibration Error (Naeini et al. 2015), binned by predicted
    probability.

    ECE = sum_b  (|B_b| / n) * |avg_pred_b - avg_true_b|

    where the sum is over n_bins equal-width probability bins.

    Returns:
        ECE in [0, 1]. Lower is better-calibrated.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0
    if len(y_pred_proba) != n:
        raise ValueError(
            f"y_true length ({n}) does not match y_pred_proba length "
            f"({len(y_pred_proba)})"
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        # Right-inclusive for the last bin so proba == 1.0 is captured
        if b == n_bins - 1:
            mask = (y_pred_proba >= bin_edges[b]) & (y_pred_proba <= bin_edges[b + 1])
        else:
            mask = (y_pred_proba >= bin_edges[b]) & (y_pred_proba < bin_edges[b + 1])
        bin_size = int(mask.sum())
        if bin_size == 0:
            continue
        bin_frac = bin_size / n
        bin_avg_pred = float(y_pred_proba[mask].mean())
        bin_avg_true = float(y_true[mask].mean())
        ece += bin_frac * abs(bin_avg_pred - bin_avg_true)
    return float(ece)


def compute_brier(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Brier score = mean squared error between predicted probability and
    binary outcome (Brier 1950).

    Brier = (1/n) * sum_i (p_i - y_i)^2

    Returns:
        Brier in [0, 1]. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64).ravel()
    if len(y_true) == 0:
        return 0.0
    if len(y_pred_proba) != len(y_true):
        raise ValueError(
            f"y_true length ({len(y_true)}) does not match y_pred_proba "
            f"length ({len(y_pred_proba)})"
        )
    return float(np.mean((y_pred_proba - y_true) ** 2))


# ---------------------------------------------------------------------------
# σ_calibration primitive
# ---------------------------------------------------------------------------

def sigma_calibration(per_rotation_metric: np.ndarray) -> float:
    """
    Compute σ (standard deviation) of a calibration metric across K
    rotations.

    Mirrors pinkmate.ml.attribution.sigma_shap signature-wise: takes the
    per-k vector and returns the scalar σ via the sample standard
    deviation (ddof=1).

    Args:
        per_rotation_metric: shape (K,), the calibration metric (ECE or
            Brier) at each of the K rotations.

    Returns:
        σ as a Python float. Returns 0.0 for K < 2 (no variance defined
        on a single point).
    """
    per_rotation_metric = np.asarray(per_rotation_metric, dtype=np.float64).ravel()
    if (per_rotation_metric < 0).any():
        raise ValueError("per_rotation_metric must be non-negative")
    if len(per_rotation_metric) < 2:
        return 0.0
    return float(np.std(per_rotation_metric, ddof=1))


# ---------------------------------------------------------------------------
# K-rotated calibration variance (main entry point)
# ---------------------------------------------------------------------------

def compute_rotated_calibration_variance(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    model_name: str = "",
    k_rotations: int = K_DEFAULT,
    n_bins: int = ECE_N_BINS_DEFAULT,
    seed: int = 42,
) -> CalibrationResult:
    """
    Measure σ_calibration via K-rotated bootstrap resamples of the test
    set.

    For each rotation k:
        1. Draw a bootstrap resample (sample with replacement, same n)
           using a deterministic per-k seed derived from `seed`.
        2. Predict probabilities via predict_proba_fn on the resample.
        3. Compute ECE_k and Brier_k on the resample.
    Aggregate:
        σ_ECE   = sample std of [ECE_1, ..., ECE_K]
        σ_Brier = sample std of [Brier_1, ..., Brier_K]

    Args:
        X_test: shape (n, d) — features of the test set.
        y_test: shape (n,) — binary labels in {0, 1}.
        predict_proba_fn: callable taking X and returning either
            shape (n,) positive-class probas OR shape (n, 2) sklearn-style
            probas (positive class column is taken as [:, 1]).
        model_name: optional label carried into the result.
        k_rotations: number of bootstrap rotations. Defaults to K=50 to
            match paper's canonical σ_SHAP protocol.
        n_bins: ECE binning (default 10, matches sklearn convention).
        seed: base seed for the per-rotation RNG. Reproducible.

    Returns:
        CalibrationResult with sigma_ece, sigma_brier, means, and the
        per-rotation vectors for downstream K-stability analysis.
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    n = len(y_test)
    if n == 0:
        raise ValueError("test set is empty")
    if X_test.shape[0] != n:
        raise ValueError(
            f"X_test rows ({X_test.shape[0]}) != y_test length ({n})"
        )
    if k_rotations < 1:
        raise ValueError(f"k_rotations must be >= 1, got {k_rotations}")

    rng = np.random.default_rng(seed)

    per_rotation_ece = np.zeros(k_rotations, dtype=np.float64)
    per_rotation_brier = np.zeros(k_rotations, dtype=np.float64)

    for k in range(k_rotations):
        idx = rng.choice(n, size=n, replace=True)
        X_k = X_test[idx]
        y_k = y_test[idx]
        proba_k = predict_proba_fn(X_k)
        proba_k = np.asarray(proba_k, dtype=np.float64)
        # Accept either (n,) or (n, 2) probability outputs
        if proba_k.ndim == 2:
            if proba_k.shape[1] < 2:
                raise ValueError(
                    f"predict_proba_fn returned shape {proba_k.shape}; "
                    f"expected (n,) or (n, 2)"
                )
            proba_k = proba_k[:, 1]
        proba_k = proba_k.ravel()
        # Clip to [0, 1] to guard against numerical drift
        proba_k = np.clip(proba_k, 0.0, 1.0)
        per_rotation_ece[k] = compute_ece(y_k, proba_k, n_bins=n_bins)
        per_rotation_brier[k] = compute_brier(y_k, proba_k)

    return CalibrationResult(
        sigma_ece=sigma_calibration(per_rotation_ece),
        sigma_brier=sigma_calibration(per_rotation_brier),
        mean_ece=float(per_rotation_ece.mean()),
        mean_brier=float(per_rotation_brier.mean()),
        per_rotation_ece=per_rotation_ece,
        per_rotation_brier=per_rotation_brier,
        k_rotations=k_rotations,
        n_samples=n,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# K-stability sweep helper
# ---------------------------------------------------------------------------

def sweep_k_stability(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    k_values: Optional[tuple] = None,
    n_bins: int = ECE_N_BINS_DEFAULT,
    seed: int = 42,
) -> dict:
    """
    Compute σ_calibration at a sequence of K values to verify K-stability
    (Property 2).

    A well-behaved σ_calibration estimator produces σ at K=50 within a
    small neighbourhood of σ at K=10 — i.e. the metric is NOT a function
    of K, only of the model + data. Dependence on K would indicate
    pathological rotation-noise.

    Returns a dict mapping K -> CalibrationResult.
    """
    if k_values is None:
        k_values = (5, 10, 20, 50)
    return {
        k: compute_rotated_calibration_variance(
            X_test=X_test,
            y_test=y_test,
            predict_proba_fn=predict_proba_fn,
            k_rotations=k,
            n_bins=n_bins,
            seed=seed,
        )
        for k in k_values
    }
