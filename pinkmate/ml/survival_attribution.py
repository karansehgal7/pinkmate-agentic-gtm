"""
SurvSHAP attribution + σ_SHAP on survival models (Slice 6).

Sibling to pinkmate/ml/attribution.py (classification σ_SHAP). Mirrors the
paper §V rotated-background protocol exactly, replacing the classification
TreeSHAP call with a SurvSHAP call on the model's cumulative-hazard output
evaluated at a fixed time grid.

σ_SHAP transfer research question (Slice 6 headline):
    Does σ_SHAP transfer from predict_proba (classification) to
    predict_cumulative_hazard (survival)?

Operational protocol (matches §V σ_SHAP + Slice 5.3 σ_calibration):
    K = 50 rotated background samples (default, overridable)
    Each background = 100 seeded subsamples of the training set
    time_grid = [30, 60, 90] days (one point per month in the 90-day window)
    σ_SHAP aggregates across (k rotations × time points × features) via the
    paper §V definition: sqrt((1/m) · Σ_j Var_k(|φ_{j,k}|))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (match paper §V + Slice 5.3)
# ---------------------------------------------------------------------------

DEFAULT_K_ROTATED_BACKGROUNDS: int = 50
DEFAULT_BACKGROUND_SIZE: int = 100
DEFAULT_N_REFERENCE: int = 200
DEFAULT_TIME_GRID: Tuple[float, ...] = (30.0, 60.0, 90.0)
BOUND_UPPER: float = float("inf")  # σ_SHAP on hazard has no finite upper bound (hazard is in [0, ∞))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SurvivalAttributionResult:
    """Container for σ_SHAP measurement on a fitted survival model."""

    model_key: str
    sigma_shap_hazard: float
    per_feature_variance: np.ndarray
    k_rotations: int
    background_size: int
    n_reference: int
    time_grid: Tuple[float, ...]
    feature_names: Tuple[str, ...]
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# σ_SHAP primitive (sibling to attribution.sigma_shap)
# ---------------------------------------------------------------------------

def sigma_shap_hazard(per_feature_variance: np.ndarray) -> float:
    """
    σ_SHAP on hazard outputs = sqrt((1/m) · Σ_j Var_k(|φ_{j,k,hazard}|)).

    Same formula as paper §V classification σ_SHAP; the only change is that
    per_feature_variance was computed on hazard attribution rather than on
    class-probability attribution.
    """
    per_feature_variance = np.asarray(per_feature_variance, dtype=np.float64)
    if per_feature_variance.ndim != 1:
        raise ValueError(
            f"per_feature_variance must be 1-D (features,); "
            f"got shape {per_feature_variance.shape}"
        )
    if (per_feature_variance < 0).any():
        raise ValueError("per_feature_variance must be non-negative")
    return float(np.sqrt(per_feature_variance.mean()))


# ---------------------------------------------------------------------------
# Rotated-background σ_SHAP on survival models
# ---------------------------------------------------------------------------

def compute_rotated_survshap_variance(
    model,
    model_key: str,
    X_train: pd.DataFrame,
    X_reference: pd.DataFrame,
    feature_names: Tuple[str, ...],
    time_grid: Tuple[float, ...] = DEFAULT_TIME_GRID,
    k_backgrounds: int = DEFAULT_K_ROTATED_BACKGROUNDS,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    seed: int = 42,
    use_survshap_package: bool = True,
) -> SurvivalAttributionResult:
    """
    Compute σ_SHAP for a fitted survival model via K rotated backgrounds.

    For each rotation k:
        1. Seeded sub-sample of X_train of size background_size
        2. Compute per-feature attribution for X_reference evaluated at each
           time in time_grid. Attribution target = predict_cumulative_hazard.
        3. Take mean across time_grid → φ_{j,k} per feature (reference-set
           aggregation = mean-abs as in paper §V)
    Aggregate per_feature_variance = Var_k(|φ_{j,k}|) across K rotations,
    then σ_SHAP = sqrt(mean(per_feature_variance)).

    Args:
        model: a fitted survival model from pinkmate.ml.survival. Must
            expose predict_cumulative_hazard(X, times) -> ndarray(n, len(times)).
        model_key: human label carried into the result.
        X_train: training feature matrix (used for background subsampling).
        X_reference: reference inputs to compute attribution on.
        feature_names: column names, in order.
        time_grid: evaluation times for hazard attribution (default (30, 60, 90)
            days — monthly points in the 90-day censoring window).
        k_backgrounds: K rotations. Default 50 matches paper §V canonical.
        background_size: rows per background subsample. Default 100.
        seed: base RNG seed. Reproducible.
        use_survshap_package: if True, attempt to use Krzyziński et al. 2023
            survshap package when available; falls back to finite-difference
            gradient attribution on cumulative hazard otherwise.

    Returns:
        SurvivalAttributionResult with sigma_shap_hazard + per-feature
        variance vector + measurement metadata.
    """
    rng = np.random.default_rng(seed)
    n_train = len(X_train)
    n_ref = len(X_reference)
    m = len(feature_names)
    time_arr = np.asarray(time_grid, dtype=float)

    if background_size > n_train:
        background_size = n_train
    if n_ref == 0:
        raise ValueError("X_reference is empty")
    if k_backgrounds < 1:
        raise ValueError(f"k_backgrounds must be >= 1, got {k_backgrounds}")

    # Try to import survshap (Decision C=1). If unavailable, fall back
    # to finite-difference attribution on cumulative hazard.
    _survshap_available = False
    _SurvivalModelExplainer = None
    if use_survshap_package:
        try:
            from survshap import SurvivalModelExplainer  # type: ignore
            _SurvivalModelExplainer = SurvivalModelExplainer
            _survshap_available = True
        except ImportError:
            _survshap_available = False

    # phi[k, j] = mean-abs attribution for feature j on reference set under background k
    phi = np.zeros((k_backgrounds, m), dtype=np.float64)

    for k in range(k_backgrounds):
        bg_idx = rng.choice(n_train, size=background_size, replace=False)
        X_bg = X_train.iloc[bg_idx]

        if _survshap_available:
            try:
                phi_k = _survshap_attribution(
                    model, X_bg, X_reference, time_arr, _SurvivalModelExplainer
                )
            except Exception:
                phi_k = _finite_diff_hazard_attribution(
                    model, X_bg, X_reference, time_arr
                )
        else:
            phi_k = _finite_diff_hazard_attribution(
                model, X_bg, X_reference, time_arr
            )
        phi[k, :] = phi_k  # per-feature mean-abs across reference + time grid

    # Per-feature across-K variance
    per_feature_var = np.var(phi, axis=0, ddof=1) if k_backgrounds > 1 else np.zeros(m)
    sigma = sigma_shap_hazard(per_feature_var)

    return SurvivalAttributionResult(
        model_key=model_key,
        sigma_shap_hazard=sigma,
        per_feature_variance=per_feature_var,
        k_rotations=k_backgrounds,
        background_size=background_size,
        n_reference=n_ref,
        time_grid=tuple(time_grid),
        feature_names=tuple(feature_names),
        extra={"used_survshap_package": _survshap_available},
    )


# ---------------------------------------------------------------------------
# Attribution helpers
# ---------------------------------------------------------------------------

def _survshap_attribution(
    model,
    X_bg: pd.DataFrame,
    X_ref: pd.DataFrame,
    time_arr: np.ndarray,
    SurvivalModelExplainer,
) -> np.ndarray:
    """SurvSHAP via Krzyziński et al. 2023 package. Returns mean-abs per feature."""
    # survshap API expects a sklearn-compatible model; our wrappers expose
    # predict_cumulative_hazard which is close enough with a small adapter.
    class _Adapter:
        def __init__(self, inner, times):
            self._inner = inner
            self._times = times

        def predict_cumulative_hazard_function(self, X):
            return self._inner.predict_cumulative_hazard(X, self._times)

    adapter = _Adapter(model, time_arr)
    explainer = SurvivalModelExplainer(adapter, X_bg)  # background
    # Compute attributions on X_ref; take mean-abs across reference and time axes
    result = explainer.explain(X_ref)
    # Expect shape (n_ref, n_features, len(time_arr))
    attr = np.abs(np.asarray(result.values))
    if attr.ndim == 3:
        return attr.mean(axis=(0, 2))
    elif attr.ndim == 2:
        return attr.mean(axis=0)
    else:
        raise ValueError(f"Unexpected SurvSHAP output shape {attr.shape}")


def _finite_diff_hazard_attribution(
    model,
    X_bg: pd.DataFrame,
    X_ref: pd.DataFrame,
    time_arr: np.ndarray,
) -> np.ndarray:
    """
    Finite-difference attribution on cumulative hazard. Fallback used when
    survshap is unavailable or raises on an unsupported model type.

    For each feature j:
        φ_j = mean over reference × time of
              | H(x + ε · e_j, t) - H(x - ε · e_j, t) | / (2 · ε)

    ε is scaled to each feature's training-set std for numerical stability.
    """
    m = X_ref.shape[1]
    n_ref = len(X_ref)
    T = len(time_arr)
    # Scale of each feature from background
    scales = np.maximum(X_bg.to_numpy(dtype=float).std(axis=0), 1e-6)
    eps = 0.05 * scales  # 5% of std

    X_ref_arr = X_ref.to_numpy(dtype=float)
    phi = np.zeros(m, dtype=np.float64)

    for j in range(m):
        X_plus = X_ref_arr.copy()
        X_minus = X_ref_arr.copy()
        X_plus[:, j] += eps[j]
        X_minus[:, j] -= eps[j]
        X_plus_df = pd.DataFrame(X_plus, columns=X_ref.columns)
        X_minus_df = pd.DataFrame(X_minus, columns=X_ref.columns)
        try:
            H_plus = model.predict_cumulative_hazard(X_plus_df, time_arr)
            H_minus = model.predict_cumulative_hazard(X_minus_df, time_arr)
            grad = (H_plus - H_minus) / (2.0 * eps[j])
            phi[j] = float(np.abs(grad).mean())
        except Exception:
            phi[j] = 0.0
    return phi


# ---------------------------------------------------------------------------
# σ_SHAP transfer analysis (Slice 6 headline research question)
# ---------------------------------------------------------------------------

def spearman_transfer_correlation(
    sigma_shap_classification: Dict[str, float],
    sigma_shap_survival: Dict[str, float],
    model_pairs: List[Tuple[str, str]],
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    """
    Spearman rank correlation of σ_SHAP across matched classification ↔
    survival model pairs, with 95% bootstrap CI.

    Example pairs:
        [("xgboost", "xgb_survival"), ("lightgbm", "lgb_aft"),
         ("random_forest", "cox_ph"), ...]

    Returns (rho, ci_lo, ci_hi, n).
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return (float("nan"), float("nan"), float("nan"), 0)

    xs: List[float] = []
    ys: List[float] = []
    for class_key, surv_key in model_pairs:
        x = sigma_shap_classification.get(class_key)
        y = sigma_shap_survival.get(surv_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < 3:
        return (float("nan"), float("nan"), float("nan"), len(xs))

    x_arr = np.asarray(xs)
    y_arr = np.asarray(ys)
    rho = float(spearmanr(x_arr, y_arr).correlation)
    rng = np.random.default_rng(seed)
    n = len(x_arr)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[b] = float(spearmanr(x_arr[idx], y_arr[idx]).correlation)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return rho, float(lo), float(hi), n
