"""
Weibull mixture model for latent buyer-regime discovery (Slice 7).

Fits a K-component Weibull mixture to survival data (time, event) via the
EM algorithm, with BIC-based selection of K across {2, 3, 4}. The fitted
mixture components correspond to latent buyer "regimes" — for example a
fast-converter regime with low mean time + high shape vs a slow-converter
regime with high mean time + lower shape.

Reference:
    McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models.
    Wiley Series in Probability and Statistics.

Decision C (17 Nov 2026, actor-critic): K selected by BIC sweep over
K in {2, 3, 4}. Not committing to K=2 upfront; not non-parametric
(Dirichlet Process would be Slice 8+ overkill).

Right-censoring is handled honestly: censored observations contribute to
the likelihood via the survival function S(t) = exp(-(t/scale)^shape)
rather than the density. This matches the Slice 6 lifelines convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import weibull_min


DEFAULT_K_CANDIDATES: Tuple[int, ...] = (2, 3, 4)
EM_MAX_ITER: int = 200
EM_TOL: float = 1e-5


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WeibullComponent:
    """One component of a Weibull mixture."""
    shape: float   # k in Weibull(k, lambda)
    scale: float   # lambda in Weibull(k, lambda)

    def pdf(self, t: np.ndarray) -> np.ndarray:
        """Weibull density at time t."""
        return weibull_min.pdf(t, self.shape, scale=self.scale)

    def survival(self, t: np.ndarray) -> np.ndarray:
        """Weibull survival function S(t)."""
        return weibull_min.sf(t, self.shape, scale=self.scale)

    def log_pdf(self, t: np.ndarray) -> np.ndarray:
        return weibull_min.logpdf(t, self.shape, scale=self.scale)

    def log_survival(self, t: np.ndarray) -> np.ndarray:
        return weibull_min.logsf(t, self.shape, scale=self.scale)


@dataclass
class MixtureFitResult:
    """Container for a fitted Weibull mixture."""
    k: int
    components: List[WeibullComponent]
    mixing_weights: np.ndarray       # shape (k,), sum to 1
    log_likelihood: float
    bic: float
    aic: float
    n_iter: int
    converged: bool
    n_samples: int
    n_events: int

    @property
    def n_parameters(self) -> int:
        """Free parameters: 2 per component (shape + scale) + (k-1) weights."""
        return 2 * self.k + (self.k - 1)


# ---------------------------------------------------------------------------
# EM algorithm for Weibull mixture
# ---------------------------------------------------------------------------

def _weighted_weibull_mle(
    t: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    init_shape: float = 1.5,
    init_scale: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Weighted MLE for a single Weibull distribution with right-censoring.
    Used inside the EM M-step.

    Returns (shape, scale).
    """
    t = np.asarray(t, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-12)

    if init_scale is None:
        init_scale = max(float(np.average(t, weights=weights)), 1.0)

    def neg_loglik(params):
        log_k, log_lam = params
        k = np.exp(log_k)
        lam = np.exp(log_lam)
        # log pdf for events + log S for censored
        log_contrib = np.where(
            event > 0.5,
            weibull_min.logpdf(t, k, scale=lam),
            weibull_min.logsf(t, k, scale=lam),
        )
        return -float(np.sum(weights * log_contrib))

    res = minimize(
        neg_loglik,
        x0=np.array([np.log(init_shape), np.log(init_scale)]),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-6},
    )
    k = float(np.exp(res.x[0]))
    lam = float(np.exp(res.x[1]))
    return k, lam


def fit_weibull_mixture(
    time: np.ndarray,
    event: np.ndarray,
    k: int,
    max_iter: int = EM_MAX_ITER,
    tol: float = EM_TOL,
    seed: int = 42,
) -> MixtureFitResult:
    """
    EM fit of a K-component Weibull mixture to right-censored survival data.

    Args:
        time: shape (n,), non-negative durations.
        event: shape (n,), 0 for right-censored, 1 for observed event.
        k: number of mixture components.
        max_iter / tol: EM stopping criteria.
        seed: RNG seed for initial component assignment.

    Returns:
        MixtureFitResult with components, weights, log-likelihood, BIC, AIC.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    n = len(time)
    if n != len(event):
        raise ValueError("time and event must have same length")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    rng = np.random.default_rng(seed)

    # ----- Initialise components via quantile-based scales + random shapes -----
    quantiles = np.quantile(time, np.linspace(0.2, 0.8, k))
    components: List[WeibullComponent] = []
    for q in quantiles:
        components.append(WeibullComponent(shape=1.0 + rng.uniform(0.2, 0.8), scale=max(float(q), 1.0)))
    mixing = np.ones(k) / k

    prev_ll = -np.inf
    converged = False
    n_iter_done = 0

    for it in range(max_iter):
        n_iter_done = it + 1
        # ----- E-step: responsibilities -----
        log_resp = np.zeros((n, k), dtype=np.float64)
        for j, comp in enumerate(components):
            log_contrib = np.where(
                event > 0.5,
                comp.log_pdf(time),
                comp.log_survival(time),
            )
            log_resp[:, j] = np.log(max(mixing[j], 1e-12)) + log_contrib
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        log_resp = log_resp - log_norm
        resp = np.exp(log_resp)  # shape (n, k)

        current_ll = float(np.sum(log_norm))

        # ----- M-step -----
        # mixing weights
        nk = resp.sum(axis=0)
        mixing = nk / nk.sum()
        # per-component MLE
        new_components: List[WeibullComponent] = []
        for j in range(k):
            w = resp[:, j]
            if w.sum() < 1e-6:
                new_components.append(components[j])
                continue
            k_j, lam_j = _weighted_weibull_mle(
                time, event, w,
                init_shape=components[j].shape,
                init_scale=components[j].scale,
            )
            new_components.append(WeibullComponent(shape=k_j, scale=lam_j))
        components = new_components

        if abs(current_ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = current_ll

    # ----- Final log-likelihood -----
    log_resp = np.zeros((n, k), dtype=np.float64)
    for j, comp in enumerate(components):
        log_contrib = np.where(
            event > 0.5,
            comp.log_pdf(time),
            comp.log_survival(time),
        )
        log_resp[:, j] = np.log(max(mixing[j], 1e-12)) + log_contrib
    final_ll = float(np.sum(logsumexp(log_resp, axis=1)))

    n_params = 2 * k + (k - 1)
    bic = -2 * final_ll + n_params * np.log(n)
    aic = -2 * final_ll + 2 * n_params

    return MixtureFitResult(
        k=k,
        components=components,
        mixing_weights=mixing,
        log_likelihood=final_ll,
        bic=bic,
        aic=aic,
        n_iter=n_iter_done,
        converged=converged,
        n_samples=n,
        n_events=int((event > 0.5).sum()),
    )


def select_k_by_bic(
    time: np.ndarray,
    event: np.ndarray,
    k_candidates: Sequence[int] = DEFAULT_K_CANDIDATES,
    seed: int = 42,
) -> Tuple[int, Dict[int, MixtureFitResult]]:
    """
    BIC sweep over k_candidates. Returns (best_k, {k: MixtureFitResult}).
    Lower BIC is better. Default candidates are {2, 3, 4}.
    """
    results: Dict[int, MixtureFitResult] = {}
    for k in k_candidates:
        try:
            results[k] = fit_weibull_mixture(time, event, k=k, seed=seed)
        except Exception as e:
            print(f"[mixture_survival] k={k} fit failed: {e}")
    if not results:
        raise RuntimeError("All k candidates failed to fit")
    best_k = min(results, key=lambda kk: results[kk].bic)
    return best_k, results
