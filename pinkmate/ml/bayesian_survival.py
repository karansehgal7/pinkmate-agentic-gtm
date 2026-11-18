"""
Bayesian Weibull survival with PyMC (Slice 7).

Puts priors on the Weibull shape and scale parameters, draws posterior
samples via NUTS, and reports credible intervals on survival probabilities.
Stronger than the Slice 6 point-estimate Weibull AFT because it quantifies
parameter uncertainty — useful for the regulatory evidence line where
"what's the 95% credible interval on P(convert by day 30)?" is a more
defensible statement than a single number.

Reference:
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A.,
    & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.

Decision A (17 Nov 2026): PyMC 5.x over NumPyro (maturity + Windows
compatibility). Decision B: NUTS sampler (standard for Bayesian survival).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


DEFAULT_N_DRAWS: int = 1000
DEFAULT_N_TUNE: int = 500
DEFAULT_N_CHAINS: int = 2


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BayesianSurvivalResult:
    """Container for a Bayesian Weibull survival fit."""

    posterior_shape: np.ndarray        # (n_samples,)
    posterior_scale: np.ndarray        # (n_samples,)
    n_samples: int                     # total posterior draws (chains * draws)
    n_chains: int
    n_tune: int
    rhat_shape: float                  # Gelman-Rubin diagnostic
    rhat_scale: float
    ess_shape: float                   # Effective sample size
    ess_scale: float
    n_observations: int
    n_events: int

    def posterior_mean(self) -> Tuple[float, float]:
        return float(self.posterior_shape.mean()), float(self.posterior_scale.mean())

    def credible_interval(
        self, parameter: str = "shape", level: float = 0.95
    ) -> Tuple[float, float]:
        """Equal-tailed credible interval at the given level."""
        if parameter == "shape":
            samples = self.posterior_shape
        elif parameter == "scale":
            samples = self.posterior_scale
        else:
            raise ValueError(f"parameter must be 'shape' or 'scale', got {parameter}")
        alpha = (1.0 - level) / 2.0
        return (
            float(np.quantile(samples, alpha)),
            float(np.quantile(samples, 1.0 - alpha)),
        )

    def survival_probability(
        self, t: float, level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Posterior credible interval on S(t) = P(T > t) under the Weibull.

        S(t) = exp(-(t / scale) ** shape)

        Returns (mean, ci_lo, ci_hi) for S(t) over the posterior.
        """
        s_samples = np.exp(-((t / self.posterior_scale) ** self.posterior_shape))
        alpha = (1.0 - level) / 2.0
        return (
            float(s_samples.mean()),
            float(np.quantile(s_samples, alpha)),
            float(np.quantile(s_samples, 1.0 - alpha)),
        )


# ---------------------------------------------------------------------------
# Bayesian Weibull fitter
# ---------------------------------------------------------------------------

def fit_bayesian_weibull(
    time: np.ndarray,
    event: np.ndarray,
    n_draws: int = DEFAULT_N_DRAWS,
    n_tune: int = DEFAULT_N_TUNE,
    n_chains: int = DEFAULT_N_CHAINS,
    target_accept: float = 0.9,
    seed: int = 42,
    progressbar: bool = False,
) -> BayesianSurvivalResult:
    """
    Fit a Bayesian Weibull model with weakly-informative priors to
    right-censored survival data via PyMC NUTS.

    Priors:
        shape ~ HalfNormal(sigma=2.0)
        scale ~ HalfNormal(sigma=max(time) * 2)

    Likelihood:
        event == 1: Weibull pdf
        event == 0: Weibull survival function (right-censored)

    Args:
        time: shape (n,), non-negative durations.
        event: shape (n,), 0 for censored, 1 for observed event.
        n_draws / n_tune / n_chains: NUTS sampler knobs.
        target_accept: NUTS target acceptance rate.
        seed: RNG seed for reproducibility.
        progressbar: whether PyMC prints a sampling progress bar.

    Returns:
        BayesianSurvivalResult with posterior samples + diagnostics.
    """
    import pymc as pm
    import pytensor.tensor as pt
    try:
        import arviz as az
    except ImportError:
        az = None

    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    n = len(time)
    if n != len(event):
        raise ValueError("time and event must have same length")
    if n < 5:
        raise ValueError(f"need at least 5 observations for Bayesian fit, got {n}")

    n_events = int((event > 0.5).sum())
    max_t = max(float(time.max()), 1.0)

    with pm.Model() as model:
        shape = pm.HalfNormal("shape", sigma=2.0)
        scale = pm.HalfNormal("scale", sigma=max_t * 2.0)

        # Log-likelihood contribution per observation (hand-rolled to handle
        # right-censoring correctly — PyMC's Weibull.observed cannot easily
        # mask censoring).
        log_pdf = (
            pt.log(shape)
            - pt.log(scale)
            + (shape - 1) * (pt.log(time + 1e-12) - pt.log(scale))
            - (time / scale) ** shape
        )
        log_surv = -((time / scale) ** shape)
        loglik_contrib = pt.switch(pt.gt(event, 0.5), log_pdf, log_surv)
        pm.Potential("loglik", pt.sum(loglik_contrib))

        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    # Extract posterior samples
    post_shape = trace.posterior["shape"].values.reshape(-1)
    post_scale = trace.posterior["scale"].values.reshape(-1)

    # Diagnostics (R-hat, ESS). arviz provides these; fall back to NaN if arviz missing.
    if az is not None:
        try:
            summary = az.summary(trace, var_names=["shape", "scale"])
            rhat_shape = float(summary.loc["shape", "r_hat"])
            rhat_scale = float(summary.loc["scale", "r_hat"])
            ess_shape = float(summary.loc["shape", "ess_bulk"])
            ess_scale = float(summary.loc["scale", "ess_bulk"])
        except Exception:
            rhat_shape = rhat_scale = float("nan")
            ess_shape = ess_scale = float("nan")
    else:
        rhat_shape = rhat_scale = float("nan")
        ess_shape = ess_scale = float("nan")

    return BayesianSurvivalResult(
        posterior_shape=post_shape,
        posterior_scale=post_scale,
        n_samples=len(post_shape),
        n_chains=n_chains,
        n_tune=n_tune,
        rhat_shape=rhat_shape,
        rhat_scale=rhat_scale,
        ess_shape=ess_shape,
        ess_scale=ess_scale,
        n_observations=n,
        n_events=n_events,
    )
