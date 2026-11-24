"""
Split-conformal prediction wrapper for the PinkMate scoring ensemble.

Slice 7.5 — Production hardening + PRA SS1/23 alignment, 7.5a.

This module wraps the 6-model classification ensemble's ``predict_proba`` output
with a MAPIE split-conformal regressor so that, for a user-specified miscoverage
rate ``alpha``, every point estimate is accompanied by a ``(lower, upper)``
probability interval with the frequentist marginal-coverage guarantee

    P(P(Y=1 | X) in [lower, upper]) >= 1 - alpha   (exchangeability)

Decisions locked for this slice
-------------------------------
A. Library:            MAPIE >= 0.8.0 (actual installed: 1.6.0).
D. Default alpha:      0.10 (90% empirical coverage on the calibration set).
                       Credit-grade 99% (alpha=0.01) is overkill for GTM
                       decisioning; 95% is tunable via the constructor.

Degenerate commit mode
----------------------
Slice 1 shipped ``governance_commit(belief, theta=0.75)`` as a hard threshold on
the point score. This module introduces a strictly stricter governance rule:
commit ONLY when the lower bound of the conformal interval exceeds theta. This
matches PRA SS1/23 Principle 3 (conservatism in model use for credit and risk
decisions) and is how BoE-aligned deployments SHOULD interpret the commit
primitive in production.

                commit iff  lower_bound(X) >= theta

Usage sketch
------------
::

    from pinkmate.ml.conformal import ConformalScoringWrapper

    wrapper = ConformalScoringWrapper(ensemble, alpha=0.10)
    wrapper.conformalize(X_calibration, y_calibration)

    for lead_features in production_batch:
        point, lower, upper, committed = wrapper.predict_interval_single(
            lead_features, theta=0.75
        )

Theoretical references
----------------------
* Vovk et al. (2005), *Algorithmic Learning in a Random World*, Springer.
* Angelopoulos & Bates (2023), *Conformal Prediction: A Gentle Introduction*,
  Foundations and Trends in Machine Learning 16(4), 494-591.
* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. (2018),
  *Distribution-free predictive inference for regression*, JASA 113(523).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted

try:
    from mapie.regression import SplitConformalRegressor as _MapieSCR

    _MAPIE_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard for optional dep
    _MAPIE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public dataclass for structured conformal output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformalScore:
    """Structured single-lead output from a conformalised ensemble.

    Attributes
    ----------
    point_score:
        The ensemble's mean ``predict_proba(class=1)`` estimate in [0, 1].
    lower_bound:
        Lower end of the ``(1 - alpha)`` conformal interval on P(Y=1|X),
        clipped to [0, 1].
    upper_bound:
        Upper end of the ``(1 - alpha)`` conformal interval on P(Y=1|X),
        clipped to [0, 1].
    is_committed_at_theta:
        ``True`` iff the conservative commit rule fires:
        ``lower_bound >= theta``.
    alpha:
        The miscoverage rate that generated this interval. Stored on the
        output so downstream audit trails can reconstruct the coverage
        level without re-reading agent config.
    theta:
        The commit threshold that the commit flag was evaluated against.
    """

    point_score: float
    lower_bound: float
    upper_bound: float
    is_committed_at_theta: bool
    alpha: float
    theta: float

    def interval_width(self) -> float:
        """Return ``upper_bound - lower_bound`` for audit trail compactness."""
        return float(self.upper_bound - self.lower_bound)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "point_score": float(self.point_score),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "is_committed_at_theta": bool(self.is_committed_at_theta),
            "alpha": float(self.alpha),
            "theta": float(self.theta),
            "interval_width": float(self.interval_width()),
        }


# ---------------------------------------------------------------------------
# Ensemble-to-regressor adapter
# ---------------------------------------------------------------------------


class _EnsembleProbabilityRegressor(BaseEstimator, RegressorMixin):
    """Adapter exposing an ensemble's P(Y=1|X) as a sklearn RegressorMixin.

    MAPIE's ``SplitConformalRegressor`` requires a RegressorMixin. The
    PinkMate ensemble is a dict of classifiers, so this adapter does three
    things:

      1. Averages ``predict_proba(X)[:, 1]`` across all member classifiers.
      2. Exposes the mean as ``.predict(X)`` so MAPIE sees a regressor.
      3. Declares ``is_fitted_`` so ``check_is_fitted`` passes with
         ``prefit=True``.
    """

    def __init__(self, ensemble: dict, class_index: int = 1):
        self.ensemble = ensemble
        self.class_index = class_index
        self.is_fitted_ = True  # we never retrain; ensemble is prefit

        # Expose sklearn conventional "fitted" attributes so that MAPIE's
        # internal check_is_fitted passes cleanly without emitting a
        # UserWarning. We infer n_features_in_ from the first member
        # classifier that exposes it, and classes_ as the binary
        # positive/negative label pair (we only support binary here).
        self._init_sklearn_fitted_attrs()

    def _init_sklearn_fitted_attrs(self) -> None:
        n_features = None
        classes = None
        for _name, model in self.ensemble.items():
            n_features = getattr(model, "n_features_in_", n_features)
            classes = getattr(model, "classes_", classes)
            if n_features is not None and classes is not None:
                break
        if n_features is not None:
            self.n_features_in_ = int(n_features)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    # sklearn requires fit for the estimator protocol, but we no-op it
    # because all member models are already trained by the time this
    # adapter is constructed.
    def fit(self, X, y=None):  # noqa: D401 - sklearn contract
        """No-op (ensemble is already trained)."""
        self.is_fitted_ = True
        # Refresh exposed sklearn attributes in case the ensemble was
        # swapped out after construction.
        self._init_sklearn_fitted_attrs()
        return self

    def predict(self, X) -> NDArray[np.float64]:
        """Return mean P(Y=class_index | X) averaged over the ensemble."""
        check_is_fitted(self, "is_fitted_")
        probas = []
        for _name, model in self.ensemble.items():
            probas.append(model.predict_proba(X)[:, self.class_index])
        return np.mean(np.vstack(probas), axis=0)


# ---------------------------------------------------------------------------
# Public conformal-scoring wrapper
# ---------------------------------------------------------------------------


class ConformalScoringWrapper:
    """Split-conformal wrapper over a classification ensemble.

    Parameters
    ----------
    ensemble:
        Dict mapping model name -> sklearn-compatible classifier with
        ``predict_proba``. The ensemble itself is NOT retrained here.
    alpha:
        Miscoverage rate. Coverage guarantee is ``1 - alpha``. Default 0.10
        per Slice 7.5 Decision D.
    class_index:
        Which class the interval is on. Default 1 (positive class =
        conversion). Only binary classification is supported here.
    random_state:
        For MAPIE's internal splitting (unused when ``prefit=True``, but
        kept for API symmetry with future CQR extensions).

    Notes
    -----
    This wrapper is deliberately thin: it exposes the two MAPIE methods the
    scoring agent needs (``conformalize``, ``predict_interval_batch``) plus
    a single-lead convenience shim (``predict_interval_single``) that
    returns a ``ConformalScore`` dataclass.

    The coverage guarantee holds in expectation over exchangeable
    calibration + test draws. Distribution shift between calibration and
    production breaks it; Slice 7.5b (drift sentinels) is the companion
    safeguard that should warn when shift exceeds PSI > 0.25.
    """

    def __init__(
        self,
        ensemble: dict,
        alpha: float = 0.10,
        class_index: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        if not _MAPIE_AVAILABLE:
            raise ImportError(
                "mapie>=0.8.0 is required for ConformalScoringWrapper. "
                "Install with: pip install 'mapie>=0.8.0'."
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if class_index not in (0, 1):
            raise ValueError(
                f"class_index must be 0 or 1 (binary); got {class_index}"
            )
        if not isinstance(ensemble, dict) or len(ensemble) == 0:
            raise ValueError("ensemble must be a non-empty dict of models")

        self.ensemble = ensemble
        self.alpha = float(alpha)
        self.class_index = int(class_index)
        self.random_state = random_state

        self._adapter = _EnsembleProbabilityRegressor(
            ensemble=ensemble, class_index=class_index
        )
        self._mapie: Optional[_MapieSCR] = None
        self._is_conformalised: bool = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def conformalize(
        self,
        X_calibration: NDArray,
        y_calibration: NDArray,
    ) -> "ConformalScoringWrapper":
        """Compute conformity scores on a held-out calibration set.

        Parameters
        ----------
        X_calibration:
            Features for the calibration set. Must have the same shape
            schema as production inputs.
        y_calibration:
            Binary labels (0 or 1) for the calibration set. If
            ``class_index=1``, these are the true conversion flags.

        Returns
        -------
        self (for fluent chaining).
        """
        X_calibration = np.asarray(X_calibration)
        y_calibration = np.asarray(y_calibration).astype(float)

        if X_calibration.shape[0] != y_calibration.shape[0]:
            raise ValueError(
                f"X_calibration rows ({X_calibration.shape[0]}) != "
                f"y_calibration length ({y_calibration.shape[0]})"
            )
        if X_calibration.shape[0] < 20:
            raise ValueError(
                f"Calibration set too small ({X_calibration.shape[0]}). "
                "Need at least 20 points for a stable 90% interval; 200+ "
                "recommended for production deployments."
            )

        confidence_level = 1.0 - self.alpha
        # Note: SplitConformalRegressor does not accept random_state when
        # prefit=True (no internal splitting happens). We keep the field on
        # self for future CQR support (where random_state matters).
        self._mapie = _MapieSCR(
            estimator=self._adapter,
            confidence_level=confidence_level,
            conformity_score="absolute",
            prefit=True,
        )
        # MAPIE 1.x expects .conformalize on an unfitted (or prefit=True)
        # estimator. We pass our prefit adapter plus the calibration data.
        self._mapie.conformalize(X_calibration, y_calibration)
        self._is_conformalised = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_interval_batch(
        self, X: NDArray
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(point, lower, upper)`` arrays for a batch.

        All three arrays have the same length as ``X``. Lower and upper
        bounds are clipped to [0, 1] because they live on the probability
        simplex.
        """
        self._require_conformalised()
        assert self._mapie is not None

        X = np.asarray(X)
        points = self._adapter.predict(X)
        _pred_from_mapie, intervals = self._mapie.predict_interval(X)
        # intervals shape: (n_samples, 2, n_confidence_levels)
        intervals = np.asarray(intervals)
        if intervals.ndim == 3:
            lower = intervals[:, 0, 0]
            upper = intervals[:, 1, 0]
        elif intervals.ndim == 2:
            lower = intervals[:, 0]
            upper = intervals[:, 1]
        else:  # pragma: no cover
            raise RuntimeError(
                f"Unexpected interval shape from MAPIE: {intervals.shape}"
            )

        # Clip to probability simplex
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)

        return points, lower, upper

    def predict_interval_single(
        self,
        features: Union[NDArray, Sequence[float]],
        theta: float = 0.75,
    ) -> ConformalScore:
        """Return a ``ConformalScore`` for a single lead.

        Parameters
        ----------
        features:
            1-D feature vector or 2-D with a single row.
        theta:
            Governance commit threshold. The ``is_committed_at_theta`` flag
            is ``True`` iff ``lower_bound >= theta`` (strictly conservative).

        Notes
        -----
        The conservative commit rule is a REFINEMENT of the Slice 1
        ``governance_commit(belief, theta=0.75)`` primitive: the Slice 1
        rule fires whenever ``point >= theta``, this rule fires only when
        ``lower >= theta``. The two rules can be run side-by-side in trace
        output for comparison; the conformal lower-bound rule should be
        regarded as the production default under PRA SS1/23 Principle 3.
        """
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1]; got {theta}")

        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[0] != 1:
            raise ValueError(
                "predict_interval_single expects exactly one lead; got "
                f"{features.shape[0]}. Use predict_interval_batch for many."
            )

        points, lowers, uppers = self.predict_interval_batch(features)
        return ConformalScore(
            point_score=float(points[0]),
            lower_bound=float(lowers[0]),
            upper_bound=float(uppers[0]),
            is_committed_at_theta=bool(lowers[0] >= theta),
            alpha=self.alpha,
            theta=float(theta),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def empirical_coverage(
        self,
        X_test: NDArray,
        y_test: NDArray,
    ) -> float:
        """Return the fraction of ``y_test`` falling inside the predicted
        interval.

        A well-calibrated conformal wrapper should return approximately
        ``1 - alpha`` on an exchangeable test set. Use this method as a
        post-deployment smoke test; large deviations from the target
        indicate distribution shift between calibration and test time
        (Slice 7.5b drift sentinels catch this earlier at the feature
        level).
        """
        self._require_conformalised()
        y_test = np.asarray(y_test).astype(float)
        points, lowers, uppers = self.predict_interval_batch(X_test)
        in_interval = (y_test >= lowers) & (y_test <= uppers)
        return float(np.mean(in_interval))

    def interval_width_summary(
        self, X: NDArray
    ) -> dict:
        """Return mean/median/min/max/std of the interval widths on ``X``.

        Useful for detecting pathologically wide intervals (model very
        uncertain) or pathologically narrow ones (model overconfident,
        possibly undercovered).
        """
        self._require_conformalised()
        _p, lowers, uppers = self.predict_interval_batch(X)
        widths = uppers - lowers
        return {
            "mean": float(np.mean(widths)),
            "median": float(np.median(widths)),
            "min": float(np.min(widths)),
            "max": float(np.max(widths)),
            "std": float(np.std(widths)),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_conformalised(self) -> None:
        if not self._is_conformalised:
            raise RuntimeError(
                "ConformalScoringWrapper not yet conformalised. Call "
                ".conformalize(X_calibration, y_calibration) before "
                "requesting intervals."
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def commit_at_lower_bound(
    score: ConformalScore,
    theta: Optional[float] = None,
) -> bool:
    """Return True iff the lower bound of the interval is >= theta.

    This is the companion to Slice 1's ``governance_commit(belief, theta)``
    for the conformal path. If ``theta`` is None, the theta recorded on the
    score dataclass is used.
    """
    effective_theta = score.theta if theta is None else theta
    if not 0.0 <= effective_theta <= 1.0:
        raise ValueError(
            f"theta must be in [0, 1]; got {effective_theta}"
        )
    return bool(score.lower_bound >= effective_theta)


def interval_is_informative(
    score: ConformalScore, max_width: float = 0.5
) -> bool:
    """Return True iff the interval is narrower than ``max_width``.

    A width-0.5 interval on a probability simplex is uninformative; the
    production system should treat such outputs as low-confidence and
    either defer the commit or escalate to human review.
    """
    if not 0.0 < max_width <= 1.0:
        raise ValueError(f"max_width must be in (0, 1]; got {max_width}")
    return bool(score.interval_width() < max_width)


__all__ = [
    "ConformalScore",
    "ConformalScoringWrapper",
    "commit_at_lower_bound",
    "interval_is_informative",
]
