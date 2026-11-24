"""
Production-hardening pipeline that composes the Slice 7.5 modules.

This module provides a single entry point — the ``ProductionHardeningPipeline``
class — that the ScoringAgent instantiates once at startup and calls on every
prediction. Keeping the logic here as a dedicated pipeline class means the
ScoringAgent integration is a small, focused change (a few lines in
``__init__`` and two hooks in the prediction path) rather than invasive
rewiring.

Pipeline responsibilities
-------------------------
1. **Pre-prediction drift check** (Slice 7.5b): given the inference feature
   vector, compare each feature's value against the training-time snapshot
   and produce a list of ``'<level>:<feature>'`` flags.
2. **Post-prediction conformal interval** (Slice 7.5a): wrap the ensemble's
   predict_proba with a conformal interval (lower/upper bounds + conservative
   commit flag at the lower bound).
3. **Batch-mode fairness diagnostic** (Slice 7.5c): on completion of a batch,
   compute Fairlearn demographic-parity + equalized-odds metrics across each
   protected attribute in the batch.

ScoringAgent integration pattern (see SLICE_7_5_INTEGRATION.md)
---------------------------------------------------------------
The live ScoringAgent uses the LangGraph ``__call__(state: PipelineState)``
node pattern (not a stand-alone ``score_single`` method). The hook points
are ``_load_models`` (load calibration.npz) + ``_predict_proba_real``
(attach conformal interval) + ``__call__`` (write to ``ScoreOutput``).
::

    # In agents/scoring_agent.py __init__, after _load_models():
    self.hardening = ProductionHardeningPipeline.from_models_dir(
        models_dir=self.models_dir,
        ensemble={n: m for n, m in self.models.items() if m is not None},
        X_calibration=self._X_calibration,
        y_calibration=self._y_calibration,
        alpha=0.10,
    )

    # In _predict_proba_real, after `consensus_proba = float(np.mean(probs))`:
    if self.hardening is not None:
        self._last_conformal = (
            self.hardening.conformal.predict_interval_single(x[0], theta=0.75)
        )

    # In __call__, after `state.score = ScoreOutput(final_score=..., ...)`:
    if getattr(self, "_last_conformal", None) is not None:
        state.score.conformal_lower = round(100.0 * self._last_conformal.lower_bound, 2)
        state.score.conformal_upper = round(100.0 * self._last_conformal.upper_bound, 2)

    # In batch handoff (new, called when batch complete):
    fairness_reports = self.hardening.audit_batch_fairness(
        batch_predictions, batch_protected_attributes, batch_true_labels
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from pinkmate.ml.conformal import (
    ConformalScore,
    ConformalScoringWrapper,
)
from pinkmate.ml.drift import (
    DriftThresholds,
    FeatureDriftResult,
    FeatureSnapshot,
    compute_feature_drift,
    load_distribution_snapshot,
    summarise_drift_flags,
)
from pinkmate.ml.fairness import (
    FairnessReport,
    FairnessThresholds,
    audit_fairness_multi,
    summarise_fairness_flags,
)


# ---------------------------------------------------------------------------
# Structured single-lead output
# ---------------------------------------------------------------------------


@dataclass
class HardenedScore:
    """Single-lead output combining conformal score + drift flags.

    Attributes
    ----------
    conformal:
        The :class:`pinkmate.ml.conformal.ConformalScore` dataclass carrying
        point_score, lower_bound, upper_bound and the conservative commit
        flag.
    drift_flags:
        List of ``'<level>:<feature>'`` strings, one per feature in the
        reference snapshot. Empty list when no snapshot is loaded (first-run
        on a fresh clone).
    drift_is_alerting:
        True if any feature is at level ``'red'``. The scoring agent should
        propagate this to the trace and, if a hard-fail-on-alert policy is
        enabled, abort the commit.
    """

    conformal: ConformalScore
    drift_flags: List[str] = field(default_factory=list)
    drift_is_alerting: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict suitable for trace logging."""
        return {
            "conformal": self.conformal.as_dict(),
            "drift_flags": list(self.drift_flags),
            "drift_is_alerting": bool(self.drift_is_alerting),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ProductionHardeningPipeline:
    """Compose conformal + drift + fairness checks behind a single API.

    Parameters
    ----------
    conformal_wrapper:
        A pre-conformalised :class:`ConformalScoringWrapper`. Construct
        separately with calibration data.
    reference_snapshots:
        Dict of feature-name -> :class:`FeatureSnapshot`. Optional; empty
        or missing snapshots mean drift checks are no-ops (first-run on a
        fresh clone falls into this bucket, with a startup log noting the
        snapshot was not found).
    drift_thresholds:
        PSI sentinel thresholds. Defaults to the PRA credit-risk convention.
    fairness_thresholds:
        Fairness parity-difference thresholds. Defaults to EU AI Act / FCA
        DP24/1 convention.
    """

    def __init__(
        self,
        conformal_wrapper: ConformalScoringWrapper,
        reference_snapshots: Optional[Dict[str, FeatureSnapshot]] = None,
        drift_thresholds: Optional[DriftThresholds] = None,
        fairness_thresholds: Optional[FairnessThresholds] = None,
    ) -> None:
        self.conformal = conformal_wrapper
        self.reference_snapshots = reference_snapshots or {}
        self.drift_thresholds = drift_thresholds or DriftThresholds()
        self.fairness_thresholds = fairness_thresholds or FairnessThresholds()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_models_dir(
        cls,
        models_dir: Union[str, Path],
        ensemble: Dict,
        X_calibration: NDArray,
        y_calibration: NDArray,
        alpha: float = 0.10,
        snapshot_filename: str = "train_distribution_snapshot.json",
        drift_thresholds: Optional[DriftThresholds] = None,
        fairness_thresholds: Optional[FairnessThresholds] = None,
    ) -> "ProductionHardeningPipeline":
        """Build a pipeline from a models directory + calibration set.

        Loads the training distribution snapshot from
        ``<models_dir>/<snapshot_filename>`` if it exists; otherwise
        instantiates with an empty snapshot dict and drift checks become
        no-ops.
        """
        models_dir = Path(models_dir)
        snapshot_path = models_dir / snapshot_filename

        reference_snapshots: Dict[str, FeatureSnapshot] = {}
        if snapshot_path.exists():
            reference_snapshots = load_distribution_snapshot(snapshot_path)

        wrapper = ConformalScoringWrapper(ensemble=ensemble, alpha=alpha)
        wrapper.conformalize(X_calibration, y_calibration)

        return cls(
            conformal_wrapper=wrapper,
            reference_snapshots=reference_snapshots,
            drift_thresholds=drift_thresholds,
            fairness_thresholds=fairness_thresholds,
        )

    # ------------------------------------------------------------------
    # Single-lead scoring
    # ------------------------------------------------------------------

    def check_drift(
        self,
        current_batch: Mapping[str, NDArray],
        raise_warning_on_red: bool = True,
    ) -> List[FeatureDriftResult]:
        """Compute per-feature drift between snapshot and current batch.

        If no reference snapshot is loaded (first-run on a fresh clone),
        returns an empty list and the scoring agent should log a warning
        once at startup and proceed without drift checks.
        """
        if not self.reference_snapshots:
            return []
        return compute_feature_drift(
            reference_snapshots=self.reference_snapshots,
            current_batch={
                k: np.asarray(v) for k, v in current_batch.items()
            },
            thresholds=self.drift_thresholds,
            raise_warning_on_red=raise_warning_on_red,
        )

    def score_with_interval(
        self,
        lead_features: NDArray,
        theta: float = 0.75,
    ) -> ConformalScore:
        """Produce a conformal score for a single lead.

        This is the primary single-lead scoring entry point. The returned
        :class:`ConformalScore` carries the Slice 1 point score plus
        Slice 7.5a lower/upper bounds and the conservative
        ``is_committed_at_theta`` flag at ``theta``.
        """
        return self.conformal.predict_interval_single(lead_features, theta=theta)

    def score_lead(
        self,
        lead_features: NDArray,
        feature_dict_for_drift: Optional[Mapping[str, NDArray]] = None,
        theta: float = 0.75,
    ) -> HardenedScore:
        """Combined single-lead scoring: conformal + drift flags.

        Parameters
        ----------
        lead_features:
            1-D feature vector (model input).
        feature_dict_for_drift:
            Mapping feature-name -> single-value array. Used for drift
            checks. Can be ``None`` to skip drift (e.g. in unit tests).
        theta:
            Governance commit threshold.
        """
        conformal_score = self.score_with_interval(lead_features, theta=theta)

        drift_flags: List[str] = []
        drift_is_alerting = False
        if feature_dict_for_drift is not None and self.reference_snapshots:
            drift_results = self.check_drift(feature_dict_for_drift)
            drift_flags = summarise_drift_flags(drift_results)
            drift_is_alerting = any(r.level == "red" for r in drift_results)

        return HardenedScore(
            conformal=conformal_score,
            drift_flags=drift_flags,
            drift_is_alerting=drift_is_alerting,
        )

    # ------------------------------------------------------------------
    # Batch fairness
    # ------------------------------------------------------------------

    def audit_batch_fairness(
        self,
        y_pred: NDArray,
        protected_attributes: Mapping[str, NDArray],
        y_true: Optional[NDArray] = None,
    ) -> List[FairnessReport]:
        """Run Fairlearn audit across all protected attributes in a batch.

        This is a BATCH-MODE operation (needs >= ~50 leads for stable
        metrics). Call at batch handoff, not on every single lead.

        Parameters
        ----------
        y_pred:
            Predictions, length ``n_batch``.
        protected_attributes:
            Mapping attribute-name -> group labels array. Comes from
            aggregating ``LeadInput.protected_attributes`` across the batch.
        y_true:
            Optional true labels for equalized-odds metrics. In live
            inference ``y_true`` is unavailable; fall back to
            demographic-parity-only reports.
        """
        return audit_fairness_multi(
            y_pred=y_pred,
            protected_attributes=protected_attributes,
            y_true=y_true,
            thresholds=self.fairness_thresholds,
        )

    def summarise_fairness(
        self,
        reports: Sequence[FairnessReport],
    ) -> List[str]:
        """Return a flat sorted list of ``'<level>:<attribute>'`` strings."""
        return summarise_fairness_flags(reports)


__all__ = [
    "HardenedScore",
    "ProductionHardeningPipeline",
]
