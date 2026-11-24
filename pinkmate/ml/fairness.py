"""
Fairness diagnostics for the PinkMate scoring agent.

Slice 7.5 — Production hardening + PRA SS1/23 alignment, 7.5c.

This module wraps Microsoft's Fairlearn library to produce a pure-diagnostic
fairness report over the scoring agent's predictions. Protected attributes
are specified via the ``LeadInput`` schema extension
(``protected_attributes: dict``); ``employee_band`` is the default audit axis
(SMB-vs-mid-market-vs-enterprise disparate-impact detection, as flagged by
the EU AI Act and FCA DP24/1 as a stated regulatory interest).

The fairness report is a DIAGNOSTIC: it does NOT change predictions. The
metrics live in the agent trace for human-reviewer consumption and downstream
model-risk-management reporting under PRA SS1/23 Principle 4 (independent
challenge of model use).

Decisions locked for this slice
-------------------------------
C. Library:            Fairlearn >= 0.10.0 (actual installed: 0.15.0).
Meta-choice 2:         Default protected attribute = ``employee_band``;
                       deployments may configure arbitrary additional
                       attributes via ``LeadInput.protected_attributes``.

Three metrics reported per protected group
------------------------------------------
* **selection_rate** — fraction of leads in each group with a positive
  prediction. Large disparities across groups suggest the scorer is
  systematically ranking one segment higher.
* **demographic_parity_difference** — max group selection rate minus min.
  Target: close to zero. The EU AI Act Article 10 cites this as a baseline
  bias check.
* **equalized_odds_difference** — max of ``|TPR_g - TPR_g'|`` and
  ``|FPR_g - FPR_g'|`` across groups. A stricter criterion than demographic
  parity because it conditions on the ground-truth label.

Interpretation threshold (industry convention, EU AI Act context):
``difference <= 0.10`` = acceptable; ``0.10 < diff <= 0.20`` = yellow;
``> 0.20`` = red (requires intervention or documented mitigation).

References
----------
* Hardt, M., Price, E., Srebro, N. (2016), *Equality of Opportunity in
  Supervised Learning*, NeurIPS 29.
* European Union (2024), Artificial Intelligence Act, Article 10
  (Data and data governance) and Annex III.
* Financial Conduct Authority (2024), DP24/1 *AI in Financial Services*,
  discussion paper.
* Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., Wallach, H.
  (2018), *A Reductions Approach to Fair Classification*, ICML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )

    _FAIRLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard for optional dep
    _FAIRLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Thresholds for the three-level traffic-light report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FairnessThresholds:
    """Thresholds mapping a parity difference to a traffic-light level.

    Defaults follow the EU AI Act / FCA DP24/1 informal convention of 10%
    as the acceptable upper bound and 20% as the red line. Deployments
    operating in higher-risk domains (credit scoring, housing, employment)
    may want stricter thresholds; the config is per-deployment.

    Attributes
    ----------
    warn:
        Difference strictly above this is 'yellow'.
    alert:
        Difference strictly above this is 'red'.
    """

    warn: float = 0.10
    alert: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 < self.warn < self.alert:
            raise ValueError(
                f"Thresholds must satisfy 0 < warn ({self.warn}) < "
                f"alert ({self.alert})."
            )

    def classify(self, difference: float) -> str:
        """Return 'green', 'yellow' or 'red' for a given parity difference."""
        if difference <= self.warn:
            return "green"
        if difference <= self.alert:
            return "yellow"
        return "red"


# ---------------------------------------------------------------------------
# Public dataclass for structured fairness output
# ---------------------------------------------------------------------------


@dataclass
class FairnessReport:
    """Structured fairness audit output for one protected attribute.

    Attributes
    ----------
    attribute:
        Name of the protected attribute this report covers, e.g.
        'employee_band'.
    group_selection_rates:
        Dict mapping group label -> selection rate (fraction predicted
        positive within the group). Empty if labels were not supplied.
    group_true_positive_rates:
        Dict mapping group label -> TPR. ``None`` if ``y_true`` was not
        passed (e.g. live-inference mode).
    group_false_positive_rates:
        Dict mapping group label -> FPR. ``None`` if ``y_true`` was not
        passed.
    demographic_parity_difference:
        Max group selection rate minus min.
    demographic_parity_ratio:
        Min group selection rate divided by max. In [0, 1]; 1 is perfect
        parity.
    equalized_odds_difference:
        Max of the group TPR/FPR disparities. ``None`` without y_true.
    equalized_odds_ratio:
        Fairlearn's ratio variant (min/max of TPR/FPR disparities).
        ``None`` without y_true.
    level:
        Overall traffic-light level: 'green', 'yellow' or 'red'. Derived
        from the max of the available difference metrics.
    n_samples:
        Number of samples the audit was run on.
    """

    attribute: str
    group_selection_rates: Dict[str, float] = field(default_factory=dict)
    group_true_positive_rates: Optional[Dict[str, float]] = None
    group_false_positive_rates: Optional[Dict[str, float]] = None
    demographic_parity_difference: float = 0.0
    demographic_parity_ratio: float = 1.0
    equalized_odds_difference: Optional[float] = None
    equalized_odds_ratio: Optional[float] = None
    level: str = "green"
    n_samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict for trace logging."""
        return {
            "attribute": self.attribute,
            "group_selection_rates": {
                str(k): float(v)
                for k, v in self.group_selection_rates.items()
            },
            "group_true_positive_rates": (
                {str(k): float(v) for k, v in self.group_true_positive_rates.items()}
                if self.group_true_positive_rates is not None
                else None
            ),
            "group_false_positive_rates": (
                {str(k): float(v) for k, v in self.group_false_positive_rates.items()}
                if self.group_false_positive_rates is not None
                else None
            ),
            "demographic_parity_difference": float(
                self.demographic_parity_difference
            ),
            "demographic_parity_ratio": float(self.demographic_parity_ratio),
            "equalized_odds_difference": (
                None
                if self.equalized_odds_difference is None
                else float(self.equalized_odds_difference)
            ),
            "equalized_odds_ratio": (
                None
                if self.equalized_odds_ratio is None
                else float(self.equalized_odds_ratio)
            ),
            "level": self.level,
            "n_samples": int(self.n_samples),
        }


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------


def _ensure_binary_predictions(y_pred: NDArray) -> NDArray[np.int64]:
    """Return y_pred coerced to int {0, 1}, thresholding probabilities at 0.5."""
    y_pred = np.asarray(y_pred)
    if y_pred.dtype.kind in ("f",):
        return (y_pred >= 0.5).astype(np.int64)
    return y_pred.astype(np.int64)


def audit_fairness(
    y_pred: NDArray,
    sensitive_features: NDArray,
    attribute_name: str = "employee_band",
    y_true: Optional[NDArray] = None,
    thresholds: Optional[FairnessThresholds] = None,
) -> FairnessReport:
    """Run a Fairlearn-based fairness audit for one protected attribute.

    Parameters
    ----------
    y_pred:
        Predicted labels or probabilities. Probabilities are thresholded
        at 0.5 for selection-rate computation.
    sensitive_features:
        Categorical group labels, one per sample. String / int / mixed
        dtypes are all fine.
    attribute_name:
        Human-readable name of the attribute (default 'employee_band',
        per Slice 7.5 meta-choice 2).
    y_true:
        True labels. If provided, equalized-odds metrics are added to the
        report; without them, only demographic-parity-style metrics.
    thresholds:
        Traffic-light thresholds. Defaults to ``FairnessThresholds()``.

    Returns
    -------
    A fully-populated :class:`FairnessReport`.

    Notes
    -----
    This is a PURE DIAGNOSTIC — predictions are unchanged. The report
    lives in the agent trace. If the deployment wants to mitigate (e.g.
    re-weight, re-threshold per group), Fairlearn's ``reductions`` or
    ``postprocessing`` APIs provide those; adding a mitigation pass is a
    Slice 8+ follow-on.
    """
    if not _FAIRLEARN_AVAILABLE:
        raise ImportError(
            "fairlearn>=0.10.0 is required for audit_fairness. "
            "Install with: pip install 'fairlearn>=0.10.0'."
        )

    y_pred_binary = _ensure_binary_predictions(y_pred)
    sensitive_features = np.asarray(sensitive_features)
    if y_pred_binary.shape[0] != sensitive_features.shape[0]:
        raise ValueError(
            f"y_pred length ({y_pred_binary.shape[0]}) does not match "
            f"sensitive_features length ({sensitive_features.shape[0]})."
        )
    if y_pred_binary.shape[0] == 0:
        raise ValueError("Empty y_pred / sensitive_features arrays.")

    if thresholds is None:
        thresholds = FairnessThresholds()

    # Selection rate per group (always available)
    frame = MetricFrame(
        metrics={"selection_rate": selection_rate},
        y_true=np.zeros_like(y_pred_binary),  # dummy for selection_rate
        y_pred=y_pred_binary,
        sensitive_features=sensitive_features,
    )
    group_selection = {
        str(group): float(frame.by_group["selection_rate"].loc[group])
        for group in frame.by_group.index
    }

    rates = np.array(list(group_selection.values()), dtype=float)
    dp_diff = float(rates.max() - rates.min())
    # Fairlearn's demographic_parity_ratio is min/max over nonzero groups
    if rates.max() > 0:
        dp_ratio = float(rates.min() / rates.max())
    else:
        dp_ratio = 1.0  # all zero predictions in every group

    # Equalized odds requires y_true
    if y_true is not None:
        y_true = np.asarray(y_true).astype(np.int64)
        if y_true.shape[0] != y_pred_binary.shape[0]:
            raise ValueError(
                f"y_true length ({y_true.shape[0]}) does not match "
                f"y_pred length ({y_pred_binary.shape[0]})."
            )

        frame_odds = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred_binary,
            sensitive_features=sensitive_features,
        )
        tpr_by_group = {
            str(g): float(frame_odds.by_group["tpr"].loc[g])
            for g in frame_odds.by_group.index
        }
        fpr_by_group = {
            str(g): float(frame_odds.by_group["fpr"].loc[g])
            for g in frame_odds.by_group.index
        }
        eo_diff = float(
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred_binary,
                sensitive_features=sensitive_features,
            )
        )
        eo_ratio = float(
            equalized_odds_ratio(
                y_true=y_true,
                y_pred=y_pred_binary,
                sensitive_features=sensitive_features,
            )
        )
    else:
        tpr_by_group = None
        fpr_by_group = None
        eo_diff = None
        eo_ratio = None

    # Overall level: worst of the available difference metrics
    worst_diff = dp_diff if eo_diff is None else max(dp_diff, eo_diff)
    level = thresholds.classify(worst_diff)

    return FairnessReport(
        attribute=attribute_name,
        group_selection_rates=group_selection,
        group_true_positive_rates=tpr_by_group,
        group_false_positive_rates=fpr_by_group,
        demographic_parity_difference=dp_diff,
        demographic_parity_ratio=dp_ratio,
        equalized_odds_difference=eo_diff,
        equalized_odds_ratio=eo_ratio,
        level=level,
        n_samples=int(y_pred_binary.shape[0]),
    )


def audit_fairness_multi(
    y_pred: NDArray,
    protected_attributes: Mapping[str, NDArray],
    y_true: Optional[NDArray] = None,
    thresholds: Optional[FairnessThresholds] = None,
) -> List[FairnessReport]:
    """Run :func:`audit_fairness` across several protected attributes.

    Parameters
    ----------
    y_pred:
        Predicted labels or probabilities, length ``n_samples``.
    protected_attributes:
        Dict mapping attribute name -> array of group labels, each of
        length ``n_samples``. The ``LeadInput.protected_attributes``
        schema field feeds directly into this parameter after collection
        across a batch.
    y_true:
        Optional true labels for equalized-odds metrics.
    thresholds:
        Traffic-light thresholds, shared across attributes.

    Returns
    -------
    List of :class:`FairnessReport`, one per attribute.
    """
    if not protected_attributes:
        return []
    return [
        audit_fairness(
            y_pred=y_pred,
            sensitive_features=sens,
            attribute_name=name,
            y_true=y_true,
            thresholds=thresholds,
        )
        for name, sens in protected_attributes.items()
    ]


def summarise_fairness_flags(reports: Sequence[FairnessReport]) -> List[str]:
    """Return a flat list of ``'<level>:<attribute>'`` strings.

    Useful for attaching to a trace via ``ScoringOutput.drift_flags``'s
    fairness companion, or for a dashboard view.
    """
    return sorted(f"{r.level}:{r.attribute}" for r in reports)


__all__ = [
    "FairnessThresholds",
    "FairnessReport",
    "audit_fairness",
    "audit_fairness_multi",
    "summarise_fairness_flags",
]
