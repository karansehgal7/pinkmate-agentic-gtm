"""
Distribution drift sentinels for the PinkMate scoring agent.

Slice 7.5 — Production hardening + PRA SS1/23 alignment, 7.5b.

This module implements three complementary scalar drift metrics between a
reference distribution (the training-time feature snapshot) and a current
distribution (an inference-time feature sample):

  * **PSI** — Population Stability Index, the industry-standard credit-risk
    metric for monitoring feature stability between score development and
    deployment. Bounded in [0, infinity) with PRA SS1/23 Section 5
    thresholds: ``<= 0.10`` = stable, ``0.10 < x <= 0.25`` = shift warning,
    ``> 0.25`` = significant shift.
  * **KL divergence** — KL(current || reference). Asymmetric by
    construction; informative when PSI fires but direction of shift matters.
    Reported in nats.
  * **KS test** — two-sample Kolmogorov-Smirnov. The p-value and D statistic
    together say whether the empirical CDFs differ beyond sampling noise.

All three metrics are implemented purely in numpy/scipy; no new dependencies
are added. Train-time distribution snapshots are serialised to JSON by each
training script and loaded by the ScoringAgent at startup.

Decisions locked for this slice
-------------------------------
B. Backend:  Pure numpy/scipy. Alibi Detect is a Slice 8+ upgrade path for
             online / streaming drift detection.
E. PSI thresholds (PRA credit-risk convention):

   ====================================  ================  ==================
   PSI value                              Level             Action
   ====================================  ================  ==================
   psi <= 0.10                            green             stable, proceed
   0.10 < psi <= 0.25                     yellow            log + proceed
   psi > 0.25                             red               raise RuntimeWarning
   ====================================  ================  ==================

   Both thresholds are configurable via :class:`DriftThresholds` for
   deployments that prefer stricter cutoffs (e.g. 0.05 warn / 0.15 alert
   for more conservative stacks).

References
----------
* Siddiqi, N. (2017), *Intelligent Credit Scoring*, Wiley, ch. 7.
* Bank of England PRA SS1/23, *Model Risk Management Principles for Banks*,
  May 2023, Principle 3 on ongoing monitoring.
* Yurdakul, B. (2018), *Statistical Properties of Population Stability
  Index*, PhD thesis, Western Michigan University.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable PSI sentinel thresholds.

    The defaults match the PRA SS1/23 credit-risk convention (0.10 warn,
    0.25 alert) referenced in the module docstring. Deployments that need
    stricter cutoffs can construct with ``DriftThresholds(warn=0.05,
    alert=0.15)``.

    Attributes
    ----------
    warn:
        Below this PSI value the feature is green ("stable").
    alert:
        Above this PSI value the feature is red ("significant shift") and
        the ``psi_sentinel`` raises a ``RuntimeWarning``.
    """

    warn: float = 0.10
    alert: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.warn < self.alert:
            raise ValueError(
                f"Thresholds must satisfy 0 < warn ({self.warn}) < "
                f"alert ({self.alert})."
            )

    def classify(self, psi_value: float) -> str:
        """Return 'green', 'yellow' or 'red' for a given PSI value."""
        if psi_value <= self.warn:
            return "green"
        if psi_value <= self.alert:
            return "yellow"
        return "red"


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def _compute_bin_edges(
    reference: NDArray[np.float64], n_bins: int
) -> NDArray[np.float64]:
    """Build ``n_bins`` equal-frequency bin edges from reference values.

    Uses quantiles so each bin holds roughly the same share of the
    reference sample. Deduplicates edges so degenerate features (many
    identical values, e.g. binary flags) still yield a valid partition.
    The two extreme edges are extended to +/- infinity so that values
    slightly outside the reference range at inference time are still
    assigned to a bin.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    # Deduplicate (degenerate feature) by inserting a tiny jitter
    unique_edges = np.unique(edges)
    if len(unique_edges) < 2:
        # Feature has only one unique value in reference; make a trivial
        # two-bin partition centered at that value.
        v = unique_edges[0]
        unique_edges = np.array([v - 1e-9, v + 1e-9])
    # Extend extremes to catch inference points slightly beyond reference range
    unique_edges[0] = -np.inf
    unique_edges[-1] = np.inf
    return unique_edges


def _bucket_fractions(
    sample: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    """Return ``p_i`` = fraction of ``sample`` in each bucket.

    An ``epsilon`` floor prevents log(0) in PSI / KL computations. This
    floor also handles the edge case where a bucket is empty in one
    distribution; without it PSI would diverge even for an obvious match.
    """
    counts, _ = np.histogram(sample, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        raise ValueError("Sample is empty; cannot compute PSI buckets.")
    fractions = counts.astype(float) / total
    return np.clip(fractions, epsilon, 1.0)


def compute_psi(
    reference: NDArray,
    current: NDArray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI between a reference and a current 1-D sample.

    Formula::

        PSI = sum_i (current_i - reference_i) * ln(current_i / reference_i)

    where ``current_i`` and ``reference_i`` are the fractions of each
    sample in bucket ``i``, under an equal-frequency partition of the
    reference sample.

    Parameters
    ----------
    reference:
        Training-time sample. Used to build the equal-frequency bins.
    current:
        Inference-time sample.
    n_bins:
        Number of buckets. Credit-risk convention is 10.
    epsilon:
        Floor applied to zero-fraction buckets to keep PSI finite. 1e-6 is
        standard.

    Returns
    -------
    PSI value in ``[0, inf)``.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError(
            "Both reference and current must contain at least one "
            "non-NaN value."
        )

    edges = _compute_bin_edges(ref, n_bins=n_bins)
    p_ref = _bucket_fractions(ref, edges, epsilon=epsilon)
    p_cur = _bucket_fractions(cur, edges, epsilon=epsilon)

    psi_terms = (p_cur - p_ref) * np.log(p_cur / p_ref)
    return float(np.sum(psi_terms))


def psi_sentinel(
    psi_value: float,
    feature_name: str,
    thresholds: Optional[DriftThresholds] = None,
    raise_warning_on_red: bool = True,
) -> str:
    """Classify a PSI value and optionally raise a ``RuntimeWarning``.

    Parameters
    ----------
    psi_value:
        The PSI value from :func:`compute_psi`.
    feature_name:
        Human-readable feature name, used in the warning text.
    thresholds:
        Thresholds to apply. Defaults to ``DriftThresholds()``.
    raise_warning_on_red:
        If ``True`` (default), raise a ``RuntimeWarning`` when the PSI
        exceeds the ``alert`` threshold. The warning text names the
        feature and the PSI value for audit-trail capture.

    Returns
    -------
    'green', 'yellow' or 'red'.
    """
    if thresholds is None:
        thresholds = DriftThresholds()
    level = thresholds.classify(psi_value)
    if level == "red" and raise_warning_on_red:
        warnings.warn(
            f"Significant feature drift detected: {feature_name} "
            f"PSI={psi_value:.4f} > alert threshold {thresholds.alert}",
            RuntimeWarning,
            stacklevel=2,
        )
    return level


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def compute_kl_divergence(
    reference: NDArray,
    current: NDArray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute KL(current || reference) in nats on binned samples.

    Note the asymmetry: ``KL(P||Q) != KL(Q||P)``. This function computes
    the divergence in the direction most relevant to drift detection,
    namely how much the CURRENT distribution deviates from the REFERENCE
    one. A large value with a small PSI indicates concentrated mass has
    moved between buckets without changing marginal shape.

    Implementation uses the same equal-frequency partition as
    :func:`compute_psi` so the two metrics are directly comparable.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError(
            "Both reference and current must contain at least one "
            "non-NaN value."
        )

    edges = _compute_bin_edges(ref, n_bins=n_bins)
    p_ref = _bucket_fractions(ref, edges, epsilon=epsilon)
    p_cur = _bucket_fractions(cur, edges, epsilon=epsilon)

    kl_terms = p_cur * np.log(p_cur / p_ref)
    return float(np.sum(kl_terms))


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov
# ---------------------------------------------------------------------------


def compute_ks_test(
    reference: NDArray,
    current: NDArray,
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test on continuous samples.

    Returns ``(statistic, p_value)``. The statistic is the max absolute
    difference between the two empirical CDFs; the p-value is derived
    from scipy's two-sample KS implementation.

    Rule of thumb for drift sentinels:
    ``p_value < 0.05`` + ``PSI > 0.10`` = strong evidence of distribution
    shift. A KS p-value alone is prone to firing on large-sample noise;
    always pair with PSI.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError(
            "Both reference and current must contain at least one "
            "non-NaN value."
        )

    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Train-time distribution snapshot persistence
# ---------------------------------------------------------------------------


@dataclass
class FeatureSnapshot:
    """A single feature's training-time distribution snapshot.

    Stored as:

      - ``values``: a sample of up to ``max_sample_size`` training values
        (so PSI/KL/KS can re-bin on demand at inference time)
      - summary statistics for a fast sanity check at agent startup

    Attributes
    ----------
    name:
        Feature name (must match the LeadInput schema field name).
    values:
        Training sample (subsampled if larger than ``max_sample_size``).
    mean:
        Sample mean of the full training distribution.
    std:
        Sample std of the full training distribution.
    median:
        Sample median of the full training distribution.
    n_training:
        Full size of the original training sample (before subsampling).
    """

    name: str
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    n_training: int = 0

    @classmethod
    def from_array(
        cls,
        name: str,
        values: NDArray,
        max_sample_size: int = 5000,
        random_state: Optional[int] = 42,
    ) -> "FeatureSnapshot":
        """Build a snapshot from a 1-D training array."""
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            raise ValueError(
                f"Feature {name!r}: training sample has no non-NaN values."
            )

        rng = np.random.default_rng(random_state)
        if arr.size > max_sample_size:
            idx = rng.choice(arr.size, size=max_sample_size, replace=False)
            subsample = arr[idx]
        else:
            subsample = arr.copy()

        return cls(
            name=name,
            values=[float(v) for v in subsample],
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            n_training=int(arr.size),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "name": self.name,
            "values": list(self.values),
            "mean": float(self.mean),
            "std": float(self.std),
            "median": float(self.median),
            "n_training": int(self.n_training),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSnapshot":
        """Rebuild from a dict produced by ``to_dict``."""
        return cls(
            name=str(d["name"]),
            values=list(d.get("values", [])),
            mean=float(d.get("mean", 0.0)),
            std=float(d.get("std", 0.0)),
            median=float(d.get("median", 0.0)),
            n_training=int(d.get("n_training", 0)),
        )


def save_distribution_snapshot(
    path: Union[str, os.PathLike],
    snapshots: Dict[str, FeatureSnapshot],
    schema_version: str = "slice75_production_hardening_v1",
) -> None:
    """Serialise a dict of ``FeatureSnapshot`` values to a JSON file.

    The ``schema_version`` key is written into the top-level JSON
    envelope for forward compatibility: if the snapshot format changes
    in a later slice, the loader can detect and migrate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": schema_version,
        "features": {
            name: snapshot.to_dict() for name, snapshot in snapshots.items()
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2)


def load_distribution_snapshot(
    path: Union[str, os.PathLike],
) -> Dict[str, FeatureSnapshot]:
    """Load a snapshot file produced by :func:`save_distribution_snapshot`.

    Raises
    ------
    FileNotFoundError:
        If the snapshot file does not exist. Callers that want graceful
        degradation on first-run (no snapshot yet) should catch this and
        fall back to no drift checks.
    ValueError:
        If the file exists but is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No training distribution snapshot at {path}. Run the "
            "training script that writes "
            f"``{path.name}`` first, or disable drift checks."
        )
    with open(path, "r", encoding="utf-8") as fh:
        envelope = json.load(fh)
    features = envelope.get("features", {})
    if not isinstance(features, dict):
        raise ValueError(
            f"Snapshot at {path} is malformed: 'features' is not a dict."
        )
    return {
        name: FeatureSnapshot.from_dict(payload)
        for name, payload in features.items()
    }


# ---------------------------------------------------------------------------
# Batch drift computation
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    """Per-feature drift output.

    Attributes
    ----------
    feature:
        Feature name.
    psi:
        PSI value, or ``None`` if the feature was missing from the
        current batch.
    kl:
        KL(current || reference), or ``None`` if the feature was missing.
    ks_statistic:
        KS D statistic, or ``None`` if the feature was missing.
    ks_pvalue:
        KS p-value, or ``None`` if the feature was missing.
    level:
        'green', 'yellow', 'red' or 'missing'.
    """

    feature: str
    psi: Optional[float]
    kl: Optional[float]
    ks_statistic: Optional[float]
    ks_pvalue: Optional[float]
    level: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict for trace logging."""
        return {
            "feature": self.feature,
            "psi": self.psi,
            "kl": self.kl,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "level": self.level,
        }


def compute_feature_drift(
    reference_snapshots: Dict[str, FeatureSnapshot],
    current_batch: Dict[str, NDArray],
    thresholds: Optional[DriftThresholds] = None,
    raise_warning_on_red: bool = True,
    n_bins: int = 10,
) -> List[FeatureDriftResult]:
    """Compute drift for every feature in both reference and current.

    Parameters
    ----------
    reference_snapshots:
        Dict mapping feature name -> ``FeatureSnapshot`` (as produced by
        :func:`load_distribution_snapshot`).
    current_batch:
        Dict mapping feature name -> 1-D array of inference-time values.
        Features missing here are marked ``level='missing'``.
    thresholds:
        PSI thresholds. Defaults to ``DriftThresholds()``.
    raise_warning_on_red:
        Whether to raise a ``RuntimeWarning`` when PSI > alert.
    n_bins:
        Number of PSI buckets. Default 10.

    Returns
    -------
    List of :class:`FeatureDriftResult`, one entry per feature in
    ``reference_snapshots``.
    """
    if thresholds is None:
        thresholds = DriftThresholds()

    results: List[FeatureDriftResult] = []
    for name, snapshot in reference_snapshots.items():
        if name not in current_batch:
            results.append(
                FeatureDriftResult(
                    feature=name,
                    psi=None,
                    kl=None,
                    ks_statistic=None,
                    ks_pvalue=None,
                    level="missing",
                )
            )
            continue

        reference_values = np.asarray(snapshot.values)
        current_values = np.asarray(current_batch[name])
        try:
            psi_value = compute_psi(
                reference_values, current_values, n_bins=n_bins
            )
            kl_value = compute_kl_divergence(
                reference_values, current_values, n_bins=n_bins
            )
            ks_stat, ks_p = compute_ks_test(reference_values, current_values)
            level = psi_sentinel(
                psi_value,
                feature_name=name,
                thresholds=thresholds,
                raise_warning_on_red=raise_warning_on_red,
            )
        except ValueError:
            # Empty samples or other edge cases: mark as missing.
            results.append(
                FeatureDriftResult(
                    feature=name,
                    psi=None,
                    kl=None,
                    ks_statistic=None,
                    ks_pvalue=None,
                    level="missing",
                )
            )
            continue

        results.append(
            FeatureDriftResult(
                feature=name,
                psi=psi_value,
                kl=kl_value,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                level=level,
            )
        )
    return results


def summarise_drift_flags(results: List[FeatureDriftResult]) -> List[str]:
    """Return a flat list of ``'<level>:<feature>'`` strings.

    Useful for attaching to a single-lead trace via
    ``ScoringOutput.drift_flags``: a compact, sorted representation
    of which features are in which bucket.
    """
    flags: List[str] = []
    for r in results:
        flags.append(f"{r.level}:{r.feature}")
    return sorted(flags)


__all__ = [
    "DriftThresholds",
    "FeatureSnapshot",
    "FeatureDriftResult",
    "compute_psi",
    "compute_kl_divergence",
    "compute_ks_test",
    "psi_sentinel",
    "save_distribution_snapshot",
    "load_distribution_snapshot",
    "compute_feature_drift",
    "summarise_drift_flags",
]
