"""
LeadForge → survival schema transform (Slice 6).

Converts LeadForge classification data (features + binary ``converted`` label)
into the (time, event) representation required by Cox PH + parametric AFT
+ XGBoost survival:cox + LightGBM AFT models.

Decision B (16 November 2026, actor-critic iteration): time is constructed
SYNTHETICALLY from LeadForge's touch primitives because the dataset lacks
a native ``days_since_first_touch`` column.

    days_since_first_touch = touch_count * touch_interval_days
                             + days_since_last_touch (when available)
    time  = min(days_since_first_touch, max_followup_days)
    event = converted AND time <= max_followup_days  (else right-censored)

Honesty bar (also surfaced in docs/survival_training_report.md on every
training run):
    touch_interval_days = 7.0 (weekly cadence) is a modelling assumption.
    Sensitivity analysis is baked in: tests enforce that the σ_SHAP
    transfer VERDICT is stable across touch_interval_days ∈ {5.0, 7.0, 14.0}.
    Reported survival curves should be read as "time-to-conversion under
    plausible synthetic time construction" NOT as "time-to-conversion from
    directly observed first-touch timestamps".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (locked by Decision B, 16 Nov 2026)
# ---------------------------------------------------------------------------

DEFAULT_MAX_FOLLOWUP_DAYS: float = 90.0        # right-censoring horizon
DEFAULT_TOUCH_INTERVAL_DAYS: float = 7.0       # weekly cadence assumption
SENSITIVITY_TOUCH_INTERVALS: tuple = (5.0, 7.0, 14.0)  # verdict-stability range


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SurvivalFrame:
    """Container for a dataset transformed into survival analysis form."""

    X: pd.DataFrame            # feature matrix (same columns as source df, minus time/event/converted)
    time: np.ndarray           # shape (n,), float, in days, non-negative
    event: np.ndarray          # shape (n,), int 0/1, 1 = conversion observed before censoring
    feature_names: tuple       # column names of X, in order
    n_samples: int
    n_events: int              # number of uncensored observations
    censoring_rate: float      # 1 - n_events/n_samples
    max_followup_days: float
    touch_interval_days: float

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"SurvivalFrame(n={self.n_samples}, events={self.n_events}, "
            f"censoring={self.censoring_rate:.2%}, "
            f"features={len(self.feature_names)}, "
            f"max_followup={self.max_followup_days}d, "
            f"touch_interval={self.touch_interval_days}d)"
        )


# ---------------------------------------------------------------------------
# Transform (main entry point)
# ---------------------------------------------------------------------------

def transform_leadforge_to_survival(
    df: pd.DataFrame,
    max_followup_days: float = DEFAULT_MAX_FOLLOWUP_DAYS,
    touch_interval_days: float = DEFAULT_TOUCH_INTERVAL_DAYS,
    label_column: str = "converted",
    touch_count_column: str = "touch_count",
    days_since_last_touch_column: str = "days_since_last_touch",
    drop_columns: Optional[tuple] = None,
) -> SurvivalFrame:
    """
    Transform a LeadForge-style classification dataframe into a SurvivalFrame.

    Deterministic — no randomness, repeated calls on the same df produce
    identical outputs. (Tests enforce this.)

    Args:
        df: input dataframe. Must contain ``label_column`` and at least
            one of ``touch_count_column`` / ``days_since_last_touch_column``
            (ValueError raised if both are absent — no safe time construction).
        max_followup_days: right-censoring horizon (default 90.0).
        touch_interval_days: assumed days between touches (default 7.0;
            see SENSITIVITY_TOUCH_INTERVALS for the validated range).
        label_column: column name holding binary ``converted`` ∈ {0, 1}.
        touch_count_column: column name holding integer touch count.
        days_since_last_touch_column: column name holding non-negative float.
        drop_columns: additional columns to drop from X (e.g. ids, emails).

    Returns:
        SurvivalFrame with X (features), time (days), event (0/1), and
        descriptive metadata.

    Raises:
        ValueError if label_column is missing, OR if BOTH touch_count and
        days_since_last_touch are missing, OR if any constructed time
        is negative, OR if label values are outside {0, 1}.
    """
    if label_column not in df.columns:
        raise ValueError(
            f"label_column '{label_column}' not found in df; "
            f"columns = {list(df.columns)[:10]}..."
        )

    has_tc = touch_count_column in df.columns
    has_dslt = days_since_last_touch_column in df.columns
    if not (has_tc or has_dslt):
        raise ValueError(
            f"At least one of '{touch_count_column}' or "
            f"'{days_since_last_touch_column}' must be present in df; "
            f"no safe time construction possible without either."
        )

    if max_followup_days <= 0:
        raise ValueError(f"max_followup_days must be > 0, got {max_followup_days}")
    if touch_interval_days <= 0:
        raise ValueError(f"touch_interval_days must be > 0, got {touch_interval_days}")

    # ----- Validate labels -----
    y = df[label_column].to_numpy()
    if not np.all(np.isin(y, [0, 1])):
        bad = np.unique(y[~np.isin(y, [0, 1])])
        raise ValueError(
            f"label_column '{label_column}' must contain only {{0, 1}}; "
            f"found unexpected values: {bad.tolist()[:5]}"
        )

    # ----- Construct days_since_first_touch (Decision B=3) -----
    # touch_count * touch_interval_days gives "how long they've been in the pipeline"
    # (pre-observation window). days_since_last_touch gives "how long since
    # we last saw engagement". Summing both approximates total observation duration.
    n = len(df)
    base_touches = (
        df[touch_count_column].to_numpy(dtype=float)
        if has_tc
        else np.zeros(n, dtype=float)
    )
    dslt = (
        df[days_since_last_touch_column].to_numpy(dtype=float)
        if has_dslt
        else np.zeros(n, dtype=float)
    )
    # Guard against NaN in touch primitives → treat as 0
    base_touches = np.nan_to_num(base_touches, nan=0.0)
    dslt = np.nan_to_num(dslt, nan=0.0)
    # Non-negative
    base_touches = np.maximum(base_touches, 0.0)
    dslt = np.maximum(dslt, 0.0)

    days_since_first_touch = base_touches * touch_interval_days + dslt
    # Clip to max_followup_days
    time = np.minimum(days_since_first_touch, max_followup_days)

    # ----- Construct event -----
    # event = 1 iff converted AND time <= max_followup_days
    # (the latter is trivially true since time was already clipped, but
    # we keep the predicate explicit for readability and future-proofing
    # against a time-origin shift)
    event = ((y == 1) & (time <= max_followup_days)).astype(int)

    # ----- Build feature matrix X -----
    cols_to_drop = {label_column}
    if drop_columns:
        cols_to_drop.update(drop_columns)
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
    feature_names = tuple(X.columns)

    if (time < 0).any():
        bad_idx = np.where(time < 0)[0][:3]
        raise ValueError(
            f"Constructed time contains negative values at indices {bad_idx.tolist()}; "
            f"this indicates corrupted input. Values: {time[bad_idx].tolist()}"
        )

    n_events = int(event.sum())
    return SurvivalFrame(
        X=X,
        time=time,
        event=event,
        feature_names=feature_names,
        n_samples=n,
        n_events=n_events,
        censoring_rate=1.0 - n_events / n if n else 0.0,
        max_followup_days=float(max_followup_days),
        touch_interval_days=float(touch_interval_days),
    )


# ---------------------------------------------------------------------------
# Sensitivity helper
# ---------------------------------------------------------------------------

def sweep_touch_interval_sensitivity(
    df: pd.DataFrame,
    touch_intervals: tuple = SENSITIVITY_TOUCH_INTERVALS,
    **kwargs,
) -> dict:
    """
    Transform df at several touch_interval_days values to support the
    honesty-bar sensitivity analysis required by the training report.

    Returns dict mapping touch_interval_days -> SurvivalFrame.
    """
    return {
        ti: transform_leadforge_to_survival(
            df, touch_interval_days=ti, **kwargs
        )
        for ti in touch_intervals
    }
