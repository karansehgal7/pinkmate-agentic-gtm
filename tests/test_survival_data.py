"""
Tests for pinkmate.ml.survival_data — LeadForge → survival schema transform.

Enforces:
    - Deterministic transformation (no randomness)
    - time construction formula (Decision B=3)
    - 90-day right-censoring correctness
    - Edge cases (zero touches, missing columns, NaN handling)
    - Sensitivity to touch_interval_days (verdict-stability range 5/7/14)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pinkmate.ml.survival_data import (
    DEFAULT_MAX_FOLLOWUP_DAYS,
    DEFAULT_TOUCH_INTERVAL_DAYS,
    SENSITIVITY_TOUCH_INTERVALS,
    SurvivalFrame,
    sweep_touch_interval_sensitivity,
    transform_leadforge_to_survival,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_leadforge_df():
    """Minimal 5-row LeadForge-like dataframe."""
    return pd.DataFrame({
        "touch_count": [0, 3, 10, 5, 15],
        "days_since_last_touch": [0.0, 5.0, 30.0, 14.0, 7.0],
        "industry_saas": [1, 0, 1, 1, 0],
        "funding_stage_series_a": [0, 1, 0, 1, 1],
        "converted": [0, 1, 0, 1, 1],
    })


@pytest.fixture
def larger_leadforge_df():
    """200-row synthetic LeadForge for robustness tests."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "touch_count": rng.integers(0, 20, n),
        "days_since_last_touch": rng.uniform(0, 30, n),
        "industry_saas": rng.integers(0, 2, n),
        "funding_stage_series_a": rng.integers(0, 2, n),
        "role_seniority_score": rng.uniform(0, 1, n),
        "converted": rng.integers(0, 2, n),
    })


# ===========================================================================
# TestTimeConstruction
# ===========================================================================

class TestTimeConstruction:
    def test_formula_matches_decision_b(self, minimal_leadforge_df):
        """time = min(touch_count * touch_interval + days_since_last_touch, max_followup)."""
        sf = transform_leadforge_to_survival(
            minimal_leadforge_df, touch_interval_days=7.0, max_followup_days=90.0
        )
        expected = np.minimum(
            minimal_leadforge_df["touch_count"].to_numpy(float) * 7.0
            + minimal_leadforge_df["days_since_last_touch"].to_numpy(float),
            90.0,
        )
        np.testing.assert_allclose(sf.time, expected)

    def test_time_non_negative(self, larger_leadforge_df):
        """Constructed time is always >= 0."""
        sf = transform_leadforge_to_survival(larger_leadforge_df)
        assert (sf.time >= 0).all()

    def test_time_bounded_by_max_followup(self, minimal_leadforge_df):
        """time never exceeds max_followup_days."""
        sf = transform_leadforge_to_survival(
            minimal_leadforge_df, max_followup_days=30.0
        )
        assert (sf.time <= 30.0 + 1e-9).all()

    def test_touch_interval_affects_time_linearly(self, minimal_leadforge_df):
        """Doubling touch_interval_days roughly doubles the touch-contribution."""
        sf7 = transform_leadforge_to_survival(
            minimal_leadforge_df, touch_interval_days=7.0, max_followup_days=1000.0
        )
        sf14 = transform_leadforge_to_survival(
            minimal_leadforge_df, touch_interval_days=14.0, max_followup_days=1000.0
        )
        touch = minimal_leadforge_df["touch_count"].to_numpy(float)
        dslt = minimal_leadforge_df["days_since_last_touch"].to_numpy(float)
        diff = sf14.time - sf7.time
        expected_diff = touch * 7.0  # (14 - 7) × touch_count
        np.testing.assert_allclose(diff, expected_diff)


# ===========================================================================
# TestCensoring
# ===========================================================================

class TestCensoring:
    def test_event_one_when_converted_within_window(self, minimal_leadforge_df):
        """event == 1 iff converted == 1 AND time <= max_followup."""
        sf = transform_leadforge_to_survival(
            minimal_leadforge_df, max_followup_days=90.0
        )
        for i in range(len(minimal_leadforge_df)):
            converted = minimal_leadforge_df["converted"].iloc[i] == 1
            within_window = sf.time[i] <= 90.0
            expected = int(converted and within_window)
            assert sf.event[i] == expected, (
                f"Row {i}: converted={converted}, time={sf.time[i]}, "
                f"expected event={expected}, got {sf.event[i]}"
            )

    def test_censoring_rate_computed_correctly(self, minimal_leadforge_df):
        """censoring_rate = 1 - n_events/n_samples."""
        sf = transform_leadforge_to_survival(minimal_leadforge_df)
        expected = 1.0 - sf.n_events / sf.n_samples
        assert sf.censoring_rate == pytest.approx(expected)

    def test_censoring_rate_bounded_zero_one(self, larger_leadforge_df):
        """censoring_rate ∈ [0, 1]."""
        sf = transform_leadforge_to_survival(larger_leadforge_df)
        assert 0.0 <= sf.censoring_rate <= 1.0


# ===========================================================================
# TestDeterminism
# ===========================================================================

class TestDeterminism:
    def test_repeated_call_identical_output(self, larger_leadforge_df):
        """Transform is pure — same input twice gives identical output."""
        sf1 = transform_leadforge_to_survival(larger_leadforge_df)
        sf2 = transform_leadforge_to_survival(larger_leadforge_df)
        np.testing.assert_array_equal(sf1.time, sf2.time)
        np.testing.assert_array_equal(sf1.event, sf2.event)
        assert sf1.n_samples == sf2.n_samples
        assert sf1.n_events == sf2.n_events

    def test_no_hidden_randomness_through_different_rngs(self, larger_leadforge_df):
        """Calling under different numpy RNG states still produces same output."""
        np.random.seed(0)
        sf1 = transform_leadforge_to_survival(larger_leadforge_df)
        np.random.seed(999)
        sf2 = transform_leadforge_to_survival(larger_leadforge_df)
        np.testing.assert_array_equal(sf1.time, sf2.time)


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    def test_zero_touches_and_zero_dslt_gives_time_zero(self):
        df = pd.DataFrame({
            "touch_count": [0, 0],
            "days_since_last_touch": [0.0, 0.0],
            "converted": [0, 1],
        })
        sf = transform_leadforge_to_survival(df)
        np.testing.assert_array_equal(sf.time, [0.0, 0.0])

    def test_nan_touch_count_treated_as_zero(self):
        df = pd.DataFrame({
            "touch_count": [np.nan, 5.0],
            "days_since_last_touch": [10.0, 20.0],
            "converted": [1, 0],
        })
        sf = transform_leadforge_to_survival(df)
        # First row's touch_count is NaN → treated as 0 → time = 10
        assert sf.time[0] == pytest.approx(10.0)

    def test_nan_dslt_treated_as_zero(self):
        df = pd.DataFrame({
            "touch_count": [3, 0],
            "days_since_last_touch": [np.nan, 5.0],
            "converted": [0, 1],
        })
        sf = transform_leadforge_to_survival(df, touch_interval_days=7.0)
        # First row: dslt NaN → 0 → time = 3*7 = 21
        assert sf.time[0] == pytest.approx(21.0)

    def test_missing_label_column_raises(self):
        df = pd.DataFrame({"touch_count": [1], "days_since_last_touch": [0.0]})
        with pytest.raises(ValueError, match="label_column"):
            transform_leadforge_to_survival(df)

    def test_missing_both_touch_columns_raises(self):
        df = pd.DataFrame({"converted": [1, 0], "industry": ["saas", "fintech"]})
        with pytest.raises(ValueError, match="At least one"):
            transform_leadforge_to_survival(df)

    def test_invalid_label_values_raises(self):
        df = pd.DataFrame({
            "touch_count": [1, 2],
            "days_since_last_touch": [1.0, 2.0],
            "converted": [0, 3],  # 3 is invalid
        })
        with pytest.raises(ValueError, match="only"):
            transform_leadforge_to_survival(df)

    def test_invalid_max_followup_raises(self, minimal_leadforge_df):
        with pytest.raises(ValueError, match="max_followup_days"):
            transform_leadforge_to_survival(minimal_leadforge_df, max_followup_days=-1)

    def test_invalid_touch_interval_raises(self, minimal_leadforge_df):
        with pytest.raises(ValueError, match="touch_interval_days"):
            transform_leadforge_to_survival(minimal_leadforge_df, touch_interval_days=0)


# ===========================================================================
# TestSensitivityToTouchInterval
# ===========================================================================

class TestSensitivityToTouchInterval:
    def test_sweep_returns_all_intervals(self, larger_leadforge_df):
        """sweep_touch_interval_sensitivity returns one SurvivalFrame per interval."""
        results = sweep_touch_interval_sensitivity(larger_leadforge_df)
        for interval in SENSITIVITY_TOUCH_INTERVALS:
            assert interval in results
            assert isinstance(results[interval], SurvivalFrame)

    def test_sweep_times_differ_across_intervals(self, larger_leadforge_df):
        """Different touch_intervals yield different times (sanity)."""
        results = sweep_touch_interval_sensitivity(larger_leadforge_df)
        t5 = results[5.0].time
        t14 = results[14.0].time
        # At least some rows differ (those with touch_count > 0)
        assert not np.array_equal(t5, t14)

    def test_sensitivity_intervals_include_canonical_7(self):
        """7.0 (default) must be in the sensitivity sweep."""
        assert 7.0 in SENSITIVITY_TOUCH_INTERVALS
        assert DEFAULT_TOUCH_INTERVAL_DAYS == 7.0


# ===========================================================================
# TestSurvivalFrame
# ===========================================================================

class TestSurvivalFrame:
    def test_feature_names_exclude_label(self, minimal_leadforge_df):
        """X columns must not contain the label_column."""
        sf = transform_leadforge_to_survival(minimal_leadforge_df)
        assert "converted" not in sf.feature_names

    def test_drop_columns_applied(self, minimal_leadforge_df):
        """drop_columns are removed from X."""
        sf = transform_leadforge_to_survival(
            minimal_leadforge_df, drop_columns=("industry_saas",)
        )
        assert "industry_saas" not in sf.feature_names

    def test_summary_contains_key_metrics(self, minimal_leadforge_df):
        sf = transform_leadforge_to_survival(minimal_leadforge_df)
        s = sf.summary()
        assert "n=5" in s
        assert "max_followup=90" in s
