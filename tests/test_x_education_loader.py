"""
Slice 5.1 — X Education loader correctness tests.

Scope:
    - Verify the synthetic fixture produces the expected shape, schema,
      and class balance (~38.5% minority, matching real X Education).
    - Verify Tags + Lead Quality + Lead Profile + Last Notable Activity
      (the 4-column defensive leakage-trap exclusion) are excluded.
    - Verify identity + near-zero-variance columns are excluded.
    - Verify the Zenodo-unreachable fallback path returns a synthetic
      fixture with the correct source tag (environment-agnostic via
      monkeypatch — passes on Zenodo-reachable AND blocked environments).
    - Verify Select-placeholder → NaN normalization.
    - Verify Yes/No boolean → Int8 normalization.

Not in scope:
    - Verification of the real Zenodo CSV's exact column names (would
      require live network access on every test run). That is covered
      by the integration tests in test_xeducation_training.py.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinkmate.ml.x_education_loader import (
    BOOLEAN_YESNO_COLUMNS,
    CATEGORICAL_COLUMNS,
    IDENTITY_COLUMNS,
    LEAKAGE_COLUMNS,
    NEAR_ZERO_VARIANCE_COLUMNS,
    NUMERIC_COLUMNS,
    SELECT_PLACEHOLDER_VALUE,
    TARGET_COLUMN,
    XEDU_APPROX_MINORITY_RATE,
    XEDU_N_LEADS,
    XEDU_ZENODO_CSV_URL,
    XEducationDataset,
    load_x_education_lead_scoring,
    make_synthetic_xeducation_like,
    _normalize_select_placeholders,
    _normalize_yesno_booleans,
)


# ---------------------------------------------------------------------------
# Synthetic fixture correctness
# ---------------------------------------------------------------------------

class TestSyntheticFixture:
    """The synthetic X Education-shaped fixture is the primary CI backstop."""

    def test_default_shape_matches_xedu_n_leads(self):
        ds = make_synthetic_xeducation_like()
        assert len(ds.X) == XEDU_N_LEADS
        assert len(ds.y) == XEDU_N_LEADS

    def test_source_tag_is_synthetic(self):
        ds = make_synthetic_xeducation_like()
        assert ds.source == "synthetic"

    def test_minority_rate_near_target(self):
        """
        Synthetic fixture targets ~38.5% minority. Allow ±1% jitter for
        seed-induced rounding.
        """
        ds = make_synthetic_xeducation_like()
        rate = float(ds.y.mean())
        assert abs(rate - XEDU_APPROX_MINORITY_RATE) < 0.015, (
            f"Synthetic minority rate {rate:.4f} drifted from target "
            f"{XEDU_APPROX_MINORITY_RATE}"
        )

    def test_y_values_are_binary(self):
        ds = make_synthetic_xeducation_like()
        unique = set(ds.y.unique())
        assert unique == {0, 1}

    def test_no_nan_in_feature_matrix(self):
        ds = make_synthetic_xeducation_like()
        assert not ds.X.isna().any().any()

    # ------- Identity exclusion -------

    def test_identity_columns_not_in_features(self):
        ds = make_synthetic_xeducation_like()
        for col in IDENTITY_COLUMNS:
            assert col not in ds.X.columns, (
                f"Identity column {col} should not appear in feature matrix"
            )

    # ------- Leakage exclusion (THE critical tests) -------

    def test_tags_column_not_in_features(self):
        """THE critical leakage test — Tags is the known 0.98 AUC trap."""
        ds = make_synthetic_xeducation_like()
        assert "Tags" not in ds.X.columns
        # Also ensure no one-hot of it leaked in
        assert not any(c.startswith("Tags_") for c in ds.X.columns)

    def test_lead_quality_not_in_features(self):
        ds = make_synthetic_xeducation_like()
        assert "Lead Quality" not in ds.X.columns
        assert not any(c.startswith("Lead Quality_") for c in ds.X.columns)

    def test_lead_profile_not_in_features(self):
        ds = make_synthetic_xeducation_like()
        assert "Lead Profile" not in ds.X.columns
        assert not any(c.startswith("Lead Profile_") for c in ds.X.columns)

    def test_last_notable_activity_not_in_features(self):
        ds = make_synthetic_xeducation_like()
        assert "Last Notable Activity" not in ds.X.columns
        assert not any(c.startswith("Last Notable Activity_") for c in ds.X.columns)

    # ------- Near-zero-variance exclusion -------

    def test_near_zero_variance_columns_dropped(self):
        ds = make_synthetic_xeducation_like()
        for col in NEAR_ZERO_VARIANCE_COLUMNS:
            assert col not in ds.X.columns, (
                f"Near-zero-variance column {col} should be dropped"
            )

    # ------- Target exclusion -------

    def test_target_column_not_in_features(self):
        ds = make_synthetic_xeducation_like()
        assert TARGET_COLUMN not in ds.X.columns

    # ------- Encoding correctness -------

    def test_categoricals_one_hot_encoded(self):
        """Raw categorical column names should not appear; one-hot columns
        with the categorical name as prefix should be present."""
        ds = make_synthetic_xeducation_like()
        for raw_col in CATEGORICAL_COLUMNS:
            assert raw_col not in ds.X.columns
        # At least one Lead Origin one-hot should exist
        lead_origin_cols = [c for c in ds.X.columns if c.startswith("Lead Origin_")]
        assert len(lead_origin_cols) >= 2, (
            f"Expected at least 2 'Lead Origin_*' one-hot columns, got {lead_origin_cols}"
        )

    def test_yesno_normalized_to_int(self):
        """Yes/No columns should be Int8 (0/1), not strings."""
        ds = make_synthetic_xeducation_like()
        for col in BOOLEAN_YESNO_COLUMNS:
            assert col in ds.X.columns, f"Boolean column {col} missing"
            # Values should be 0 or 1 (or NaN-coded, but synthetic has none)
            unique_vals = set(ds.X[col].dropna().unique())
            assert unique_vals.issubset({0, 1}), (
                f"Column {col} has non-binary values: {unique_vals}"
            )

    def test_numeric_columns_present(self):
        ds = make_synthetic_xeducation_like()
        for col in NUMERIC_COLUMNS:
            assert col in ds.X.columns, f"Numeric column {col} missing"

    # ------- SHA-256 determinism -------

    def test_sha256_deterministic(self):
        ds1 = make_synthetic_xeducation_like(seed=42)
        ds2 = make_synthetic_xeducation_like(seed=42)
        assert ds1.sha256 == ds2.sha256

    def test_sha256_differs_across_seeds(self):
        ds1 = make_synthetic_xeducation_like(seed=42)
        ds2 = make_synthetic_xeducation_like(seed=123)
        assert ds1.sha256 != ds2.sha256

    # ------- Feature-name consistency -------

    def test_feature_names_match_columns(self):
        ds = make_synthetic_xeducation_like()
        assert ds.feature_names == tuple(ds.X.columns)


# ---------------------------------------------------------------------------
# Zenodo-fallback behaviour (unreachable → synthetic) — environment-agnostic
# ---------------------------------------------------------------------------

class TestZenodoFallback:
    """
    When Zenodo is unreachable, the loader should fall back to synthetic
    if fallback_to_synthetic=True. Uses monkeypatch to simulate failure
    regardless of actual network state — same pattern as Slice 5's
    TestHFFallback that the Windows-environment-agnostic fix brought in.
    """

    def test_fallback_returns_synthetic_dataset(self, monkeypatch, tmp_path):
        """Simulated Zenodo failure → loader returns synthetic."""
        import pinkmate.ml.x_education_loader as xedu_mod

        def _raise_connection(*args, **kwargs):
            raise urllib.error.URLError("Simulated Zenodo unreachable")

        monkeypatch.setattr(xedu_mod, "_fetch_zenodo_csv", _raise_connection)

        ds = load_x_education_lead_scoring(
            cache_dir=tmp_path / "no_cache",  # Force network path
            allow_cache=False,
            fallback_to_synthetic=True,
        )

        assert ds.source == "synthetic", (
            f"Expected synthetic fallback when Zenodo is simulated unreachable; "
            f"got {ds.source}"
        )
        assert isinstance(ds, XEducationDataset)
        assert len(ds.X) > 0

    def test_fallback_disabled_raises(self, monkeypatch, tmp_path):
        """Without fallback_to_synthetic=True, Zenodo failure raises."""
        import pinkmate.ml.x_education_loader as xedu_mod

        def _raise_connection(*args, **kwargs):
            raise urllib.error.URLError("Simulated Zenodo unreachable")

        monkeypatch.setattr(xedu_mod, "_fetch_zenodo_csv", _raise_connection)

        with pytest.raises(urllib.error.URLError):
            load_x_education_lead_scoring(
                cache_dir=tmp_path / "no_cache",
                allow_cache=False,
                fallback_to_synthetic=False,
            )


# ---------------------------------------------------------------------------
# Leakage exclusion — explicit regression tests
# ---------------------------------------------------------------------------

class TestLeakageExclusion:
    """
    Explicit regression tests for the 4-column defensive leakage-trap
    exclusion. If any of these fail, the methodology-integrity test in
    test_xeducation_training.py will also fail (AUC > 0.95), but these
    give faster + more targeted failure signals.
    """

    def test_all_four_leakage_columns_defined(self):
        """Compile-time guard against accidental removal of a leakage column."""
        expected = {"Tags", "Lead Quality", "Lead Profile", "Last Notable Activity"}
        assert set(LEAKAGE_COLUMNS) == expected, (
            f"LEAKAGE_COLUMNS changed. Expected {expected}, got {set(LEAKAGE_COLUMNS)}. "
            "If you intentionally changed the leakage exclusion list, update this test."
        )

    def test_leakage_columns_dropped_on_synthetic(self):
        """None of the 4 leakage columns should appear in the encoded features."""
        ds = make_synthetic_xeducation_like()
        for col in LEAKAGE_COLUMNS:
            assert col not in ds.X.columns, (
                f"Leakage column {col} must be excluded from features"
            )
            assert not any(c.startswith(f"{col}_") for c in ds.X.columns), (
                f"One-hot encoding of leakage column {col} must not appear"
            )


# ---------------------------------------------------------------------------
# Normalization unit tests
# ---------------------------------------------------------------------------

class TestNormalizations:
    def test_select_placeholder_becomes_nan(self):
        df = pd.DataFrame({
            "Specialization": ["Finance Management", SELECT_PLACEHOLDER_VALUE,
                               "Marketing Management"],
            "City": [SELECT_PLACEHOLDER_VALUE, "Mumbai", SELECT_PLACEHOLDER_VALUE],
            "TotalVisits": [5, 10, 15],
        })
        result = _normalize_select_placeholders(df)
        assert pd.isna(result.loc[1, "Specialization"])
        assert pd.isna(result.loc[0, "City"])
        assert pd.isna(result.loc[2, "City"])
        # Non-Select values preserved
        assert result.loc[0, "Specialization"] == "Finance Management"
        # Numeric columns unchanged (object-dtype check should skip them)
        assert result.loc[0, "TotalVisits"] == 5

    def test_yesno_normalized_to_int(self):
        df = pd.DataFrame({
            "Do Not Email": ["Yes", "No", "Yes"],
            "Do Not Call": ["No", "Yes", "No"],
            "Search": ["Yes", "Yes", "No"],
            # Non-Yes/No column should be left alone
            "Specialization": ["Finance", "HR", "Marketing"],
        })
        result = _normalize_yesno_booleans(df)
        assert result["Do Not Email"].tolist() == [1, 0, 1]
        assert result["Do Not Call"].tolist() == [0, 1, 0]
        assert result["Search"].tolist() == [1, 1, 0]
        assert result["Do Not Email"].dtype == "Int8"
        # Non-Yes/No column unchanged
        assert result["Specialization"].tolist() == ["Finance", "HR", "Marketing"]


# ---------------------------------------------------------------------------
# URL encoding sanity
# ---------------------------------------------------------------------------

class TestZenodoURL:
    def test_url_is_percent_encoded(self):
        """
        Filename 'Lead Scoring (1).csv' needs URL encoding — spaces as %20,
        parens as %28 and %29.
        """
        assert "%20" in XEDU_ZENODO_CSV_URL, "URL should encode space as %20"
        assert "%28" in XEDU_ZENODO_CSV_URL, "URL should encode '(' as %28"
        assert "%29" in XEDU_ZENODO_CSV_URL, "URL should encode ')' as %29"
        assert XEDU_ZENODO_CSV_URL.startswith("https://zenodo.org/records/23791277/files/")
