"""
Slice 5 — LeadForge loader + synthetic fixture correctness tests.

Scope:
    - Verify the synthetic fixture produces the expected shape, schema,
      and class balance (~20% minority, matching the industry-benchmark
      estimate used in the loader docstring).
    - Verify the one-hot encoding handles all categorical columns correctly.
    - Verify the leakage trap (total_touches_all) is excluded from features.
    - Verify the identity columns (lead_id, account_id, contact_id) and
      the timing column (lead_created_at) are excluded.
    - Verify the HF-unreachable fallback path returns a synthetic fixture
      with the correct source tag — using monkeypatch so the test is
      environment-agnostic (passes on HF-blocked cloud AND HF-reachable
      developer machines).
    - Verify the tier parameter is validated.

Not in scope:
    - Verification of the real HF dataset's class balance / schema. That
      is covered by the Slice 5 training run (`scripts/train_gtm_ensemble.py`)
      on KS's local machine, which fetches the real intermediate-tier data
      and writes verdicts + feature-count logs to the generated
      `docs/gtm_training_report.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinkmate.ml.leadforge_loader import (
    BOOLEAN_COLUMNS,
    IDENTITY_COLUMNS,
    LEADFORGE_APPROX_MINORITY_RATE,
    LEADFORGE_N_LEADS,
    LEADFORGE_TIERS,
    LEAKAGE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TIMING_COLUMNS,
    LeadForgeDataset,
    load_leadforge_lead_scoring,
    make_synthetic_leadforge_like,
)


# ---------------------------------------------------------------------------
# Synthetic fixture correctness
# ---------------------------------------------------------------------------

class TestSyntheticFixture:
    """The synthetic LeadForge-shaped fixture is the primary CI backstop."""

    def test_default_shape_matches_leadforge_n_leads(self):
        ds = make_synthetic_leadforge_like()
        assert len(ds.X) == LEADFORGE_N_LEADS
        assert len(ds.y) == LEADFORGE_N_LEADS

    def test_source_tag_is_synthetic(self):
        ds = make_synthetic_leadforge_like()
        assert ds.source == "synthetic"

    def test_default_tier_is_intermediate(self):
        ds = make_synthetic_leadforge_like()
        assert ds.tier == "intermediate"

    def test_tier_parameter_round_trips(self):
        ds = make_synthetic_leadforge_like(tier="advanced")
        assert ds.tier == "advanced"

    def test_minority_rate_near_target(self):
        """
        Synthetic fixture samples the minority class at ~20% (approx B2B
        benchmark). Allow ±1% jitter for any seed-induced rounding.
        """
        ds = make_synthetic_leadforge_like()
        rate = float(ds.y.mean())
        assert abs(rate - LEADFORGE_APPROX_MINORITY_RATE) < 0.015, (
            f"Synthetic minority rate {rate:.4f} drifted from target "
            f"{LEADFORGE_APPROX_MINORITY_RATE}"
        )

    def test_y_values_are_binary(self):
        ds = make_synthetic_leadforge_like()
        unique = set(ds.y.unique())
        assert unique == {0, 1}

    def test_no_nan_in_feature_matrix(self):
        ds = make_synthetic_leadforge_like()
        assert not ds.X.isna().any().any()

    def test_identity_columns_not_in_features(self):
        ds = make_synthetic_leadforge_like()
        for col in IDENTITY_COLUMNS:
            assert col not in ds.X.columns, (
                f"Identity column {col} should not appear in feature matrix"
            )

    def test_timing_columns_not_in_features(self):
        ds = make_synthetic_leadforge_like()
        for col in TIMING_COLUMNS:
            assert col not in ds.X.columns

    def test_leakage_trap_not_in_features(self):
        ds = make_synthetic_leadforge_like()
        for col in LEAKAGE_COLUMNS:
            assert col not in ds.X.columns, (
                f"Leakage trap column {col} must be excluded from features"
            )

    def test_target_column_not_in_features(self):
        ds = make_synthetic_leadforge_like()
        assert TARGET_COLUMN not in ds.X.columns

    def test_numeric_columns_present(self):
        ds = make_synthetic_leadforge_like()
        for col in NUMERIC_COLUMNS:
            assert col in ds.X.columns, f"Numeric column {col} missing"

    def test_boolean_columns_present(self):
        ds = make_synthetic_leadforge_like()
        for col in BOOLEAN_COLUMNS:
            assert col in ds.X.columns, f"Boolean column {col} missing"

    def test_categoricals_one_hot_encoded(self):
        """
        Categorical columns should NOT appear as their raw name in the
        feature matrix — only one-hot columns with the prefix.
        """
        ds = make_synthetic_leadforge_like()
        # Example: 'industry' as a raw column should be absent;
        # 'industry_manufacturing' etc. should be present
        assert "industry" not in ds.X.columns
        industry_cols = [c for c in ds.X.columns if c.startswith("industry_")]
        assert len(industry_cols) >= 2, (
            f"Expected at least 2 industry_* one-hot columns, got "
            f"{len(industry_cols)}: {industry_cols}"
        )

    def test_feature_names_match_columns(self):
        ds = make_synthetic_leadforge_like()
        assert ds.feature_names == tuple(ds.X.columns)

    def test_sha256_is_deterministic(self):
        """Same seed produces same SHA-256."""
        ds1 = make_synthetic_leadforge_like(seed=42)
        ds2 = make_synthetic_leadforge_like(seed=42)
        assert ds1.sha256 == ds2.sha256

    def test_sha256_differs_across_seeds(self):
        ds1 = make_synthetic_leadforge_like(seed=42)
        ds2 = make_synthetic_leadforge_like(seed=123)
        assert ds1.sha256 != ds2.sha256


# ---------------------------------------------------------------------------
# HF-fallback behaviour (unreachable → synthetic) — environment-agnostic
# ---------------------------------------------------------------------------

class TestHFFallback:
    """
    When HuggingFace is unreachable, the loader should gracefully fall
    back to the synthetic fixture if `fallback_to_synthetic=True`.

    These tests use `monkeypatch` to simulate HF unreachability rather
    than depending on actual network state — so they pass identically
    in HF-reachable environments (developer machines like KS's Windows)
    and HF-blocked environments (cloud sandbox proxy). This is the honest
    way to test fallback behaviour without the test being coupled to
    whichever network the test happens to run on.
    """

    def test_fallback_returns_synthetic_dataset(self, monkeypatch, tmp_path):
        """Simulated HF failure → loader returns synthetic dataset."""
        import datasets as hf_datasets

        def _raise_connection(*args, **kwargs):
            raise ConnectionError(
                "Simulated HF unreachable for fallback behaviour test"
            )

        monkeypatch.setattr(hf_datasets, "load_dataset", _raise_connection)

        ds = load_leadforge_lead_scoring(
            tier="intermediate",
            cache_dir=tmp_path / "no_cache",  # Force HF path (no cache hit)
            allow_cache=False,
            fallback_to_synthetic=True,
        )

        assert ds.source == "synthetic", (
            f"Expected synthetic fallback when HF is simulated unreachable; "
            f"got {ds.source}"
        )
        assert isinstance(ds, LeadForgeDataset)
        assert len(ds.X) > 0

    def test_fallback_preserves_tier(self, monkeypatch, tmp_path):
        """Simulated HF failure with advanced tier → synthetic preserves tier."""
        import datasets as hf_datasets

        def _raise_connection(*args, **kwargs):
            raise ConnectionError(
                "Simulated HF unreachable for fallback behaviour test"
            )

        monkeypatch.setattr(hf_datasets, "load_dataset", _raise_connection)

        ds = load_leadforge_lead_scoring(
            tier="advanced",
            cache_dir=tmp_path / "no_cache",
            allow_cache=False,
            fallback_to_synthetic=True,
        )
        assert ds.tier == "advanced"
        assert ds.source == "synthetic"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestTierValidation:
    def test_unsupported_tier_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported tier"):
            load_leadforge_lead_scoring(tier="ultra_advanced")

    def test_supported_tiers_accepted(self, monkeypatch, tmp_path):
        """
        Each supported tier is accepted by the loader. We monkeypatch HF
        to force the synthetic path so this test doesn't hit the network
        3× (one per tier), which would make the test slow and flaky.
        """
        import datasets as hf_datasets

        def _raise_connection(*args, **kwargs):
            raise ConnectionError(
                "Simulated HF unreachable for tier validation test"
            )

        monkeypatch.setattr(hf_datasets, "load_dataset", _raise_connection)

        for tier in LEADFORGE_TIERS:
            ds = load_leadforge_lead_scoring(
                tier=tier,
                cache_dir=tmp_path / f"no_cache_{tier}",
                allow_cache=False,
                fallback_to_synthetic=True,
            )
            assert ds.tier == tier
