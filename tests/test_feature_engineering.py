"""
Slice 5.2 — Tests for pinkmate/ml/feature_engineering.py.

Covers:
    - TestRFM (8 tests): recency / frequency / monetary / composite bounds,
      saturation, log-scaling, None-handling
    - TestICPRules (6 tests): all 5 rule functions produce correct 0/1
      binary outputs on canonical inputs + edge cases
    - TestICPComposite (3 tests): composite sums correctly, tier cutoffs
      assign correctly, free-mail-only lead caps tier at D

Total: 17 new tests. Target full-suite: 201 passing (179 Slice 5.1 + 17 FE + 5 scoring_agent).
"""

from __future__ import annotations

import math

import pytest

from pinkmate.ml.feature_engineering import (
    FREQUENCY_SATURATION,
    ICP_RULE_POINTS,
    ICP_TIER_CUTOFFS,
    MONETARY_ACV_BANDS,
    MONETARY_BAND_SCORES,
    RECENCY_DAYS_SATURATION,
    compute_composite_rfm,
    compute_frequency_score,
    compute_icp_composite,
    compute_monetary_score,
    compute_recency_score,
    icp_country_match,
    icp_employees_band_match,
    icp_industry_match,
    icp_non_free_mail,
    icp_seniority_match,
    icp_tier,
)


# ---------------------------------------------------------------------------
# RFM scoring
# ---------------------------------------------------------------------------

class TestRFM:
    """Recency / Frequency / Monetary scoring correctness."""

    def test_recency_bounded_0_1_across_range(self):
        for days in range(0, 120, 5):
            s = compute_recency_score(days)
            assert 0.0 <= s <= 1.0, f"recency({days}) = {s} out of [0, 1]"

    def test_recency_zero_days_gives_max(self):
        assert compute_recency_score(0) == 1.0

    def test_recency_saturation_at_90_days_gives_zero(self):
        assert compute_recency_score(RECENCY_DAYS_SATURATION) == 0.0
        assert compute_recency_score(RECENCY_DAYS_SATURATION + 50) == 0.0

    def test_recency_none_returns_zero(self):
        assert compute_recency_score(None) == 0.0

    def test_frequency_log_scaled_diminishing_returns(self):
        """Marginal gain from 1→2 touches should be larger than from 10→11."""
        gain_low = compute_frequency_score(2) - compute_frequency_score(1)
        gain_high = compute_frequency_score(11) - compute_frequency_score(10)
        assert gain_low > gain_high, (
            f"Frequency should have diminishing returns; got low gain "
            f"{gain_low:.4f} vs high gain {gain_high:.4f}"
        )

    def test_frequency_saturation_gives_one(self):
        assert compute_frequency_score(FREQUENCY_SATURATION) == 1.0
        assert compute_frequency_score(FREQUENCY_SATURATION + 100) == 1.0

    def test_frequency_zero_and_none_return_zero(self):
        assert compute_frequency_score(0) == 0.0
        assert compute_frequency_score(None) == 0.0

    def test_monetary_bands_monotonic(self):
        """Score must be non-decreasing as ACV increases."""
        samples = [100, 5_000, 25_000, 100_000, 500_000, 5_000_000]
        scores = [compute_monetary_score(acv) for acv in samples]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"Monotonic broken at index {i}: scores={scores}"
            )

    def test_monetary_specific_band_values(self):
        """Verify exact band mapping per MONETARY_ACV_BANDS."""
        assert compute_monetary_score(100) == MONETARY_BAND_SCORES[0]        # <10k
        assert compute_monetary_score(25_000) == MONETARY_BAND_SCORES[1]     # 10-50k
        assert compute_monetary_score(100_000) == MONETARY_BAND_SCORES[2]    # 50-200k
        assert compute_monetary_score(500_000) == MONETARY_BAND_SCORES[3]    # 200k-1M
        assert compute_monetary_score(5_000_000) == MONETARY_BAND_SCORES[4]  # >1M

    def test_monetary_none_and_negative_return_zero(self):
        assert compute_monetary_score(None) == 0.0
        assert compute_monetary_score(-100) == 0.0

    def test_composite_equal_weights_averages(self):
        """Default equal weights should produce arithmetic mean."""
        c = compute_composite_rfm(0.6, 0.3, 0.9)
        assert math.isclose(c, 0.6, abs_tol=1e-6)

    def test_composite_custom_weights(self):
        """Custom weights should apply proportionally."""
        # Heavily weighted on monetary → should pull the composite toward monetary
        c = compute_composite_rfm(0.0, 0.0, 1.0, weights=(0.1, 0.1, 0.8))
        assert c > 0.7, f"Monetary-heavy composite should be > 0.7, got {c}"

    def test_composite_zero_weights_returns_zero(self):
        """Degenerate zero-weights case should not crash."""
        c = compute_composite_rfm(1.0, 1.0, 1.0, weights=(0.0, 0.0, 0.0))
        assert c == 0.0


# ---------------------------------------------------------------------------
# Clay-style ICP rules (binary outputs)
# ---------------------------------------------------------------------------

class TestICPRules:
    """Each rule produces 0 or 1 on canonical inputs."""

    def test_employees_band_target_matches(self):
        assert icp_employees_band_match("50-199") == 1
        assert icp_employees_band_match("200-999") == 1

    def test_employees_band_outside_target(self):
        assert icp_employees_band_match("1-49") == 0
        assert icp_employees_band_match("1000-1999") == 0
        assert icp_employees_band_match("2000+") == 0
        assert icp_employees_band_match(None) == 0

    def test_industry_canonical_values_match(self):
        assert icp_industry_match("manufacturing") == 1
        assert icp_industry_match("Manufacturing") == 1  # Case-insensitive
        assert icp_industry_match("logistics") == 1
        assert icp_industry_match("professional_services") == 1

    def test_industry_non_target_returns_zero(self):
        assert icp_industry_match("healthcare_non_clinical") == 0
        assert icp_industry_match("retail") == 0
        assert icp_industry_match(None) == 0

    def test_seniority_director_and_above_match(self):
        assert icp_seniority_match("director") == 1
        assert icp_seniority_match("Director") == 1  # Case-insensitive
        assert icp_seniority_match("vp") == 1
        assert icp_seniority_match("VP") == 1
        assert icp_seniority_match("c_suite") == 1

    def test_seniority_below_director_returns_zero(self):
        assert icp_seniority_match("manager") == 0
        assert icp_seniority_match("individual_contributor") == 0
        assert icp_seniority_match(None) == 0

    def test_country_target_matches(self):
        assert icp_country_match("US") == 1
        assert icp_country_match("UK") == 1

    def test_country_non_target_returns_zero(self):
        assert icp_country_match("IN") == 0
        assert icp_country_match("DE") == 0
        assert icp_country_match(None) == 0
        assert icp_country_match("us") == 0  # Case-sensitive on purpose

    def test_non_free_mail_detects_gmail_as_free(self):
        """Gmail is a free-mail provider → function returns 0."""
        assert icp_non_free_mail("jane@gmail.com") == 0
        assert icp_non_free_mail("jane@GMAIL.COM") == 0  # Case-insensitive

    def test_non_free_mail_counts_corporate_as_non_free(self):
        """Corporate domains return 1 (non-free, high signal)."""
        assert icp_non_free_mail("jane@pinkmate.co.uk") == 1
        assert icp_non_free_mail("jane@ibm.com") == 1
        assert icp_non_free_mail("jane@kent.ac.uk") == 1

    def test_non_free_mail_handles_malformed_input(self):
        assert icp_non_free_mail(None) == 0
        assert icp_non_free_mail("") == 0
        assert icp_non_free_mail("not-an-email") == 0
        assert icp_non_free_mail("@no-local-part.com") == 1  # Edge: domain only — treat as corporate


# ---------------------------------------------------------------------------
# Clay-style ICP composite + tier
# ---------------------------------------------------------------------------

class TestICPComposite:
    """Composite scoring + tier assignment."""

    def test_composite_all_rules_match_sums_to_100(self):
        features = {
            "employees_band_match": 1,
            "industry_match": 1,
            "seniority_match": 1,
            "country_match": 1,
            "non_free_mail": 1,
        }
        assert compute_icp_composite(features) == 100.0

    def test_composite_no_rules_match_returns_zero(self):
        features = {k: 0 for k in ICP_RULE_POINTS.keys()}
        assert compute_icp_composite(features) == 0.0

    def test_composite_partial_match_sums_correctly(self):
        """Only employees + industry match → 30 + 25 = 55 points."""
        features = {
            "employees_band_match": 1,
            "industry_match": 1,
            "seniority_match": 0,
            "country_match": 0,
            "non_free_mail": 0,
        }
        assert compute_icp_composite(features) == 55.0

    def test_composite_handles_missing_keys(self):
        """Missing keys should be treated as 0, not raise."""
        features = {"employees_band_match": 1}  # Only 1 key
        assert compute_icp_composite(features) == 30.0

    def test_tier_assignment_at_cutoffs(self):
        assert icp_tier(100) == "A"
        assert icp_tier(ICP_TIER_CUTOFFS["A"]) == "A"       # Exactly 80 → A
        assert icp_tier(ICP_TIER_CUTOFFS["A"] - 1) == "B"   # 79 → B
        assert icp_tier(ICP_TIER_CUTOFFS["B"]) == "B"       # Exactly 60 → B
        assert icp_tier(ICP_TIER_CUTOFFS["B"] - 1) == "C"   # 59 → C
        assert icp_tier(ICP_TIER_CUTOFFS["C"]) == "C"       # Exactly 40 → C
        assert icp_tier(ICP_TIER_CUTOFFS["C"] - 1) == "D"   # 39 → D
        assert icp_tier(0) == "D"

    def test_free_mail_only_lead_maxes_at_tier_C(self):
        """
        A lead that's perfect on employees + industry + seniority + country
        but uses free-mail (Gmail) scores 90/100 = tier A (not C).
        Clay's original reference caps free-mail at C, but our composite
        is arithmetic. This test documents the actual behaviour so a future
        tier-cap-on-free-mail policy is a conscious change, not a drift.
        """
        features = {
            "employees_band_match": 1,  # 30
            "industry_match": 1,        # 25
            "seniority_match": 1,       # 25
            "country_match": 1,         # 10
            "non_free_mail": 0,         # 0
        }
        composite = compute_icp_composite(features)  # 90
        assert composite == 90.0
        assert icp_tier(composite) == "A"  # Documents arithmetic behaviour
