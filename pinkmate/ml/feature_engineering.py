"""
Slice 5.2 — Deployment-schema feature engineering layer.

Pure RFM (Recency / Frequency / Monetary) functions + Clay-style ICP
(Ideal Customer Profile) scoring rules. Decision B lock: pure functions,
minimal machinery, maximum testability. No state, no configuration
objects, no side effects — every function is deterministic given its
inputs and returns a well-bounded primitive.

Scope:
    - RFM primitives derived from LeadForge-style touch data:
      days_since_last_touch → recency, touch_count → frequency,
      opportunity_estimated_acv → monetary
    - Clay-style ICP rules derived from the AazanM/clay-hubspot-enrichment
      reference (Phase 2 research finding, 9 Nov 2026): employees-band ×
      industry × seniority × country × non-free-mail scoring → A/B/C/D tier

Decision C lock (16 Nov 2026): BOTH rule-based binary features AND
composite score + tier are exposed to the model. Redundancy is cheap;
regularisation handles whichever is less useful; interpretability wins.

Known limitations (documented, not blocking):
    - Clay ICP target bands (employees, industry, seniority, country) are
      Western-SaaS-biased defaults. Parameterise via module-level constants
      for other ICPs — do not hardcode downstream.
    - ICP_FREE_MAIL_DOMAINS lacks APAC providers (rediffmail.com,
      qq.com, 163.com, naver.com) — rebalance in a future slice if PinkMate
      moves into APAC-primary deployment.
    - Monetary bands are a step function; a smoother monotone mapping
      (log-linear in ACV) could be swapped in without breaking tests.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# RFM calibration constants (tuned against LeadForge distributions)
# ---------------------------------------------------------------------------

#: Days beyond which recency score saturates to 0 (lead is "cold")
RECENCY_DAYS_SATURATION: int = 90

#: Touch count beyond which frequency score saturates to 1 (lead is "hot")
FREQUENCY_SATURATION: int = 20

#: Monetary ACV bands (USD). Each band maps to a score in [0, 1].
#: Bands: <10k → 0.1, 10-50k → 0.3, 50-200k → 0.5, 200k-1M → 0.75, >1M → 1.0
MONETARY_ACV_BANDS: Tuple[float, ...] = (10_000.0, 50_000.0, 200_000.0, 1_000_000.0)
MONETARY_BAND_SCORES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# Clay-style ICP constants (from AazanM/clay-hubspot-enrichment reference)
# ---------------------------------------------------------------------------

#: Target employees bands (ICP sweet spot: mid-market)
ICP_TARGET_EMPLOYEES_BANDS: Tuple[str, ...] = ("50-199", "200-999")

#: Target industries (initial Western-SaaS set; parameterise per deployment)
ICP_TARGET_INDUSTRIES: Tuple[str, ...] = (
    "manufacturing",
    "logistics",
    "professional_services",
)

#: Target seniorities (director-and-above decision authority)
ICP_TARGET_SENIORITIES: Tuple[str, ...] = (
    "director",
    "vp",
    "c_suite",
)

#: Target countries (initial Western set; parameterise per deployment)
ICP_TARGET_COUNTRIES: Tuple[str, ...] = ("US", "UK")

#: Free-mail domains (addresses from these count as low-signal)
#: NOTE: Western-biased; APAC providers not included — see module docstring.
ICP_FREE_MAIL_DOMAINS: frozenset = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "live.com", "aol.com", "me.com", "mail.com",
})

#: Point weights for each Clay ICP rule (sum to 100 when all rules match)
ICP_RULE_POINTS: Dict[str, int] = {
    "employees_band_match": 30,
    "industry_match": 25,
    "seniority_match": 25,
    "country_match": 10,
    "non_free_mail": 10,
}

#: ICP tier cutoffs on the composite 0-100 score. <40 → tier D.
ICP_TIER_CUTOFFS: Dict[str, int] = {"A": 80, "B": 60, "C": 40}


# ---------------------------------------------------------------------------
# RFM scoring functions
# ---------------------------------------------------------------------------

def compute_recency_score(days_since_last_touch: Optional[int]) -> float:
    """
    Compute recency score ∈ [0, 1]. Lower days = higher score (recent lead
    is a "warmer" lead). Linear decay from 1.0 at 0 days to 0.0 at
    RECENCY_DAYS_SATURATION (90) days.

    None input → returns 0.0 (treat unknown touch as cold).
    """
    if days_since_last_touch is None:
        return 0.0
    if days_since_last_touch < 0:
        return 1.0  # Clamp negative (shouldn't happen but defensive)
    if days_since_last_touch >= RECENCY_DAYS_SATURATION:
        return 0.0
    return 1.0 - (days_since_last_touch / RECENCY_DAYS_SATURATION)


def compute_frequency_score(touch_count: Optional[int]) -> float:
    """
    Compute frequency score ∈ [0, 1]. More touches = higher score, with
    log-scaled diminishing returns past ~10 touches. Saturates at 1.0
    beyond FREQUENCY_SATURATION (20) touches.

    None input → returns 0.0 (treat unknown as zero-touches).
    """
    if touch_count is None or touch_count <= 0:
        return 0.0
    if touch_count >= FREQUENCY_SATURATION:
        return 1.0
    # log1p gives diminishing returns; normalised against saturation
    return math.log1p(touch_count) / math.log1p(FREQUENCY_SATURATION)


def compute_monetary_score(opportunity_estimated_acv: Optional[float]) -> float:
    """
    Compute monetary score ∈ [0, 1] via step-function over ACV bands.
    Returns the score of the band the ACV falls into.

    None or negative input → returns 0.0.
    """
    if opportunity_estimated_acv is None or opportunity_estimated_acv < 0:
        return 0.0
    for band_idx, upper in enumerate(MONETARY_ACV_BANDS):
        if opportunity_estimated_acv < upper:
            return MONETARY_BAND_SCORES[band_idx]
    # Beyond the highest band
    return MONETARY_BAND_SCORES[-1]


def compute_composite_rfm(
    recency: float,
    frequency: float,
    monetary: float,
    weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float:
    """
    Weighted average of the three RFM scores. Default: equal thirds.
    Weights normalised internally so they don't need to sum to 1.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    weighted_sum = weights[0] * recency + weights[1] * frequency + weights[2] * monetary
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Clay-style ICP rule functions (binary outputs)
# ---------------------------------------------------------------------------

def icp_employees_band_match(employee_band: Optional[str]) -> int:
    """1 if band is in ICP_TARGET_EMPLOYEES_BANDS, else 0."""
    if employee_band is None:
        return 0
    return 1 if employee_band in ICP_TARGET_EMPLOYEES_BANDS else 0


def icp_industry_match(industry: Optional[str]) -> int:
    """1 if industry (case-insensitive) is in ICP_TARGET_INDUSTRIES, else 0."""
    if industry is None:
        return 0
    return 1 if industry.lower() in ICP_TARGET_INDUSTRIES else 0


def icp_seniority_match(seniority: Optional[str]) -> int:
    """1 if seniority (case-insensitive) is in ICP_TARGET_SENIORITIES, else 0."""
    if seniority is None:
        return 0
    return 1 if seniority.lower() in ICP_TARGET_SENIORITIES else 0


def icp_country_match(country: Optional[str]) -> int:
    """1 if country (case-sensitive ISO-2) is in ICP_TARGET_COUNTRIES, else 0."""
    if country is None:
        return 0
    return 1 if country in ICP_TARGET_COUNTRIES else 0


def icp_non_free_mail(email: Optional[str]) -> int:
    """
    1 if email's domain is NOT in ICP_FREE_MAIL_DOMAINS (so corporate-ish),
    else 0 (free-mail provider, lower signal).

    None or malformed email → returns 0 (no signal).
    """
    if email is None or "@" not in email:
        return 0
    domain = email.rsplit("@", 1)[1].lower().strip()
    if not domain:
        return 0
    return 0 if domain in ICP_FREE_MAIL_DOMAINS else 1


# ---------------------------------------------------------------------------
# Clay-style ICP composite + tier
# ---------------------------------------------------------------------------

def compute_icp_composite(features: Dict[str, int]) -> float:
    """
    Sum the 5 Clay ICP rule outputs (each 0 or 1) weighted by their point
    allocations in ICP_RULE_POINTS. Returns a score in [0, 100].

    Expected keys in features dict:
        employees_band_match, industry_match, seniority_match,
        country_match, non_free_mail

    Missing keys treated as 0.
    """
    total = 0
    for rule_name, points in ICP_RULE_POINTS.items():
        total += int(features.get(rule_name, 0)) * points
    return float(total)


def icp_tier(composite_score: float) -> Literal["A", "B", "C", "D"]:
    """
    Assign A/B/C/D tier based on ICP_TIER_CUTOFFS.
        ≥80 → A
        ≥60 → B
        ≥40 → C
        <40 → D
    """
    if composite_score >= ICP_TIER_CUTOFFS["A"]:
        return "A"
    if composite_score >= ICP_TIER_CUTOFFS["B"]:
        return "B"
    if composite_score >= ICP_TIER_CUTOFFS["C"]:
        return "C"
    return "D"
