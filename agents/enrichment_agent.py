"""
EnrichmentAgent — orchestration node V_enrichment in the Q-MDP graph (§III.B).

Attaches firmographic / intent / account-intelligence features to the lead
record so the ScoringAgent has engineered features to score against. In
Slice 1 this is a deterministic feature-synthesis function over the fields
available on LeadInput; subsequent slices will plug in real enrichment
sources (Clearbit, Apollo, intent providers) behind the same interface.

Slice 5.2 scope (shipped):
    - RFM feature engineering: recency / frequency / monetary / composite
      computed from LeadInput primitives (days_since_last_touch,
      touch_count, opportunity_estimated_acv) via pure functions in
      pinkmate.ml.feature_engineering.
    - Clay-style ICP scoring layer: 5 rule-based binary features
      (employees_band_match, industry_match, seniority_match,
      country_match, non_free_mail) + composite 0-100 score + A/B/C/D tier.
    - Decision C (16 Nov 2026) locked BOTH rules + composite exposed to
      the downstream model — interpretability wins; redundancy is cheap.
"""

from __future__ import annotations

from typing import Any, Dict

from pinkmate.state import EnrichedRecord, PipelineState, TraceEntry
from pinkmate.ml.feature_engineering import (
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


class EnrichmentAgent:
    """Produce an EnrichedRecord with an engineered feature dictionary."""

    NODE_NAME: str = "enrichment"
    ACTION: str = "enrich"

    def __call__(self, state: PipelineState) -> PipelineState:
        lead = state.lead_input

        # Slice 1 — core deterministic features (preserved for backward-compat)
        features: Dict[str, Any] = {
            "industry_saas": 1 if (lead.industry or "").lower() == "saas" else 0,
            "funding_stage_series_a": 1 if (lead.funding_stage or "").lower() == "series_a" else 0,
            "team_size_50_200": 1 if lead.team_size and 50 <= lead.team_size <= 200 else 0,
            "role_seniority_score": self._role_seniority(lead.role),
            "hiring_for_gtm": 0,  # placeholder; populated by real signal source in Slice 2
        }

        # ───────────────────────────────────────────────────────────────
        # Slice 5.2 — RFM + Clay ICP feature engineering
        # ───────────────────────────────────────────────────────────────

        # RFM scores (from LeadForge-style touch data primitives)
        r_score = compute_recency_score(getattr(lead, "days_since_last_touch", None))
        f_score = compute_frequency_score(getattr(lead, "touch_count", None))
        m_score = compute_monetary_score(getattr(lead, "opportunity_estimated_acv", None))
        features["recency_score"] = r_score
        features["frequency_score"] = f_score
        features["monetary_score"] = m_score
        features["composite_rfm_score"] = compute_composite_rfm(r_score, f_score, m_score)

        # Clay-style ICP rule-based binary features
        features["icp_employees_band_match"] = icp_employees_band_match(
            getattr(lead, "employee_band", None)
        )
        features["icp_industry_match"] = icp_industry_match(
            getattr(lead, "industry", None)
        )
        features["icp_seniority_match"] = icp_seniority_match(
            getattr(lead, "seniority", None)
        )
        features["icp_country_match"] = icp_country_match(
            getattr(lead, "country", None)
        )
        features["icp_non_free_mail"] = icp_non_free_mail(
            getattr(lead, "email", None)
        )

        # Clay composite ICP score + tier (Decision C: BOTH exposed to model)
        icp_rule_outputs = {
            "employees_band_match": features["icp_employees_band_match"],
            "industry_match":       features["icp_industry_match"],
            "seniority_match":      features["icp_seniority_match"],
            "country_match":        features["icp_country_match"],
            "non_free_mail":        features["icp_non_free_mail"],
        }
        icp_score = compute_icp_composite(icp_rule_outputs)
        features["icp_composite_score"] = icp_score
        features["icp_tier_label"] = icp_tier(icp_score)  # "A"/"B"/"C"/"D" string

        # ───────────────────────────────────────────────────────────────

        state.enriched = EnrichedRecord(
            lead_id=lead.lead_id,
            features=features,
            enrichment_sources=[
                "slice1_deterministic_synthesis",
                "slice52_rfm_icp",
            ],
        )

        current_belief = state.belief.as_dict() if state.belief else {}
        state.trace.append(
            TraceEntry(
                step=len(state.trace),
                node=self.NODE_NAME,
                state_before=current_belief,
                action=self.ACTION,
                state_after=current_belief,
                metadata={
                    "feature_count": len(features),
                    "rfm_composite": features["composite_rfm_score"],
                    "icp_composite": features["icp_composite_score"],
                    "icp_tier": features["icp_tier_label"],
                },
            )
        )
        return state

    @staticmethod
    def _role_seniority(role: str | None) -> float:
        """
        Deterministic seniority score in [0, 1]. Replaced by a trained
        classifier in Slice 2; kept rule-based here so Slice 1 is fully
        deterministic without a model dependency.
        """
        if not role:
            return 0.0
        role_lower = role.lower()
        if any(k in role_lower for k in ("ceo", "cto", "founder", "chief")):
            return 1.0
        if any(k in role_lower for k in ("vp", "head of", "director")):
            return 0.9
        if any(k in role_lower for k in ("senior", "principal", "lead")):
            return 0.7
        if any(k in role_lower for k in ("manager",)):
            return 0.5
        return 0.3
