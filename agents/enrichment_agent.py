"""
EnrichmentAgent — orchestration node V_enrichment in the Q-MDP graph (§III.B).

Attaches firmographic / intent / account-intelligence features to the lead
record so the ScoringAgent has engineered features to score against. In
Slice 1 this is a deterministic feature-synthesis function over the fields
available on LeadInput; subsequent slices will plug in real enrichment
sources (Clearbit, Apollo, intent providers) behind the same interface.
"""

from __future__ import annotations

from typing import Any, Dict

from pinkmate.state import EnrichedRecord, PipelineState, TraceEntry


class EnrichmentAgent:
    """Produce an EnrichedRecord with an engineered feature dictionary."""

    NODE_NAME: str = "enrichment"
    ACTION: str = "enrich"

    def __call__(self, state: PipelineState) -> PipelineState:
        lead = state.lead_input

        features: Dict[str, Any] = {
            "industry_saas": 1 if (lead.industry or "").lower() == "saas" else 0,
            "funding_stage_series_a": 1 if (lead.funding_stage or "").lower() == "series_a" else 0,
            "team_size_50_200": 1 if lead.team_size and 50 <= lead.team_size <= 200 else 0,
            "role_seniority_score": self._role_seniority(lead.role),
            "hiring_for_gtm": 0,  # placeholder; populated by real signal source in Slice 2
        }

        state.enriched = EnrichedRecord(
            lead_id=lead.lead_id,
            features=features,
            enrichment_sources=["slice1_deterministic_synthesis"],
        )

        current_belief = state.belief.as_dict() if state.belief else {}
        state.trace.append(
            TraceEntry(
                step=len(state.trace),
                node=self.NODE_NAME,
                state_before=current_belief,
                action=self.ACTION,
                state_after=current_belief,
                metadata={"feature_count": len(features)},
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
