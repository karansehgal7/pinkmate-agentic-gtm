"""
IngestionAgent — orchestration node V_ingestion in the Q-MDP graph (§III.B).

Responsible for normalising raw lead input (CSV, Apollo, LinkedIn export)
into a unified LeadInput record. In Slice 1 the ingestion is a pass-through
with light normalisation; downstream slices will add source-specific parsers.
"""

from __future__ import annotations

from pinkmate.state import LeadInput, PipelineState, TraceEntry


class IngestionAgent:
    """Normalise raw lead input into the canonical LeadInput schema."""

    NODE_NAME: str = "ingestion"
    ACTION: str = "ingest"

    def __call__(self, state: PipelineState) -> PipelineState:
        """
        Pydantic validates the LeadInput at construction, so Slice 1 ingestion
        is a trace append + state return. Source-specific normalisation
        (CSV/Apollo/LinkedIn parsers) is scoped for Slice 2.
        """
        current_belief = state.belief.as_dict() if state.belief else {}

        state.trace.append(
            TraceEntry(
                step=len(state.trace),
                node=self.NODE_NAME,
                state_before=current_belief,
                action=self.ACTION,
                state_after=current_belief,
                metadata={"lead_id": state.lead_input.lead_id},
            )
        )
        return state
