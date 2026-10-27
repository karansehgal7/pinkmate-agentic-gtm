"""
OutreachAgent — orchestration node V_outreach in the Q-MDP graph (§III.B).

Terminal node for the committed path: given a segmentation tier, decide
the next-best outreach action. Writes next_best_action into pipeline state
and appends the final trace entry.
"""

from __future__ import annotations

from pinkmate.state import PipelineState, TraceEntry


class OutreachAgent:
    """Decide the next-best outreach action for a segmented, committed lead."""

    NODE_NAME: str = "outreach"
    ACTION: str = "decide_next_best_action"

    _ACTION_BY_SEGMENT = {
        "Tier 1": "founder_direct_email_sequence",
        "Tier 2": "ae_warm_sequence",
        "Tier 3": "sdr_cold_sequence",
        "Nurture": "nurture_newsletter_enrol",
    }

    def __call__(self, state: PipelineState) -> PipelineState:
        if state.segment is None:
            raise RuntimeError("OutreachAgent reached without a segment in state.")

        action_label = self._ACTION_BY_SEGMENT.get(state.segment, "nurture_newsletter_enrol")
        state.next_best_action = action_label

        current_belief = state.belief.as_dict() if state.belief else {}
        state.trace.append(
            TraceEntry(
                step=len(state.trace),
                node=self.NODE_NAME,
                state_before=current_belief,
                action=self.ACTION,
                state_after=current_belief,
                metadata={"next_best_action": action_label, "segment": state.segment},
            )
        )
        return state
