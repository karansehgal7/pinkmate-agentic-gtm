"""
SegmentationAgent — orchestration node V_segmentation in the Q-MDP graph (§III.B).

Only reached when the governance commit predicate (§III.A) has fired for
the leading belief. Maps the committed belief state + score into a
segmentation tier label that downstream OutreachAgent consumes.
"""

from __future__ import annotations

from pinkmate.state import PipelineState, TraceEntry


class SegmentationAgent:
    """Assign a segmentation tier to a committed lead."""

    NODE_NAME: str = "segmentation"
    ACTION: str = "segment"

    def __call__(self, state: PipelineState) -> PipelineState:
        if state.score is None:
            raise RuntimeError("SegmentationAgent reached without a ScoreOutput in state.")

        s = state.score.final_score
        if s >= 85.0:
            segment = "Tier 1"
        elif s >= 65.0:
            segment = "Tier 2"
        elif s >= 45.0:
            segment = "Tier 3"
        else:
            segment = "Nurture"

        state.segment = segment

        current_belief = state.belief.as_dict() if state.belief else {}
        state.trace.append(
            TraceEntry(
                step=len(state.trace),
                node=self.NODE_NAME,
                state_before=current_belief,
                action=self.ACTION,
                state_after=current_belief,
                metadata={"segment": segment, "final_score": s},
            )
        )
        return state
