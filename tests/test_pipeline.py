"""
Integration smoke tests on the LangGraph Q-MDP pipeline (workflows/pipeline.py).

Verifies:
    1. The graph builds and runs end-to-end on a mock lead.
    2. The committed path populates every expected state field.
    3. The deferred path triggers correctly when the belief does not clear θ.
    4. Two independent runs on identical inputs produce identical final
       state — the deterministic replay claim of paper §III.B.
"""

from __future__ import annotations

import pytest

from workflows.pipeline import build_pipeline
from pinkmate.state import LeadInput, PipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(initial: PipelineState, thread_id: str) -> PipelineState:
    """Run the compiled graph and normalise the output to a PipelineState."""
    app = build_pipeline()
    config = {"configurable": {"thread_id": thread_id}}
    final = app.invoke(initial, config=config)
    if isinstance(final, dict):
        final = PipelineState.model_validate(final)
    return final


def _high_quality_lead() -> PipelineState:
    """A lead whose feature profile should score high enough to commit at θ=0.75."""
    return PipelineState(
        lead_input=LeadInput(
            lead_id="HIGH-001",
            company="High Fit Ltd",
            role="CTO",
            industry="SaaS",
            funding_stage="Series_A",
            team_size=120,
            source="test",
        )
    )


def _ambiguous_lead() -> PipelineState:
    """
    A lead whose feature profile should resolve to a belief where neither
    QUALIFIED nor UNQUALIFIED clears θ=0.75 — the true 'uncommitted' defer
    path from paper §III.A.
    """
    return PipelineState(
        lead_input=LeadInput(
            lead_id="AMB-001",
            company="Ambiguous Co",
            role="Senior Analyst",  # 0.7 seniority * 40 = 28
            industry="SaaS",         # 15
            funding_stage="Seed",   # 0 (only Series_A counts in Slice 1)
            team_size=30,            # 0 (outside 50-200)
            source="test",
        )
    )


def _confident_reject_lead() -> PipelineState:
    """
    A lead whose feature profile resolves to confident UNQUALIFIED (>0.75).
    The generic Q-MDP predicate fires on UNQUALIFIED, but the domain rule
    in ScoringAgent only progresses on committed QUALIFIED — so this still
    defers (terminal-reject semantic).
    """
    return PipelineState(
        lead_input=LeadInput(
            lead_id="REJ-001",
            company="Confident Reject Ltd",
            role="Intern",
            industry="Retail",
            funding_stage="Bootstrapped",
            team_size=5,
            source="test",
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCommittedPath:
    def test_runs_end_to_end(self):
        final = _run(_high_quality_lead(), thread_id="committed-smoke")
        assert final.enriched is not None, "enrichment did not populate"
        assert final.score is not None, "scoring did not populate"
        assert final.belief is not None, "belief was not emitted by scoring"
        assert final.governance_decision is not None
        assert final.committed_action is not None
        assert final.segment is not None, "segmentation did not run on committed path"
        assert final.next_best_action is not None, "outreach did not run on committed path"
        assert final.deferred is False

    def test_commits_at_qualified_state(self):
        final = _run(_high_quality_lead(), thread_id="committed-state")
        assert final.governance_decision.committed is True
        assert final.committed_action == "QUALIFIED"

    def test_trace_contains_each_node_in_order(self):
        final = _run(_high_quality_lead(), thread_id="committed-trace")
        node_sequence = [entry.node for entry in final.trace]
        assert node_sequence == ["ingestion", "enrichment", "scoring", "segmentation", "outreach"], (
            f"Unexpected trace sequence: {node_sequence}"
        )


class TestDeferredPathUncommitted:
    """
    Belief does not clear θ for any component — the paper's §III.A
    'cannot commit' case.
    """

    def test_defers_when_belief_below_theta(self):
        final = _run(_ambiguous_lead(), thread_id="deferred-smoke")
        assert final.belief is not None
        assert final.governance_decision is not None
        assert final.governance_decision.committed is False
        assert final.committed_action is None
        assert final.deferred is True
        assert final.segment is None
        assert final.next_best_action is None

    def test_defer_trace_entry_is_recorded(self):
        final = _run(_ambiguous_lead(), thread_id="deferred-trace")
        last = final.trace[-1]
        assert last.node == "defer"
        assert last.action == "defer_uncommitted"
        assert "reason" in last.metadata


class TestDeferredPathConfidentReject:
    """
    Belief resolves confidently to UNQUALIFIED (>θ). The generic Q-MDP
    commit predicate fires, but the lead-domain rule only progresses on
    confident QUALIFIED. Verifies the domain-semantic defer path.
    """

    def test_confident_unqualified_still_defers(self):
        final = _run(_confident_reject_lead(), thread_id="deferred-reject-smoke")
        assert final.belief is not None
        assert final.governance_decision is not None
        # Governance predicate itself fires — leading belief exceeded θ.
        assert final.governance_decision.committed is True
        assert final.governance_decision.state == "UNQUALIFIED"
        # But the domain rule prevents routing to outreach.
        assert final.committed_action is None
        assert final.deferred is True
        assert final.segment is None
        assert final.next_best_action is None


class TestDeterministicReplay:
    """
    Paper §III.B: deterministic execution requires that identical input +
    identical configuration produce identical final state.
    """

    def test_two_runs_produce_identical_committed_state(self):
        a = _run(_high_quality_lead(), thread_id="replay-A")
        b = _run(_high_quality_lead(), thread_id="replay-B")

        assert a.score.model_dump() == b.score.model_dump()
        assert a.belief.as_dict() == b.belief.as_dict()
        assert a.governance_decision.committed == b.governance_decision.committed
        assert a.governance_decision.state == b.governance_decision.state
        assert a.governance_decision.confidence == pytest.approx(b.governance_decision.confidence)
        assert a.segment == b.segment
        assert a.next_best_action == b.next_best_action
        assert a.committed_action == b.committed_action
        assert a.deferred == b.deferred

        # Trace equality on structural fields (step/node/action) — the full
        # trace equality asserts the orchestration path is reproducible.
        a_sig = [(e.step, e.node, e.action) for e in a.trace]
        b_sig = [(e.step, e.node, e.action) for e in b.trace]
        assert a_sig == b_sig

    def test_two_runs_produce_identical_deferred_state(self):
        a = _run(_ambiguous_lead(), thread_id="replay-defer-A")
        b = _run(_ambiguous_lead(), thread_id="replay-defer-B")
        assert a.deferred is True and b.deferred is True
        assert a.belief.as_dict() == b.belief.as_dict()
        assert a.governance_decision.confidence == pytest.approx(b.governance_decision.confidence)
