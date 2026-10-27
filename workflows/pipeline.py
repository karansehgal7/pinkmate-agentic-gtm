"""
PinkMate orchestration graph — reference implementation of the Q-MDP
deterministic event-driven orchestration layer (paper §III.B).

Formal correspondence:
    G = (V, E)   where V are orchestration micro-task vertices (agents)
                 and E are event-trigger dependencies (edges).

    The edge from the scoring node to the segmentation node is CONDITIONAL
    on the governance commit predicate (§III.A):

        commit(i*) ← argmax_i P(s_t | o_{1:t}, a_{1:t})
                       s.t.   P(s_t | o_{1:t}, a_{1:t}) > θ_governance

    When the predicate fires, the orchestrator routes to segmentation →
    outreach. When it does not fire, the orchestrator routes to the
    `defer` node, records the uncommitted belief into the execution trace,
    and halts without committing a downstream action — the paper's
    deferred-commitment semantics.

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Submission #4554. See repo README for citation.

Usage:
    python -m workflows.pipeline
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.ingestion_agent import IngestionAgent
from agents.enrichment_agent import EnrichmentAgent
from agents.scoring_agent import ScoringAgent
from agents.segmentation_agent import SegmentationAgent
from agents.outreach_agent import OutreachAgent

from pinkmate.state import PipelineState, TraceEntry


# ---------------------------------------------------------------------------
# Governance edge — pure router reading the governance decision written by
# the ScoringAgent node (which evaluates governance_commit() per §III.A).
# ---------------------------------------------------------------------------

def governance_router(state: PipelineState) -> Literal["commit", "defer"]:
    """
    Route conditionally on the governance commit outcome recorded by the
    ScoringAgent.

    Pure routing function: no state mutation here (LangGraph routers are
    pure dispatchers; durable state writes belong in node bodies, which is
    where ScoringAgent evaluates and records governance_commit()).

    Returns:
        "commit" → forward to segmentation → outreach.
        "defer"  → forward to the defer node, which records the
                   uncommitted or terminal-reject belief and halts.
    """
    if state.committed_action is not None:
        return "commit"
    return "defer"


def defer_node(state: PipelineState) -> PipelineState:
    """
    Terminal node for the deferred path. Records the uncommitted belief
    into the execution trace with the governance decision metadata so the
    audit replay can reconstruct why execution did not proceed.
    """
    state.deferred = True

    belief_snapshot = state.belief.as_dict() if state.belief else {}
    gov = state.governance_decision
    state.trace.append(
        TraceEntry(
            step=len(state.trace),
            node="defer",
            state_before=belief_snapshot,
            action="defer_uncommitted",
            state_after=belief_snapshot,
            metadata={
                "theta_governance": state.theta_governance,
                "leading_confidence": gov.confidence if gov else None,
                "reason": (
                    f"leading belief {gov.confidence:.4f} ≤ θ_governance "
                    f"{gov.theta:.2f}"
                    if gov
                    else "no belief emitted by scoring step"
                ),
            },
        )
    )
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_pipeline(checkpointer: object | None = None) -> object:
    """
    Build the compiled LangGraph StateGraph for the PinkMate Q-MDP pipeline.

    Args:
        checkpointer: Optional LangGraph checkpointer. Defaults to in-process
            MemorySaver for Slice 1. Slice 3 will swap in SqliteSaver and
            later a PostgreSQL checkpointer for durable audit-trail
            persistence.

    Returns:
        Compiled LangGraph app with .invoke() / .stream() methods.
    """
    ingestion = IngestionAgent()
    enrichment = EnrichmentAgent()
    scoring = ScoringAgent()
    segmentation = SegmentationAgent()
    outreach = OutreachAgent()

    graph = StateGraph(PipelineState)

    # Vertices V
    graph.add_node(ingestion.NODE_NAME, ingestion)
    graph.add_node(enrichment.NODE_NAME, enrichment)
    graph.add_node(scoring.NODE_NAME, scoring)
    graph.add_node(segmentation.NODE_NAME, segmentation)
    graph.add_node(outreach.NODE_NAME, outreach)
    graph.add_node("defer", defer_node)

    # Edges E
    graph.add_edge(START, ingestion.NODE_NAME)
    graph.add_edge(ingestion.NODE_NAME, enrichment.NODE_NAME)
    graph.add_edge(enrichment.NODE_NAME, scoring.NODE_NAME)

    # Governance-gated conditional edge — the paper's commit predicate.
    graph.add_conditional_edges(
        scoring.NODE_NAME,
        governance_router,
        {
            "commit": segmentation.NODE_NAME,
            "defer": "defer",
        },
    )

    graph.add_edge(segmentation.NODE_NAME, outreach.NODE_NAME)
    graph.add_edge(outreach.NODE_NAME, END)
    graph.add_edge("defer", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI entrypoint — runs an example lead through the full graph
# ---------------------------------------------------------------------------

def _demo() -> None:
    import json
    from pinkmate.state import LeadInput

    app = build_pipeline()
    initial = PipelineState(
        lead_input=LeadInput(
            lead_id="DEMO-001",
            company="Acme SaaS",
            role="VP of Engineering",
            industry="SaaS",
            funding_stage="Series_A",
            team_size=120,
            source="demo",
        )
    )

    config = {"configurable": {"thread_id": "demo-thread"}}
    final = app.invoke(initial, config=config)

    # LangGraph may return a dict or a PipelineState depending on version;
    # normalise to the Pydantic model for the demo output.
    if isinstance(final, dict):
        final = PipelineState.model_validate(final)

    print("=" * 72)
    print("PinkMate Q-MDP demo run")
    print("=" * 72)
    print(f"Lead:              {final.lead_input.lead_id} ({final.lead_input.company})")
    print(f"Final score:       {final.score.final_score if final.score else 'n/a'}")
    print(f"Belief:            {final.belief.as_dict() if final.belief else 'n/a'}")
    if final.governance_decision:
        gov = final.governance_decision
        print(
            f"Governance:        committed={gov.committed}  "
            f"state={gov.state}  confidence={gov.confidence:.4f}  "
            f"θ={gov.theta:.2f}"
        )
    print(f"Segment:           {final.segment}")
    print(f"Next-best action:  {final.next_best_action}")
    print(f"Deferred:          {final.deferred}")
    print()
    print(f"Trace ({len(final.trace)} entries):")
    for entry in final.trace:
        print(f"  [{entry.step}] {entry.node:>13s} :: {entry.action}")


if __name__ == "__main__":
    _demo()
