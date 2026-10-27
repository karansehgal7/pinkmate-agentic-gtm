"""
Pipeline state model — the typed, replayable container that flows through
the Q-MDP orchestration graph.

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Submission #4554.

Design intent:
    The paper formalises an execution trace (§III.A) as

        τ_exec = [(s_0, a_0, r_0, s_1), ..., (s_{t-1}, a_{t-1}, r_{t-1}, s_t)]

    and requires deterministic reproducibility (§III.B). Both demand that
    state is carried as a typed, immutable-ish object across orchestration
    nodes rather than ad-hoc dicts. Pydantic v2 BaseModel is used so every
    field is schema-validated and every state snapshot is JSON-serialisable
    for audit-trace persistence (Slice 3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict

from pinkmate.qmdp import SuperposedBelief, GovernanceCommit, DEFAULT_THETA_GOVERNANCE


# ---------------------------------------------------------------------------
# Lead-domain payloads (B2B GTM deployment of Q-MDP)
# ---------------------------------------------------------------------------
#
# The paper is state-domain agnostic. PinkMate instantiates Q-MDP for the
# B2B GTM lead-scoring domain: latent enterprise states correspond to lead
# qualification stages, and governance thresholds gate progression from
# scoring to downstream outreach actions.
# ---------------------------------------------------------------------------


class LeadInput(BaseModel):
    """Raw lead record as ingested from upstream sources (CSV, Apollo, etc.)."""

    model_config = ConfigDict(extra="allow")

    lead_id: str = Field(..., description="Unique identifier for the lead.")
    company: Optional[str] = None
    role: Optional[str] = None
    industry: Optional[str] = None
    funding_stage: Optional[str] = None
    team_size: Optional[int] = None
    source: str = Field(default="unknown", description="Ingestion source label.")


class EnrichedRecord(BaseModel):
    """Lead record after firmographic / intent / account-intelligence enrichment."""

    model_config = ConfigDict(extra="allow")

    lead_id: str
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engineered feature dictionary passed to the ScoringAgent.",
    )
    enrichment_sources: List[str] = Field(
        default_factory=list,
        description="Audit list of enrichment sources applied (for provenance).",
    )


class ScoreOutput(BaseModel):
    """Output of the ScoringAgent: numeric score plus per-model breakdown."""

    final_score: float = Field(..., ge=0.0, le=100.0)
    model_scores: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Execution trace entry — per paper §III.A
# ---------------------------------------------------------------------------


class TraceEntry(BaseModel):
    """
    One element of the execution trace τ_exec = [(s_t, a_t, r_t, s_{t+1}), ...].

    Reward r_t is a float placeholder in Slice 1; subsequent slices will
    populate it with the risk-sensitive loss L_risk = αFN + βFP (§IV.B).
    """

    model_config = ConfigDict(frozen=True)

    step: int
    node: str = Field(..., description="Orchestration node that produced this entry.")
    state_before: Dict[str, float] = Field(
        default_factory=dict,
        description="SuperposedBelief at entry (as {state_label: probability}).",
    )
    action: str = Field(..., description="Action label a_t taken by this node.")
    reward: float = Field(default=0.0, description="Reward r_t (placeholder in Slice 1).")
    state_after: Dict[str, float] = Field(
        default_factory=dict,
        description="SuperposedBelief after transition.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# PipelineState — the LangGraph graph state
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """
    The typed state object flowing through the LangGraph StateGraph.

    Carries lead-domain payloads, the current superposed belief, the
    execution trace, and the governance commit outcome. Serialisable
    end-to-end so each state snapshot can be persisted for audit replay.

    Determinism contract (§III.B):
        Given identical input, identical config (theta_governance, seed,
        transition matrices), the fully-run PipelineState must be identical
        across runs. Tested in tests/test_pipeline.py.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Inputs
    lead_input: LeadInput

    # Intermediate outputs
    enriched: Optional[EnrichedRecord] = None
    score: Optional[ScoreOutput] = None

    # Q-MDP belief tracking
    belief: Optional[SuperposedBelief] = Field(
        default=None,
        description="Current superposed belief B_t after the most recent step.",
    )
    governance_decision: Optional[GovernanceCommit] = Field(
        default=None,
        description="Outcome of the most recent governance_commit() evaluation.",
    )
    committed_action: Optional[str] = Field(
        default=None,
        description=(
            "The committed next action label if the governance predicate "
            "fired; None if execution was deferred."
        ),
    )

    # Config
    theta_governance: float = Field(
        default=DEFAULT_THETA_GOVERNANCE,
        gt=0.0,
        lt=1.0,
        description="Governance commit threshold θ_governance.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for any stochastic operation (replay anchor).",
    )

    # Execution trace τ_exec
    trace: List[TraceEntry] = Field(default_factory=list)

    # Downstream outputs
    segment: Optional[str] = Field(
        default=None,
        description="Segmentation tier label (e.g., 'Tier 1', 'Nurture').",
    )
    next_best_action: Optional[str] = Field(
        default=None,
        description="OutreachAgent decision label.",
    )
    deferred: bool = Field(
        default=False,
        description=(
            "True if the governance predicate did not fire and execution "
            "was deferred without committing a downstream action."
        ),
    )
