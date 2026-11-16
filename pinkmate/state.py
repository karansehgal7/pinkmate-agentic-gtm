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

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict

from pinkmate.qmdp import SuperposedBelief, GovernanceCommit, DEFAULT_THETA_GOVERNANCE


# ---------------------------------------------------------------------------
# Slice 5.2 — LeadInput schema versioning for ScoringAgent drift detection
# ---------------------------------------------------------------------------
#
# When a model is trained, the current LEADINPUT_SCHEMA_VERSION is persisted
# alongside the model artefacts (models/gtm/attribution.json). At inference
# time, ScoringAgent compares the persisted version against the live
# constant here. A mismatch triggers a WARNING + trace log, not a hard
# failure — prediction proceeds on the intersection of features (hybrid
# fail-fast contract per Slice 5.2 Decision A).
# ---------------------------------------------------------------------------

#: Slice 5.2 — LeadInput schema version for ScoringAgent drift detection.
#: Bump this whenever the field set on LeadInput changes in a way that
#: affects model training — e.g., adding RFM features (slice52_rfm_icp_v1),
#: adding survival-time fields (future slice6_survival_v1), etc.
LEADINPUT_SCHEMA_VERSION: str = "slice52_rfm_icp_v1"


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
    """
    Raw lead record as ingested from upstream sources (CSV, Apollo, etc.).

    Schema evolution:
        Slice 1-4: core fields (lead_id, company, role, industry,
            funding_stage, team_size, source).
        Slice 5 (GTM domain extension): adds optional band fields that
            map to the LeadForge Lead Scoring v1 training schema —
            employee_band, estimated_revenue_band, process_maturity_band.
            `funding_stage` is retained in the API but is NOT in the
            Slice 5 training schema (not present in LeadForge data); it
            re-enters the training schema in Slice 5.1+ when real B2B
            data with funding-stage labels arrives (see Slice 5 scope
            notes in docs/architecture.md §5d).
        Slice 5.2 (deployment-schema alignment): adds optional email +
            RFM primitives (recency/frequency/monetary/composite) +
            Clay-style ICP features (5 rule-based binary features + a
            composite score + A/B/C/D tier). All computed by the
            EnrichmentAgent from source fields. See docs/architecture.md
            §5e for the full specification.

    All Slice 5 and Slice 5.2 fields are optional with default None,
    preserving backward compatibility with all Slice 1-4 callers.
    """

    model_config = ConfigDict(extra="allow")

    lead_id: str = Field(..., description="Unique identifier for the lead.")
    company: Optional[str] = None
    role: Optional[str] = None
    industry: Optional[str] = None
    funding_stage: Optional[str] = None
    team_size: Optional[int] = None
    source: str = Field(default="unknown", description="Ingestion source label.")

    # Slice 5 — LeadForge GTM-schema band fields (optional, backward-compat)
    employee_band: Optional[str] = Field(
        default=None,
        description="Employee count band (e.g. '200-999'). Maps to LeadForge "
                    "employee_band categorical. Preferred over raw team_size "
                    "when the lead's exact headcount is noisy or unknown.",
    )
    estimated_revenue_band: Optional[str] = Field(
        default=None,
        description="Annual revenue band (e.g. '$10M-$50M'). Maps to LeadForge "
                    "estimated_revenue_band categorical.",
    )
    process_maturity_band: Optional[str] = Field(
        default=None,
        description="Buying-process maturity band ('low'/'medium'/'high'). "
                    "Maps to LeadForge process_maturity_band categorical.",
    )

    # Slice 5.2 — Email (needed for Clay ICP non-free-mail check)
    email: Optional[str] = Field(
        default=None,
        description="Lead's email address. Used by Clay-style ICP scoring "
                    "to detect free-mail providers (gmail.com, yahoo.com, "
                    "etc.) as lower-signal vs corporate domains.",
    )

    # Slice 5.2 — Source primitives for RFM computation
    days_since_last_touch: Optional[int] = Field(
        default=None,
        description="Days elapsed since the most recent touch on this lead. "
                    "Source primitive for compute_recency_score().",
    )
    touch_count: Optional[int] = Field(
        default=None,
        description="Total number of touches (emails opened, pages viewed, "
                    "events attended) accumulated for this lead. Source "
                    "primitive for compute_frequency_score().",
    )
    opportunity_estimated_acv: Optional[float] = Field(
        default=None,
        description="Estimated annual contract value in USD for the deal if it "
                    "converts. Source primitive for compute_monetary_score().",
    )
    seniority: Optional[str] = Field(
        default=None,
        description="Seniority label derived from role ('director', 'vp', "
                    "'c_suite', 'manager', 'individual_contributor'). Used "
                    "by Clay-style ICP scoring.",
    )
    country: Optional[str] = Field(
        default=None,
        description="ISO-2 country code (e.g., 'US', 'UK', 'DE'). Used by "
                    "Clay-style ICP scoring to check target-country match.",
    )

    # Slice 5.2 — RFM scores (computed by EnrichmentAgent)
    recency_score: Optional[float] = Field(
        default=None,
        description="Recency score ∈ [0, 1]. 1.0 = most recent touch, 0.0 = "
                    "no touch within 90 days. Computed from days_since_last_touch.",
    )
    frequency_score: Optional[float] = Field(
        default=None,
        description="Frequency score ∈ [0, 1], log-scaled, saturates at 20 "
                    "touches. Computed from touch_count.",
    )
    monetary_score: Optional[float] = Field(
        default=None,
        description="Monetary score ∈ [0, 1], step-function over ACV bands. "
                    "Computed from opportunity_estimated_acv.",
    )
    composite_rfm_score: Optional[float] = Field(
        default=None,
        description="Weighted composite of recency + frequency + monetary "
                    "(default equal thirds). Computed by EnrichmentAgent.",
    )

    # Slice 5.2 — Clay-style ICP features (deterministic rule outputs)
    icp_employees_band_match: Optional[int] = Field(
        default=None,
        description="1 if employee_band ∈ ICP target bands, else 0. Clay ICP rule.",
    )
    icp_industry_match: Optional[int] = Field(
        default=None,
        description="1 if industry (case-insensitive) ∈ ICP target industries, else 0.",
    )
    icp_seniority_match: Optional[int] = Field(
        default=None,
        description="1 if seniority (case-insensitive) ∈ ICP target seniorities, else 0.",
    )
    icp_country_match: Optional[int] = Field(
        default=None,
        description="1 if country ∈ ICP target countries, else 0.",
    )
    icp_non_free_mail: Optional[int] = Field(
        default=None,
        description="1 if email domain is NOT in free-mail providers list "
                    "(gmail.com, yahoo.com, etc.), else 0.",
    )
    icp_composite_score: Optional[float] = Field(
        default=None,
        description="Composite ICP score ∈ [0, 100], sum of matched rule-points.",
    )
    icp_tier: Optional[Literal["A", "B", "C", "D"]] = Field(
        default=None,
        description="A/B/C/D tier assigned from composite_icp_score. "
                    "≥80 → A, ≥60 → B, ≥40 → C, <40 → D.",
    )


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
