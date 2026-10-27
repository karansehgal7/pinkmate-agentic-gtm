"""
pinkmate — Reference deployment of the Q-MDP orchestration layer.

Companion repository to:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." Accepted at IEEE Cyber-AI
    2026, Bucharest, Romania (Submission #4554).

The package implements the formal constructs of the paper:
    - Belief-state dynamics (§III.A)
    - Superposition-aware state modelling (§III.A)
    - Governance commitment predicate (§III.A)
    - Deterministic event-driven orchestration graph (§III.B)
    - Imbalance-aware ML layer (§III.C–D, released in subsequent slice)
    - TreeSHAP attribution-stability measurement (§V, released in subsequent
      slice)
"""

__version__ = "0.1.0"

from pinkmate.qmdp import (
    BeliefState,
    SuperposedBelief,
    GovernanceCommit,
    transition,
    governance_commit,
    DEFAULT_THETA_GOVERNANCE,
)
from pinkmate.state import (
    PipelineState,
    LeadInput,
    EnrichedRecord,
    ScoreOutput,
    TraceEntry,
)

__all__ = [
    "BeliefState",
    "SuperposedBelief",
    "GovernanceCommit",
    "transition",
    "governance_commit",
    "DEFAULT_THETA_GOVERNANCE",
    "PipelineState",
    "LeadInput",
    "EnrichedRecord",
    "ScoreOutput",
    "TraceEntry",
]
