"""
Q-MDP formal constructs — reference implementation of the paper's §III.A
superposition-aware state modelling and governance commit semantics.

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Bucharest, Submission #4554.

Implements:
    - BeliefState: b_t(s) = P(s_t = s | o_{1:t}, a_{1:t})            (§III.A)
    - SuperposedBelief: B_t = {b_1(s_t), b_2(s_t), ..., b_n(s_t)}    (§III.A)
    - transition(): P(s_{t+1}|s_t, a_t) = Σ_i P(s_{t+1}|s_{t,i})·P(a_t|s_{t,i})
                                                                     (§III.A)
    - governance_commit(): commit(i*) ← argmax_i P(s_t|o_{1:t}, a_{1:t})
                             s.t. P(s_t|o_{1:t}, a_{1:t}) > θ_governance
                                                                     (§III.A)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Governance threshold
# ---------------------------------------------------------------------------

DEFAULT_THETA_GOVERNANCE: float = 0.75
"""
Default governance commitment threshold θ_governance.

The paper references θ_governance symbolically in §III.A ("commits
deterministically only when evidence strength exceeds governance-defined
confidence thresholds, i.e., P(s_t | o_{1:t}) > θ_governance"). The
numerical value is not fixed by the paper and is a configuration decision
for each deployment. 0.75 is the Slice 1 default — committed action
requires at least 75% posterior belief in the leading state.

For regulated deployments requiring stricter confidence, raise this value
(e.g., 0.90 for high-stakes financial decisions). Lower values widen the
set of beliefs that commit; higher values widen the set that defer.
"""


# ---------------------------------------------------------------------------
# BeliefState — single belief: b_t(s) = P(s_t = s | o_{1:t}, a_{1:t})
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeliefState:
    """
    A single belief component: the posterior probability that the latent
    enterprise state at time t equals a particular labelled state, given
    the full observation and action history up to t.

    Formal correspondence (§III.A):
        b_t(s) = P(s_t = s | o_{1:t}, a_{1:t})

    Attributes:
        state: Labelled latent state (e.g., 'QUALIFIED', 'DEFERRED').
               Interpreted by the specific deployment; the formalism is
               state-domain agnostic.
        probability: Posterior probability in [0, 1].

    Immutable (frozen) by design: belief states are observations about the
    world at a given t, not mutable buffers. State evolution produces new
    BeliefState instances via `transition()`.
    """

    state: str
    probability: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"BeliefState probability must be in [0, 1]; "
                f"got {self.probability} for state '{self.state}'"
            )
        if not self.state:
            raise ValueError("BeliefState.state must be a non-empty string")


# ---------------------------------------------------------------------------
# SuperposedBelief — set of simultaneous beliefs: B_t = {b_1, ..., b_n}
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperposedBelief:
    """
    The superposed belief set at time t — multiple latent enterprise state
    probabilities held simultaneously, prior to deterministic commitment.

    Formal correspondence (§III.A):
        B_t = {b_1(s_t), b_2(s_t), ..., b_n(s_t)}

    Interpretation (§I.C, §III.A): the system maintains several candidate
    latent states in superposition during exploration, deferring commitment
    to a single execution trajectory until a governance predicate is
    satisfied. "Superposition" here is probabilistic state enumeration, not
    quantum; implemented via a probability-normalised set of BeliefState
    components.

    Attributes:
        beliefs: List of BeliefState components, probabilities summing to 1.
        step: Discrete time index t (sequence counter, not wall clock —
              determinism requires a replayable integer).
    """

    beliefs: Tuple[BeliefState, ...]
    step: int

    _SUM_TOLERANCE: float = 1e-6

    def __post_init__(self) -> None:
        if not self.beliefs:
            raise ValueError("SuperposedBelief requires at least one BeliefState")

        total = sum(b.probability for b in self.beliefs)
        if not math.isclose(total, 1.0, abs_tol=self._SUM_TOLERANCE):
            raise ValueError(
                f"SuperposedBelief probabilities must sum to 1.0; "
                f"got {total:.6f} across {len(self.beliefs)} components"
            )

        state_labels = [b.state for b in self.beliefs]
        if len(set(state_labels)) != len(state_labels):
            raise ValueError(
                f"SuperposedBelief must not contain duplicate state labels; "
                f"got {state_labels}"
            )

        if self.step < 0:
            raise ValueError(f"SuperposedBelief.step must be ≥ 0; got {self.step}")

    def as_dict(self) -> Dict[str, float]:
        """Return beliefs as {state_label: probability} mapping."""
        return {b.state: b.probability for b in self.beliefs}

    def leading(self) -> BeliefState:
        """Return the single BeliefState with highest probability (argmax)."""
        return max(self.beliefs, key=lambda b: b.probability)


# ---------------------------------------------------------------------------
# transition() — belief dynamics per §III.A
# ---------------------------------------------------------------------------

def transition(
    belief: SuperposedBelief,
    action: str,
    transition_matrix: Mapping[Tuple[str, str], Mapping[str, float]],
    action_likelihood: Mapping[Tuple[str, str], float],
) -> SuperposedBelief:
    """
    Advance the superposed belief one step under an action, per §III.A:

        P(s_{t+1} | s_t, a_t) = Σ_i P(s_{t+1} | s_{t,i}) · P(a_t | s_{t,i})

    The next-step belief component for each candidate next-state s' is the
    sum over current latent components s_i of the state-transition
    probability P(s' | s_i) weighted by the action likelihood P(a_t | s_i).

    Args:
        belief: Current superposed belief B_t.
        action: Action label a_t taken at step t.
        transition_matrix: Mapping (current_state, next_state) ->
            {inner_state: P(next_state | current_state)}. Simplified for
            the deterministic case: {(s_i, s'): P(s' | s_i)}.
            For Slice 1 we use the simpler collapsed form
            {(s_i, s'): P(s' | s_i)} indexed as a dict of dicts.
        action_likelihood: Mapping (state, action) -> P(a | state).

    Returns:
        A new SuperposedBelief at step t+1 with renormalised probabilities.

    Raises:
        ValueError: If the resulting unnormalised belief sums to zero
            (meaning the action is incompatible with every current latent
            state); this is a governance-relevant signal, not a silent zero.
    """
    # Extract the collapsed-form transition probabilities. The signature
    # accepts the more general Σ_i form; we evaluate the inner marginal
    # here so callers can supply either shape without an API break later.
    #
    # Simplified read: transition_matrix[s_i] -> {s': P(s' | s_i)}
    collapsed: Dict[str, Dict[str, float]] = {}
    for (s_from, s_to), p in transition_matrix.items():
        if not isinstance(p, (int, float)):
            # Caller passed the nested form; flatten it.
            # Not used in Slice 1 default pipeline but kept for extensibility.
            raise TypeError(
                "transition_matrix values must be floats P(s'|s_i); "
                "nested-dict form not supported in Slice 1."
            )
        collapsed.setdefault(s_from, {})[s_to] = float(p)

    # Enumerate the next-state support: every s' that any current s_i
    # can reach with non-zero probability.
    next_states = set()
    for s_from in collapsed:
        next_states.update(collapsed[s_from].keys())

    # Σ_i P(s' | s_i) · P(a_t | s_i) · b_t(s_i)
    # Note: we multiply by b_t(s_i) too (the current belief weight),
    # which is the standard Bayesian belief-update normalisation. The
    # paper's displayed equation suppresses the belief weight for
    # notational brevity; it is standard in POMDP/belief-MDP derivation.
    unnormalised: Dict[str, float] = {s_to: 0.0 for s_to in next_states}

    for s_to in next_states:
        for b_i in belief.beliefs:
            s_from = b_i.state
            p_transition = collapsed.get(s_from, {}).get(s_to, 0.0)
            p_action = action_likelihood.get((s_from, action), 0.0)
            unnormalised[s_to] += p_transition * p_action * b_i.probability

    total = sum(unnormalised.values())
    if total <= 0.0:
        raise ValueError(
            f"Transition produced zero total belief mass under action "
            f"'{action}'; the action is incompatible with every current "
            f"latent state. Current belief: {belief.as_dict()}."
        )

    # Renormalise and drop zero-probability components to keep the set compact.
    next_components = tuple(
        BeliefState(state=s_to, probability=p / total)
        for s_to, p in unnormalised.items()
        if p > 0.0
    )

    return SuperposedBelief(beliefs=next_components, step=belief.step + 1)


# ---------------------------------------------------------------------------
# governance_commit() — commit predicate per §III.A
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GovernanceCommit:
    """
    Result of a governance commit evaluation.

    Attributes:
        committed: Whether the governance predicate P > θ was satisfied.
        state: The committed state label if committed; None otherwise.
        confidence: The posterior probability of the leading state (always
            populated so the caller can log why a commit did or didn't fire).
        theta: The threshold evaluated against (recorded for audit replay).
    """

    committed: bool
    state: Optional[str]
    confidence: float
    theta: float


def governance_commit(
    belief: SuperposedBelief,
    theta: float = DEFAULT_THETA_GOVERNANCE,
) -> GovernanceCommit:
    """
    Evaluate the governance commit predicate per §III.A:

        commit(i*) ← argmax_i P(s_t | o_{1:t}, a_{1:t})
                       s.t.   P(s_t | o_{1:t}, a_{1:t}) > θ_governance

    Returns a GovernanceCommit record indicating whether the leading
    belief component crossed the threshold. When committed=False the
    orchestrator should defer execution (per the paper's deferred-
    commitment semantics), record the uncommitted belief into the
    execution trace, and surface the belief for human / scheduled
    re-evaluation.

    Args:
        belief: Current superposed belief B_t.
        theta: Governance threshold θ_governance in (0, 1). Defaults to
               DEFAULT_THETA_GOVERNANCE (0.75) if not specified.

    Returns:
        GovernanceCommit record.

    Raises:
        ValueError: If theta is not in (0, 1).
    """
    if not (0.0 < theta < 1.0):
        raise ValueError(f"theta must be in (0, 1); got {theta}")

    leading = belief.leading()
    committed = leading.probability > theta

    return GovernanceCommit(
        committed=committed,
        state=leading.state if committed else None,
        confidence=leading.probability,
        theta=theta,
    )
