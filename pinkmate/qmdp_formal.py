"""
POMDP formalism + Q-MDP commit rule (Slice 7).

Formal theoretical grounding of PinkMate's SuperposedBelief + governance_commit
primitives in the canonical POMDP + Q-MDP literature. Citations:

    Littman, M. L., Cassandra, A. R., & Kaelbling, L. P. (1995).
    "Learning policies for partially observable environments: scaling up."
    In Machine Learning Proceedings 1995 (pp. 362-370). Morgan Kaufmann.

This module provides the clean mathematical definitions that `pinkmate/qmdp.py`
operationalises. It is theoretical scaffolding — the production path still
goes through `pinkmate/qmdp.py::SuperposedBelief + governance_commit`. The
isomorphism between the two is proven in `tests/test_qmdp_formal.py` via
known-answer belief updates.

Four formal objects (POMDP tuple):

    S  -- discrete latent state space       e.g. {MQL, SQL, ClosedWon}
    A  -- discrete action space              e.g. {score, govern, outreach}
    T(s, a, s') -- state-transition kernel   P(s' | s, a)
    O(s', a, o) -- observation kernel        P(o | s', a)
    R(s, a)     -- reward (optional for Q-MDP commit)
    gamma       -- discount factor in [0, 1)

Three fundamental operations:

    1. Bayesian belief filter (observation update):
           b'(s') = eta * O(s', a, o) * sum_s T(s, a, s') * b(s)
       where eta is the normaliser ensuring sum_s' b'(s') = 1.

    2. Q-MDP commit (Littman et al. 1995 equation 4):
           pi_QMDP(b) = argmax_a sum_s b(s) * Q_MDP(s, a)
       where Q_MDP(s, a) is the underlying fully-observable MDP's optimal
       Q-function — Q-MDP APPROXIMATES the optimal POMDP policy by weighting
       MDP Q-values with the current belief.

    3. Governance commit predicate (PinkMate-specific, Slice 1):
           commit(b, theta) := max_s b(s) >= theta
       Isomorphic to a degenerate Q-MDP commit where A = {commit, defer},
       Q_MDP(s, commit) = 1, Q_MDP(s, defer) = 0 when max_s b(s) < theta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# POMDP tuple
# ---------------------------------------------------------------------------

@dataclass
class POMDP:
    """
    Formal POMDP tuple (S, A, T, O, R, gamma).

    T has shape (|A|, |S|, |S|) with T[a, s, s'] = P(s' | s, a).
    O has shape (|A|, |S|, |O|) with O[a, s', o] = P(o | s', a).
    R has shape (|S|, |A|) with R[s, a] = expected immediate reward.
    """

    states: Tuple[str, ...]
    actions: Tuple[str, ...]
    observations: Tuple[str, ...]
    transition: np.ndarray       # (|A|, |S|, |S|)
    observation: np.ndarray      # (|A|, |S|, |O|)
    reward: Optional[np.ndarray] = None  # (|S|, |A|), optional for belief-only ops
    gamma: float = 0.95

    def __post_init__(self):
        n_s = len(self.states)
        n_a = len(self.actions)
        n_o = len(self.observations)
        if self.transition.shape != (n_a, n_s, n_s):
            raise ValueError(
                f"transition shape {self.transition.shape} != ({n_a}, {n_s}, {n_s})"
            )
        if self.observation.shape != (n_a, n_s, n_o):
            raise ValueError(
                f"observation shape {self.observation.shape} != ({n_a}, {n_s}, {n_o})"
            )
        # Validate stochasticity
        if not np.allclose(self.transition.sum(axis=2), 1.0, atol=1e-6):
            raise ValueError("transition rows must sum to 1 (stochastic)")
        if not np.allclose(self.observation.sum(axis=2), 1.0, atol=1e-6):
            raise ValueError("observation rows must sum to 1 (stochastic)")
        if self.reward is not None and self.reward.shape != (n_s, n_a):
            raise ValueError(
                f"reward shape {self.reward.shape} != ({n_s}, {n_a})"
            )
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def n_observations(self) -> int:
        return len(self.observations)


# ---------------------------------------------------------------------------
# Bayesian belief filter
# ---------------------------------------------------------------------------

def belief_update(
    pomdp: POMDP,
    belief: np.ndarray,
    action_idx: int,
    observation_idx: int,
) -> np.ndarray:
    """
    Bayesian belief filter for POMDPs. Given current belief b over S, action
    a_t, and observation o_{t+1}, compute the posterior belief:

        b'(s') = eta * O(s', a, o) * sum_s T(s, a, s') * b(s)

    where eta is the normalising constant. The marginal P(o|b,a) =
    1 / eta is returned implicitly through the normalisation.

    Args:
        pomdp: the POMDP tuple.
        belief: shape (|S|,), current belief distribution. Must be a
            probability distribution (non-negative, sums to 1).
        action_idx: index into pomdp.actions.
        observation_idx: index into pomdp.observations.

    Returns:
        Posterior belief distribution, shape (|S|,), guaranteed to sum to 1.

    Raises:
        ValueError if P(o | b, a) = 0 (impossible observation under current
        belief — degenerate case that cannot be Bayes-updated).
    """
    belief = np.asarray(belief, dtype=np.float64)
    if belief.shape != (pomdp.n_states,):
        raise ValueError(
            f"belief shape {belief.shape} != ({pomdp.n_states},)"
        )
    if (belief < -1e-9).any():
        raise ValueError("belief must be non-negative")
    if not np.isclose(belief.sum(), 1.0, atol=1e-6):
        raise ValueError(f"belief must sum to 1, got {belief.sum()}")

    # Predictive step: b_pred(s') = sum_s T(s, a, s') * b(s)
    T_a = pomdp.transition[action_idx]       # (|S|, |S|)
    b_pred = T_a.T @ belief                  # (|S|,)

    # Observation step: b'(s') proportional to O(s', a, o) * b_pred(s')
    O_a_o = pomdp.observation[action_idx, :, observation_idx]  # (|S|,)
    b_unnorm = O_a_o * b_pred
    marginal = b_unnorm.sum()
    if marginal <= 1e-12:
        raise ValueError(
            f"P(o={pomdp.observations[observation_idx]} | b, a="
            f"{pomdp.actions[action_idx]}) = 0; observation impossible under "
            f"current belief, cannot Bayes-update"
        )
    return b_unnorm / marginal


# ---------------------------------------------------------------------------
# Q-MDP commit rule (Littman, Cassandra & Kaelbling 1995)
# ---------------------------------------------------------------------------

def q_mdp_values(pomdp: POMDP, n_iterations: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    Compute Q_MDP(s, a) via value iteration on the UNDERLYING fully-observable
    MDP. This treats the POMDP as if the state were observable, and solves for
    the optimal Q-function under that simplification. The result is used as
    the Q-MDP commit rule's inputs.

    Requires pomdp.reward to be set.

    Returns:
        Q with shape (|S|, |A|).
    """
    if pomdp.reward is None:
        raise ValueError("pomdp.reward must be set for q_mdp_values")
    n_s = pomdp.n_states
    n_a = pomdp.n_actions
    V = np.zeros(n_s, dtype=np.float64)
    for _ in range(n_iterations):
        Q = np.zeros((n_s, n_a), dtype=np.float64)
        for a in range(n_a):
            # Q(s, a) = R(s, a) + gamma * sum_s' T(s, a, s') * V(s')
            Q[:, a] = pomdp.reward[:, a] + pomdp.gamma * pomdp.transition[a] @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q_final = np.zeros((n_s, n_a), dtype=np.float64)
    for a in range(n_a):
        Q_final[:, a] = pomdp.reward[:, a] + pomdp.gamma * pomdp.transition[a] @ V
    return Q_final


def q_mdp_commit(
    pomdp: POMDP,
    belief: np.ndarray,
    q_values: Optional[np.ndarray] = None,
) -> int:
    """
    Q-MDP commit rule (Littman et al. 1995 equation 4):

        pi_QMDP(b) = argmax_a sum_s b(s) * Q_MDP(s, a)

    Args:
        pomdp: the POMDP tuple.
        belief: shape (|S|,), probability distribution over states.
        q_values: optional pre-computed Q with shape (|S|, |A|). If None,
            computed on the fly via q_mdp_values.

    Returns:
        Index of the chosen action.
    """
    if q_values is None:
        q_values = q_mdp_values(pomdp)
    expected_q = belief @ q_values   # (|A|,)
    return int(np.argmax(expected_q))


# ---------------------------------------------------------------------------
# Governance commit predicate (PinkMate-specific)
# ---------------------------------------------------------------------------

def governance_commit(belief: np.ndarray, theta: float = 0.75) -> bool:
    """
    PinkMate's governance commit predicate (paper Section III.A, Slice 1):

        commit(b, theta) := max_s b(s) >= theta

    Formally: a degenerate Q-MDP commit with A = {commit, defer},
    Q_MDP(s, commit) proportional to 1_{max_s b(s) >= theta},
    Q_MDP(s, defer)  proportional to 1 - above.

    This function is the pure-math reference; the production path is
    `pinkmate/qmdp.py::SuperposedBelief.governance_commit(theta)`.

    Args:
        belief: shape (|S|,), probability distribution.
        theta: commit threshold, default 0.75 (matches PinkMate default).

    Returns:
        True iff the belief concentrates mass above theta on the dominant
        state — i.e. the system is confident enough to commit.
    """
    belief = np.asarray(belief, dtype=np.float64)
    if (belief < -1e-9).any():
        raise ValueError("belief must be non-negative")
    if not np.isclose(belief.sum(), 1.0, atol=1e-6):
        raise ValueError(f"belief must sum to 1, got {belief.sum()}")
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0, 1], got {theta}")
    return bool(belief.max() >= theta)


# ---------------------------------------------------------------------------
# PinkMate canonical POMDP (MQL -> SQL -> ClosedWon)
# ---------------------------------------------------------------------------

def build_pinkmate_canonical_pomdp(
    gamma: float = 0.95,
    convert_reward: float = 1.0,
    nurture_cost: float = 0.02,
    disqualify_cost: float = 0.1,
) -> POMDP:
    """
    Reference POMDP matching PinkMate's MQL -> SQL -> ClosedWon state chain.

    States: (MQL, SQL, ClosedWon, Disqualified)
    Actions: (score, nurture, convert)
    Observations: (engaged, not_engaged)

    Transition structure (per action):
        score:     deterministic self-loop (observation step only)
        nurture:   gentle MQL -> SQL progression (0.3 per step), small
                   disqualification risk
        convert:   SQL -> ClosedWon when SQL is dominant, otherwise
                   large disqualification risk

    Observation structure:
        P(engaged | SQL, *) = 0.8
        P(engaged | MQL, *) = 0.4
        P(engaged | ClosedWon, *) = 0.95
        P(engaged | Disqualified, *) = 0.05
    """
    states = ("MQL", "SQL", "ClosedWon", "Disqualified")
    actions = ("score", "nurture", "convert")
    obs = ("engaged", "not_engaged")
    n_s, n_a, n_o = 4, 3, 2
    T = np.zeros((n_a, n_s, n_s))

    # score: self-loop
    for s in range(n_s):
        T[0, s, s] = 1.0

    # nurture
    T[1, 0, 0] = 0.68  # MQL stays MQL
    T[1, 0, 1] = 0.30  # MQL -> SQL (gentle progression)
    T[1, 0, 3] = 0.02  # MQL -> Disqualified
    T[1, 1, 1] = 0.85  # SQL stays SQL
    T[1, 1, 2] = 0.10  # SQL -> ClosedWon (small direct conversion)
    T[1, 1, 3] = 0.05  # SQL -> Disqualified
    T[1, 2, 2] = 1.0   # ClosedWon absorbing
    T[1, 3, 3] = 1.0   # Disqualified absorbing

    # convert
    T[2, 0, 0] = 0.40  # MQL stays (convert too early)
    T[2, 0, 3] = 0.60  # MQL -> Disqualified (bad convert)
    T[2, 1, 2] = 0.75  # SQL -> ClosedWon (good convert)
    T[2, 1, 3] = 0.25  # SQL -> Disqualified
    T[2, 2, 2] = 1.0   # ClosedWon absorbing
    T[2, 3, 3] = 1.0   # Disqualified absorbing

    # Observations
    O = np.zeros((n_a, n_s, n_o))
    p_engaged = np.array([0.4, 0.8, 0.95, 0.05])
    for a in range(n_a):
        for s in range(n_s):
            O[a, s, 0] = p_engaged[s]      # engaged
            O[a, s, 1] = 1.0 - p_engaged[s]  # not_engaged

    # Rewards
    R = np.array([
        # score, nurture, convert
        [0.0, -nurture_cost, -disqualify_cost],   # MQL
        [0.0, -nurture_cost, convert_reward],      # SQL
        [0.0, 0.0, 0.0],                           # ClosedWon
        [0.0, 0.0, 0.0],                           # Disqualified
    ])
    return POMDP(
        states=states,
        actions=actions,
        observations=obs,
        transition=T,
        observation=O,
        reward=R,
        gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Isomorphism checks (invariants between this module and pinkmate/qmdp.py)
# ---------------------------------------------------------------------------

def pinkmate_superposed_belief_as_pomdp_belief(
    weights: Dict[str, float],
    state_order: Tuple[str, ...] = ("MQL", "SQL", "ClosedWon", "Disqualified"),
) -> np.ndarray:
    """
    Adapter: convert PinkMate's SuperposedBelief dict representation into the
    POMDP module's dense ndarray representation. Proves the two are
    mathematically equivalent (just different data shapes).
    """
    b = np.zeros(len(state_order), dtype=np.float64)
    for i, s in enumerate(state_order):
        b[i] = float(weights.get(s, 0.0))
    total = b.sum()
    if total <= 1e-12:
        raise ValueError("SuperposedBelief weights sum to 0")
    return b / total
