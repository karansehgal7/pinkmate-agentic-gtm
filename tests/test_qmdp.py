"""
Unit tests on the Q-MDP formal constructs (pinkmate/qmdp.py).

Reference: Sehgal & Bhatti (2026), IEEE Cyber-AI 2026 Submission #4554, §III.A.
"""

from __future__ import annotations

import math
import pytest

from pinkmate.qmdp import (
    BeliefState,
    SuperposedBelief,
    GovernanceCommit,
    governance_commit,
    transition,
    DEFAULT_THETA_GOVERNANCE,
)


# ---------------------------------------------------------------------------
# BeliefState invariants
# ---------------------------------------------------------------------------

class TestBeliefState:
    def test_valid_construction(self):
        b = BeliefState(state="QUALIFIED", probability=0.8)
        assert b.state == "QUALIFIED"
        assert b.probability == pytest.approx(0.8)

    def test_probability_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            BeliefState(state="QUALIFIED", probability=1.5)
        with pytest.raises(ValueError, match="must be in"):
            BeliefState(state="QUALIFIED", probability=-0.1)

    def test_empty_state_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            BeliefState(state="", probability=0.5)

    def test_is_frozen(self):
        b = BeliefState(state="QUALIFIED", probability=0.8)
        with pytest.raises((AttributeError, Exception)):
            b.probability = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SuperposedBelief invariants
# ---------------------------------------------------------------------------

class TestSuperposedBelief:
    def test_valid_two_state(self):
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("QUALIFIED", 0.8),
                BeliefState("UNQUALIFIED", 0.2),
            ),
            step=0,
        )
        assert sb.as_dict() == {"QUALIFIED": 0.8, "UNQUALIFIED": 0.2}
        assert sb.step == 0

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            SuperposedBelief(
                beliefs=(
                    BeliefState("A", 0.3),
                    BeliefState("B", 0.3),
                ),
                step=0,
            )

    def test_duplicate_state_labels_rejected(self):
        with pytest.raises(ValueError, match="duplicate state labels"):
            SuperposedBelief(
                beliefs=(
                    BeliefState("A", 0.5),
                    BeliefState("A", 0.5),
                ),
                step=0,
            )

    def test_empty_beliefs_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            SuperposedBelief(beliefs=(), step=0)

    def test_negative_step_rejected(self):
        with pytest.raises(ValueError, match="step must be"):
            SuperposedBelief(
                beliefs=(BeliefState("A", 1.0),),
                step=-1,
            )

    def test_leading_returns_argmax(self):
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("A", 0.2),
                BeliefState("B", 0.5),
                BeliefState("C", 0.3),
            ),
            step=1,
        )
        assert sb.leading().state == "B"


# ---------------------------------------------------------------------------
# governance_commit() — paper §III.A commit predicate
# ---------------------------------------------------------------------------

class TestGovernanceCommit:
    def test_commits_when_leading_exceeds_theta(self):
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("QUALIFIED", 0.85),
                BeliefState("UNQUALIFIED", 0.15),
            ),
            step=0,
        )
        result = governance_commit(sb, theta=0.75)
        assert result.committed is True
        assert result.state == "QUALIFIED"
        assert result.confidence == pytest.approx(0.85)
        assert result.theta == pytest.approx(0.75)

    def test_defers_when_leading_at_or_below_theta(self):
        # Paper's predicate is strict inequality (>), so equality defers.
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("QUALIFIED", 0.75),
                BeliefState("UNQUALIFIED", 0.25),
            ),
            step=0,
        )
        result = governance_commit(sb, theta=0.75)
        assert result.committed is False
        assert result.state is None
        assert result.confidence == pytest.approx(0.75)

    def test_defers_when_belief_well_below_theta(self):
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("QUALIFIED", 0.4),
                BeliefState("UNQUALIFIED", 0.6),
            ),
            step=0,
        )
        result = governance_commit(sb, theta=0.75)
        assert result.committed is False
        assert result.state is None
        assert result.confidence == pytest.approx(0.6)

    def test_default_theta_is_075(self):
        assert DEFAULT_THETA_GOVERNANCE == pytest.approx(0.75)

    def test_theta_out_of_range_rejected(self):
        sb = SuperposedBelief(beliefs=(BeliefState("A", 1.0),), step=0)
        with pytest.raises(ValueError, match="theta must be in"):
            governance_commit(sb, theta=0.0)
        with pytest.raises(ValueError, match="theta must be in"):
            governance_commit(sb, theta=1.0)
        with pytest.raises(ValueError, match="theta must be in"):
            governance_commit(sb, theta=1.5)


# ---------------------------------------------------------------------------
# transition() — belief dynamics per §III.A
# ---------------------------------------------------------------------------

class TestTransition:
    def test_deterministic_transition_preserves_normalisation(self):
        """A deterministic transition from a known state must yield a valid belief that sums to 1."""
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("S1", 1.0),
            ),
            step=0,
        )
        transition_matrix = {
            ("S1", "S2"): 1.0,
        }
        action_likelihood = {
            ("S1", "advance"): 1.0,
        }
        next_sb = transition(sb, "advance", transition_matrix, action_likelihood)
        assert next_sb.step == 1
        assert next_sb.as_dict() == {"S2": 1.0}

    def test_mixed_transition_renormalises(self):
        sb = SuperposedBelief(
            beliefs=(
                BeliefState("S1", 0.6),
                BeliefState("S2", 0.4),
            ),
            step=0,
        )
        # Both current states transition to S_next with equal probability under action.
        transition_matrix = {
            ("S1", "S_next"): 1.0,
            ("S2", "S_next"): 1.0,
        }
        action_likelihood = {
            ("S1", "a"): 1.0,
            ("S2", "a"): 1.0,
        }
        next_sb = transition(sb, "a", transition_matrix, action_likelihood)
        # Total belief mass collapses onto S_next and renormalises to 1.
        assert set(next_sb.as_dict().keys()) == {"S_next"}
        assert next_sb.as_dict()["S_next"] == pytest.approx(1.0)
        assert next_sb.step == 1

    def test_incompatible_action_raises(self):
        sb = SuperposedBelief(
            beliefs=(BeliefState("S1", 1.0),),
            step=0,
        )
        transition_matrix = {("S1", "S2"): 1.0}
        action_likelihood = {("S1", "irrelevant_action"): 0.0}
        with pytest.raises(ValueError, match="incompatible"):
            transition(sb, "unseen_action", transition_matrix, action_likelihood)
