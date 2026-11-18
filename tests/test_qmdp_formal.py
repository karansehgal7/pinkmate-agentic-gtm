"""Tests for pinkmate.qmdp_formal — POMDP formalism."""
from __future__ import annotations

import numpy as np
import pytest

from pinkmate.qmdp_formal import (
    POMDP,
    belief_update,
    build_pinkmate_canonical_pomdp,
    governance_commit,
    pinkmate_superposed_belief_as_pomdp_belief,
    q_mdp_commit,
    q_mdp_values,
)


@pytest.fixture
def tiger_pomdp():
    """
    Classic 2-state tiger POMDP (Cassandra, Kassirer, Kaelbling 1998).
    S = {tiger_left, tiger_right}, A = {listen, open_left, open_right}
    """
    T = np.zeros((3, 2, 2))
    for a in range(3):
        if a == 0:  # listen: state unchanged
            T[a] = np.eye(2)
        else:  # open: reset to uniform (episode ends)
            T[a] = np.array([[0.5, 0.5], [0.5, 0.5]])
    O = np.zeros((3, 2, 2))
    # listen: hear tiger correctly 85%
    O[0, 0, 0] = 0.85  # tiger_left -> hear_left
    O[0, 0, 1] = 0.15
    O[0, 1, 0] = 0.15
    O[0, 1, 1] = 0.85
    # open actions: uniform observations (episode reset)
    for a in [1, 2]:
        O[a] = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([
        [-1.0, -100.0, 10.0],   # tiger_left: listen costs 1, open_left is bad (-100), open_right safe (+10)
        [-1.0, 10.0, -100.0],   # tiger_right: mirror
    ])
    return POMDP(
        states=("tiger_left", "tiger_right"),
        actions=("listen", "open_left", "open_right"),
        observations=("hear_left", "hear_right"),
        transition=T, observation=O, reward=R, gamma=0.95,
    )


@pytest.fixture
def pinkmate_pomdp():
    return build_pinkmate_canonical_pomdp()


class TestPOMDPConstruction:
    def test_pinkmate_pomdp_well_formed(self, pinkmate_pomdp):
        assert pinkmate_pomdp.n_states == 4
        assert pinkmate_pomdp.n_actions == 3
        assert pinkmate_pomdp.n_observations == 2

    def test_non_stochastic_transition_raises(self):
        T = np.ones((1, 2, 2))  # rows don't sum to 1
        O = np.ones((1, 2, 2)) * 0.5
        with pytest.raises(ValueError, match="transition rows"):
            POMDP(
                states=("a", "b"), actions=("x",), observations=("o1", "o2"),
                transition=T, observation=O,
            )

    def test_invalid_gamma_raises(self, pinkmate_pomdp):
        with pytest.raises(ValueError, match="gamma"):
            POMDP(
                states=pinkmate_pomdp.states,
                actions=pinkmate_pomdp.actions,
                observations=pinkmate_pomdp.observations,
                transition=pinkmate_pomdp.transition,
                observation=pinkmate_pomdp.observation,
                reward=pinkmate_pomdp.reward,
                gamma=1.5,
            )


class TestBeliefUpdate:
    def test_belief_sums_to_one(self, pinkmate_pomdp):
        b = np.array([0.25, 0.25, 0.25, 0.25])
        b2 = belief_update(pinkmate_pomdp, b, action_idx=1, observation_idx=0)
        assert b2.sum() == pytest.approx(1.0)

    def test_belief_non_negative(self, pinkmate_pomdp):
        b = np.array([1.0, 0, 0, 0])
        b2 = belief_update(pinkmate_pomdp, b, action_idx=1, observation_idx=0)
        assert (b2 >= -1e-9).all()

    def test_known_answer_tiger_listen_hear_left(self, tiger_pomdp):
        """Starting uniform, after listen + hear_left, belief should shift to tiger_left."""
        b0 = np.array([0.5, 0.5])
        b1 = belief_update(tiger_pomdp, b0, action_idx=0, observation_idx=0)
        # Hand-computed: b1(left) = 0.85 * 0.5 / (0.85 * 0.5 + 0.15 * 0.5) = 0.85
        assert b1[0] == pytest.approx(0.85, abs=0.01)
        assert b1[1] == pytest.approx(0.15, abs=0.01)

    def test_impossible_observation_raises(self, tiger_pomdp):
        """Point-mass belief + observation with zero probability => raise."""
        # Construct a POMDP variant where hearing is perfect
        T = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        O = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        pomdp = POMDP(
            states=("a", "b"), actions=("x",), observations=("o1", "o2"),
            transition=T, observation=O,
        )
        b = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="impossible"):
            belief_update(pomdp, b, action_idx=0, observation_idx=1)

    def test_belief_shape_mismatch_raises(self, pinkmate_pomdp):
        with pytest.raises(ValueError, match="belief shape"):
            belief_update(pinkmate_pomdp, np.array([0.5, 0.5]), 0, 0)


class TestQMDPCommit:
    def test_q_values_shape(self, pinkmate_pomdp):
        Q = q_mdp_values(pinkmate_pomdp)
        assert Q.shape == (pinkmate_pomdp.n_states, pinkmate_pomdp.n_actions)

    def test_q_mdp_commit_prefers_convert_on_sql_dominant_belief(self, pinkmate_pomdp):
        """When belief concentrates on SQL, Q-MDP should recommend 'convert'."""
        b = np.array([0.05, 0.85, 0.05, 0.05])  # SQL-dominant
        action = q_mdp_commit(pinkmate_pomdp, b)
        assert pinkmate_pomdp.actions[action] == "convert"

    def test_q_mdp_commit_prefers_nurture_on_mql_dominant_belief(self, pinkmate_pomdp):
        """When belief concentrates on MQL, Q-MDP should avoid immediate convert."""
        b = np.array([0.85, 0.05, 0.05, 0.05])  # MQL-dominant
        action = q_mdp_commit(pinkmate_pomdp, b)
        assert pinkmate_pomdp.actions[action] != "convert"


class TestGovernanceCommit:
    def test_commits_when_max_above_theta(self):
        assert governance_commit(np.array([0.8, 0.1, 0.1]), theta=0.75) is True

    def test_defers_when_max_below_theta(self):
        assert governance_commit(np.array([0.5, 0.3, 0.2]), theta=0.75) is False

    def test_invalid_belief_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            governance_commit(np.array([0.5, 0.5, 0.5]))


class TestIsomorphismWithPinkMateSuperposedBelief:
    def test_dict_to_ndarray_adapter(self):
        weights = {"MQL": 0.4, "SQL": 0.3, "ClosedWon": 0.2, "Disqualified": 0.1}
        b = pinkmate_superposed_belief_as_pomdp_belief(weights)
        assert b.shape == (4,)
        assert b.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(b, [0.4, 0.3, 0.2, 0.1])

    def test_unnormalised_weights_get_normalised(self):
        weights = {"MQL": 1.0, "SQL": 1.0, "ClosedWon": 1.0, "Disqualified": 1.0}
        b = pinkmate_superposed_belief_as_pomdp_belief(weights)
        assert b.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(b, [0.25, 0.25, 0.25, 0.25])
