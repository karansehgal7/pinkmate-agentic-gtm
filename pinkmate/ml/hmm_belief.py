"""
Hidden Markov Model over latent buyer states (Slice 7).

Operationalises the Section 7 formalisation: the observable touch sequence
emitted by a lead is modelled as drawn from a latent-state HMM where the
hidden states are the Q-MDP belief-state primitives (MQL, SQL, ClosedWon).
Viterbi decoding recovers the most-likely latent trajectory; the
forward-backward algorithm gives the filtered belief at each step.

Reference:
    Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proceedings of the IEEE, 77(2),
    257-286.

Backed by hmmlearn >=0.3.0's CategoricalHMM. We wrap it in a thin
class that exposes a PinkMate-flavoured API (state labels, discretised
touch bins, Viterbi + forward-backward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_LATENT_STATES: Tuple[str, ...] = ("MQL", "SQL", "ClosedWon")
DEFAULT_N_OBS_BINS: int = 4   # touch-count discretisation: {0, 1-3, 4-10, 11+}


# ---------------------------------------------------------------------------
# Observation discretisation
# ---------------------------------------------------------------------------

def discretise_touch_count(
    touches: Sequence[int],
    n_bins: int = DEFAULT_N_OBS_BINS,
    boundaries: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Convert raw touch counts into categorical observation indices.

    Default boundaries (for n_bins=4): {0}, {1-3}, {4-10}, {11+}
    maps to observation indices 0, 1, 2, 3.

    Args:
        touches: iterable of non-negative integer touch counts.
        n_bins: number of discrete observation levels.
        boundaries: optional explicit upper bounds for each bin except the
            last (which is open-ended). If None, uses defaults for n_bins=4.
    """
    if boundaries is None:
        if n_bins == 4:
            boundaries = [0, 3, 10]  # upper bounds for bins 0, 1, 2
        else:
            # Simple linear-log boundaries
            boundaries = [2 ** i - 1 for i in range(1, n_bins)]
    out = np.zeros(len(touches), dtype=np.int64)
    for i, t in enumerate(touches):
        bin_idx = 0
        for b_idx, upper in enumerate(boundaries):
            if t <= upper:
                bin_idx = b_idx
                break
        else:
            bin_idx = n_bins - 1
        out[i] = bin_idx
    return out


# ---------------------------------------------------------------------------
# HMM wrapper
# ---------------------------------------------------------------------------

@dataclass
class LatentStateHMM:
    """
    HMM over PinkMate latent buyer states with categorical touch observations.

    Attributes:
        state_labels: ordered tuple of latent-state names
            (default: MQL, SQL, ClosedWon).
        n_obs_bins: number of discrete observation categories.
        start_prob: initial state probabilities, shape (n_states,).
        trans_prob: transition matrix, shape (n_states, n_states).
        emit_prob: emission matrix, shape (n_states, n_obs_bins).
    """

    state_labels: Tuple[str, ...] = DEFAULT_LATENT_STATES
    n_obs_bins: int = DEFAULT_N_OBS_BINS
    start_prob: Optional[np.ndarray] = None
    trans_prob: Optional[np.ndarray] = None
    emit_prob: Optional[np.ndarray] = None
    random_state: int = 42

    _model = None

    def __post_init__(self):
        from hmmlearn.hmm import CategoricalHMM
        self._CategoricalHMM = CategoricalHMM
        n = len(self.state_labels)
        if self.start_prob is None:
            # Default: start concentrated at MQL (typical B2B funnel entry)
            self.start_prob = np.zeros(n)
            self.start_prob[0] = 1.0
        if self.trans_prob is None:
            # Default: slow left-to-right progression with small self-loop
            self.trans_prob = _default_transition_matrix(n)
        if self.emit_prob is None:
            # Default: higher-state leads emit more touches on average
            self.emit_prob = _default_emission_matrix(n, self.n_obs_bins)
        self._validate_matrices()
        self._initialise_model()

    def _validate_matrices(self):
        n = len(self.state_labels)
        if self.start_prob.shape != (n,):
            raise ValueError(
                f"start_prob shape {self.start_prob.shape} != ({n},)"
            )
        if self.trans_prob.shape != (n, n):
            raise ValueError(
                f"trans_prob shape {self.trans_prob.shape} != ({n}, {n})"
            )
        if self.emit_prob.shape != (n, self.n_obs_bins):
            raise ValueError(
                f"emit_prob shape {self.emit_prob.shape} != ({n}, {self.n_obs_bins})"
            )
        if not np.isclose(self.start_prob.sum(), 1.0, atol=1e-6):
            raise ValueError("start_prob must sum to 1")
        if not np.allclose(self.trans_prob.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("trans_prob rows must sum to 1")
        if not np.allclose(self.emit_prob.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("emit_prob rows must sum to 1")

    def _initialise_model(self):
        n = len(self.state_labels)
        self._model = self._CategoricalHMM(
            n_components=n,
            n_features=self.n_obs_bins,
            random_state=self.random_state,
            init_params="",  # we provide our own
            params="ste",
        )
        self._model.startprob_ = self.start_prob.astype(np.float64)
        self._model.transmat_ = self.trans_prob.astype(np.float64)
        self._model.emissionprob_ = self.emit_prob.astype(np.float64)

    @property
    def n_states(self) -> int:
        return len(self.state_labels)

    # --- Decoding ---

    def viterbi(self, observations: Sequence[int]) -> Tuple[np.ndarray, float]:
        """
        Viterbi decoding: most-likely latent-state sequence.

        Returns (state_sequence, log_likelihood).
        """
        obs = np.asarray(observations, dtype=np.int64).reshape(-1, 1)
        log_prob, seq = self._model.decode(obs, algorithm="viterbi")
        return seq, float(log_prob)

    def forward_backward(self, observations: Sequence[int]) -> np.ndarray:
        """
        Forward-backward smoothing: posterior P(s_t | o_1,...,o_T) for every t.

        Returns gamma matrix, shape (T, n_states), each row sums to 1.
        """
        obs = np.asarray(observations, dtype=np.int64).reshape(-1, 1)
        _, posteriors = self._model.score_samples(obs)
        return np.asarray(posteriors)

    def filtered_belief_at_end(self, observations: Sequence[int]) -> np.ndarray:
        """Filtered belief P(s_T | o_1,...,o_T) at the final step."""
        gamma = self.forward_backward(observations)
        return gamma[-1]

    def log_likelihood(self, observations: Sequence[int]) -> float:
        """Log P(o_1,...,o_T) under the HMM."""
        obs = np.asarray(observations, dtype=np.int64).reshape(-1, 1)
        return float(self._model.score(obs))

    # --- Learning (optional; defaults encode PinkMate's prior) ---

    def fit(
        self,
        observation_sequences: Sequence[Sequence[int]],
        n_iter: int = 50,
        tol: float = 1e-4,
    ) -> "LatentStateHMM":
        """
        Baum-Welch EM re-estimation of start_prob, trans_prob, emit_prob from
        observation data. Useful when a real dataset of touch sequences exists.

        Args:
            observation_sequences: list of touch-count sequences
                (one sequence per lead). Each sequence is already discretised
                via `discretise_touch_count`.
        """
        if not observation_sequences:
            raise ValueError("observation_sequences is empty")
        self._model.n_iter = n_iter
        self._model.tol = tol
        # hmmlearn wants concatenated observations + lengths vector
        lengths = [len(s) for s in observation_sequences]
        concat = np.concatenate(
            [np.asarray(s, dtype=np.int64) for s in observation_sequences]
        ).reshape(-1, 1)
        self._model.fit(concat, lengths=lengths)
        self.start_prob = self._model.startprob_.copy()
        self.trans_prob = self._model.transmat_.copy()
        self.emit_prob = self._model.emissionprob_.copy()
        return self


# ---------------------------------------------------------------------------
# Default matrices (encode PinkMate's prior)
# ---------------------------------------------------------------------------

def _default_transition_matrix(n: int) -> np.ndarray:
    """
    Default transition matrix: slow left-to-right progression with self-loops.
    Encodes the assumption that leads progress monotonically through the funnel
    (MQL -> SQL -> ClosedWon) but can linger in any state.
    """
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        T[i, i] = 0.7       # self-loop
        if i + 1 < n:
            T[i, i + 1] = 0.3   # forward progression
        else:
            T[i, i] = 1.0       # absorbing final state
    # Renormalise rows (safety)
    T = T / T.sum(axis=1, keepdims=True)
    return T


def _default_emission_matrix(n_states: int, n_obs: int) -> np.ndarray:
    """
    Default emission matrix: higher-state leads emit more touches on average.
    Each state emits a monotonically increasing distribution over observation
    bins (0 = no touches, higher bins = more touches).
    """
    E = np.zeros((n_states, n_obs), dtype=np.float64)
    for s in range(n_states):
        # Centre the state's emission distribution shifted higher
        center = (s + 0.5) * (n_obs - 1) / n_states
        for o in range(n_obs):
            E[s, o] = np.exp(-((o - center) ** 2) / 0.8)
    E = E / E.sum(axis=1, keepdims=True)
    return E
