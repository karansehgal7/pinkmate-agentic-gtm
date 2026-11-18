"""Tests for pinkmate.ml.hmm_belief — latent-state HMM."""
from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.hmm_belief import (
    DEFAULT_LATENT_STATES,
    DEFAULT_N_OBS_BINS,
    LatentStateHMM,
    discretise_touch_count,
)


class TestDiscretiseTouchCount:
    def test_zero_maps_to_bin_zero(self):
        out = discretise_touch_count([0])
        assert out[0] == 0

    def test_default_boundaries_four_bins(self):
        out = discretise_touch_count([0, 2, 7, 25])
        assert out.tolist() == [0, 1, 2, 3]

    def test_explicit_boundaries(self):
        out = discretise_touch_count([1, 5, 10], n_bins=3, boundaries=[2, 7])
        assert out.tolist() == [0, 1, 2]

    def test_output_shape_matches_input(self):
        out = discretise_touch_count(list(range(20)))
        assert out.shape == (20,)


class TestHMMConstruction:
    def test_defaults(self):
        h = LatentStateHMM()
        assert h.n_states == 3
        assert h.state_labels == DEFAULT_LATENT_STATES
        assert h.n_obs_bins == DEFAULT_N_OBS_BINS

    def test_trans_prob_rows_sum_to_one(self):
        h = LatentStateHMM()
        np.testing.assert_allclose(h.trans_prob.sum(axis=1), 1.0)

    def test_emit_prob_rows_sum_to_one(self):
        h = LatentStateHMM()
        np.testing.assert_allclose(h.emit_prob.sum(axis=1), 1.0)

    def test_invalid_start_prob_raises(self):
        with pytest.raises(ValueError, match="start_prob"):
            LatentStateHMM(start_prob=np.array([0.5, 0.3]))  # wrong size


class TestViterbi:
    def test_returns_sequence_of_correct_length(self):
        h = LatentStateHMM()
        seq, logL = h.viterbi([0, 1, 2, 3])
        assert len(seq) == 4
        assert np.isfinite(logL)

    def test_most_likely_states_are_valid_indices(self):
        h = LatentStateHMM()
        seq, _ = h.viterbi([0, 1, 2, 3, 3])
        for s in seq:
            assert 0 <= s < h.n_states

    def test_high_touch_sequence_ends_in_higher_state(self):
        """A sequence of high-touch observations should end in SQL or ClosedWon, not MQL."""
        h = LatentStateHMM()
        seq, _ = h.viterbi([3, 3, 3, 3, 3])
        # With default emission bias (higher states emit more touches),
        # a persistent high-touch sequence should NOT end in MQL (index 0).
        assert seq[-1] != 0


class TestForwardBackward:
    def test_posterior_rows_sum_to_one(self):
        h = LatentStateHMM()
        gamma = h.forward_backward([0, 1, 2, 3, 3])
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_posterior_non_negative(self):
        h = LatentStateHMM()
        gamma = h.forward_backward([0, 1, 2, 3, 3])
        assert (gamma >= -1e-9).all()

    def test_filtered_belief_shape(self):
        h = LatentStateHMM()
        belief = h.filtered_belief_at_end([0, 1, 2])
        assert belief.shape == (3,)
        assert belief.sum() == pytest.approx(1.0)

    def test_log_likelihood_finite(self):
        h = LatentStateHMM()
        ll = h.log_likelihood([0, 1, 2, 3])
        assert np.isfinite(ll)
