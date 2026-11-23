"""
Shared pytest fixtures for the PinkMate test suite.

The `CANONICAL_SEED` constant and the `canonical_seed` fixture give every test
a single source of truth for reproducibility. Existing tests continue to use
their literal `seed=42` values; new tests from Slice 7.5 onwards should prefer
the fixture so a future "change the canonical seed to N" sweep is a one-line
edit here instead of a 30-call-site refactor.

Usage in new tests:

    def test_something(canonical_seed, fixed_rng):
        rng = fixed_rng
        # ... uses rng, implicit reproducibility

    def test_with_explicit_seed(canonical_seed):
        result = some_function(data, seed=canonical_seed)
        # ... uses the canonical seed value directly
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Canonical seed constant
# ---------------------------------------------------------------------------

CANONICAL_SEED: int = 42
"""
The one canonical seed used across the PinkMate test suite.

Chosen value 42 matches the literal most-used in the existing suite (Slice 2-7).
Changing this is a single-line edit that propagates to every test that opts
into `canonical_seed` / `fixed_rng` fixtures.

Properties the canonical seed must preserve:
    - Deterministic Bayesian NUTS sampling for `test_bayesian_survival.py`
      (converges to the correct posterior with 200 draws × 2 chains on the
      fixture's synthetic Weibull data).
    - Deterministic EM convergence for `test_mixture_survival.py` (two-regime
      scale ratio > 3 at K=2 on the fixture's synthetic two-component data).
    - Deterministic Spearman correlation p-values in `test_scoring_agent_schema`
      drift-warning tests.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_seed() -> int:
    """
    The canonical pytest seed, exposed as a fixture.

    Returns:
        The integer `CANONICAL_SEED`.
    """
    return CANONICAL_SEED


@pytest.fixture
def fixed_rng(canonical_seed: int) -> np.random.Generator:
    """
    A numpy `Generator` seeded with `CANONICAL_SEED`.

    Prefer this over `np.random.default_rng(42)` in new tests so the seed is
    centrally managed.

    Returns:
        A fresh `numpy.random.Generator` instance.
    """
    return np.random.default_rng(canonical_seed)


@pytest.fixture
def small_test_set_size() -> int:
    """
    Default sample size for fast synthetic test fixtures.

    Keeps the pytest runtime bounded on CI (currently ~3-4 minutes for the
    full 359-test suite). Tests that need a larger sample for statistical
    power should hard-code their size explicitly.

    Returns:
        100.
    """
    return 100
