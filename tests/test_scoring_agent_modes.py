"""
Slice 3.5 — ScoringAgent deployment-mode flag tests.

Two deployment modes, three deployment paths:

    1. model_source="zoo" with populated models/     — production-mode (Slice 5+)
    2. model_source="zoo" with empty models/         — research-reproduction
                                                       fallback path (Slice 2
                                                       default); emits a
                                                       RuntimeWarning.
    3. model_source="placeholder"                    — product-demo path; skips
                                                       model loading entirely,
                                                       no warnings, no file
                                                       reads.

This test suite targets paths 2 and 3 (the most common deployment contexts
on a fresh checkout). Path 1 is covered by the existing Slice 2 integration
tests when `python scripts/reproduce_table2.py` has been run to populate
`models/`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from agents.scoring_agent import ScoringAgent
from pinkmate.state import EnrichedRecord, LeadInput, PipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state() -> PipelineState:
    """Mock a lead that reaches the ScoringAgent with enrichment already done."""
    return PipelineState(
        lead_input=LeadInput(
            lead_id="SLICE35-001",
            industry="SaaS",
            funding_stage="Series_A",
            team_size=120,
            role="VP of Engineering",
        ),
        enriched=EnrichedRecord(
            lead_id="SLICE35-001",
            features={
                "industry_saas": 1,
                "funding_stage_series_a": 1,
                "team_size_50_200": 1,
                "role_seniority_score": 0.9,
                "hiring_for_gtm": 1,
            },
        ),
    )


def _rwarnings(w_list) -> list:
    """Filter a warnings.catch_warnings buffer to just RuntimeWarnings."""
    return [str(x.message) for x in w_list if issubclass(x.category, RuntimeWarning)]


# ---------------------------------------------------------------------------
# Mode="zoo" — research-reproduction path (default, Slice 2 behaviour)
# ---------------------------------------------------------------------------

class TestZooMode:
    """
    Default ``model_source="zoo"`` preserves Slice 2 behaviour: attempts to
    load real models, falls back with a RuntimeWarning if the models/
    directory is empty / missing.
    """

    def test_zoo_mode_is_the_default(self):
        agent = ScoringAgent()
        assert agent.model_source == "zoo"

    def test_zoo_mode_attempts_model_load(self, tmp_path: Path):
        """Empty tmp_path as models_dir → 6/6 missing, warning emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = ScoringAgent(models_dir=str(tmp_path), model_source="zoo")
            runtime_warnings = _rwarnings(w)

        assert len(runtime_warnings) == 1, (
            f"Expected exactly 1 RuntimeWarning in zoo mode with empty "
            f"models_dir; got {len(runtime_warnings)}: {runtime_warnings}"
        )
        assert "6/6 model pickles missing" in runtime_warnings[0]
        # Models dict is populated with None entries (one per expected model).
        assert len(agent.models) == 6
        assert all(v is None for v in agent.models.values())
        assert agent._using_real_models is False

    def test_zoo_mode_runs_pipeline_end_to_end_via_placeholder_fallback(self, tmp_path: Path):
        """Zoo mode with empty models/ still runs scoring (via fallback)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress the expected warning
            agent = ScoringAgent(models_dir=str(tmp_path), model_source="zoo")
            result = agent(_build_state())

        assert result.score is not None
        assert 0.0 <= result.score.final_score <= 100.0
        assert result.belief is not None


# ---------------------------------------------------------------------------
# Mode="placeholder" — product-demo path (Slice 3.5)
# ---------------------------------------------------------------------------

class TestPlaceholderMode:
    """
    ``model_source="placeholder"`` is the clean product-demo path. No file
    reads, no warnings, no attempted inference against real pickles.
    """

    def test_placeholder_mode_emits_no_warnings(self, tmp_path: Path):
        """The core Slice 3.5 contract: no RuntimeWarning on construction."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = ScoringAgent(
                models_dir=str(tmp_path),  # even an empty dir should produce no warning
                model_source="placeholder",
            )
            runtime_warnings = _rwarnings(w)

        assert len(runtime_warnings) == 0, (
            f"Placeholder mode must emit NO RuntimeWarnings; got "
            f"{len(runtime_warnings)}: {runtime_warnings}"
        )

    def test_placeholder_mode_skips_model_loading(self, tmp_path: Path):
        agent = ScoringAgent(models_dir=str(tmp_path), model_source="placeholder")
        assert agent.models == {}, (
            f"Placeholder mode must not populate models dict; got "
            f"{len(agent.models)} entries"
        )
        assert agent._using_real_models is False

    def test_placeholder_mode_does_not_touch_filesystem(self, tmp_path: Path):
        """
        Even if models_dir points at a path that does not exist, placeholder
        mode must not fail (no file reads attempted).
        """
        nonexistent = tmp_path / "definitely" / "not" / "here"
        assert not nonexistent.exists()

        # Should construct cleanly.
        agent = ScoringAgent(models_dir=str(nonexistent), model_source="placeholder")
        assert agent.models == {}

    def test_placeholder_mode_runs_pipeline_end_to_end(self, tmp_path: Path):
        """Placeholder mode produces a valid score via the Slice 1 predictor."""
        agent = ScoringAgent(models_dir=str(tmp_path), model_source="placeholder")
        result = agent(_build_state())

        assert result.score is not None
        assert 0.0 <= result.score.final_score <= 100.0
        assert result.belief is not None
        assert result.governance_decision is not None

    def test_placeholder_mode_uses_slice1_predictor(self, tmp_path: Path):
        """
        The placeholder predictor is the Slice 1 deterministic weighted-linear
        model. For a textbook "high-fit" lead, it should score 100.
        """
        agent = ScoringAgent(models_dir=str(tmp_path), model_source="placeholder")
        result = agent(_build_state())
        # Textbook high-fit features: 15+15+15+36+15 = 96 (score/100).
        # Actually: role_seniority_score * 40 = 36; total = 96.
        assert result.score.final_score == pytest.approx(96.0, abs=0.5)


# ---------------------------------------------------------------------------
# Equivalence contract — once Slice 5 trains GTM-matching models,
# placeholder-mode output and zoo-mode output should converge.
# ---------------------------------------------------------------------------

class TestZooAndPlaceholderEquivalence:
    """
    When `zoo` mode has no loaded models to run against (e.g., empty
    models_dir, as is the case on a fresh checkout before Slice 5 ships),
    its scoring path falls back to the same Slice 1 placeholder predictor
    that placeholder mode uses. The two modes therefore produce IDENTICAL
    final scores on identical input.

    This test pins that contract so a future change to the placeholder
    predictor does not accidentally diverge the two modes.
    """

    def test_identical_scores_on_identical_input(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zoo_agent = ScoringAgent(models_dir=str(tmp_path), model_source="zoo")
        placeholder_agent = ScoringAgent(
            models_dir=str(tmp_path), model_source="placeholder"
        )

        zoo_result = zoo_agent(_build_state())
        placeholder_result = placeholder_agent(_build_state())

        assert zoo_result.score.final_score == placeholder_result.score.final_score
        assert zoo_result.belief.as_dict() == placeholder_result.belief.as_dict()
