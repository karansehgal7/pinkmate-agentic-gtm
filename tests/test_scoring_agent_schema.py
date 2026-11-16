"""
Slice 5.2 — Tests for ScoringAgent's hybrid fail-fast schema contract.

Standalone test file (not an extension of an existing test_scoring_agent.py)
because the pre-5.2 scoring agent tests lived inline in test_pipeline.py
and test_scoring_agent_modes.py. This file focuses specifically on the
Slice 5.2 Decision A contract:

    - STRICT on required features → RuntimeError
    - LENIENT on extras → silently dropped (DEBUG log only)
    - DRIFT WARNING on schema version mismatch → log + proceed

All 5 tests use a stub ``attribution.json`` written to a tmp directory
so no real trained-model files are needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pinkmate.state import LEADINPUT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_models_dir(tmp_path):
    """
    Write a stub attribution.json so ScoringAgent._get_training_features()
    and _get_schema_version() have something to read. No real pickles.
    """
    attribution_path = tmp_path / "attribution.json"
    payload = {
        "feature_names": [
            "industry_saas",
            "funding_stage_series_a",
            "team_size_50_200",
        ],
        "leadinput_schema_version": LEADINPUT_SCHEMA_VERSION,
        "with_smote": {},
    }
    with open(attribution_path, "w") as f:
        json.dump(payload, f)
    return tmp_path


@pytest.fixture
def stub_models_dir_drifted(tmp_path):
    """
    Variant: attribution.json with an OLD schema version to trigger drift.
    """
    attribution_path = tmp_path / "attribution.json"
    payload = {
        "feature_names": [
            "industry_saas",
            "funding_stage_series_a",
            "team_size_50_200",
        ],
        "leadinput_schema_version": "slice5_intermediate_v0",  # OLD
        "with_smote": {},
    }
    with open(attribution_path, "w") as f:
        json.dump(payload, f)
    return tmp_path


def _build_agent_with_stub(stub_dir: Path):
    """Helper: build ScoringAgent pointing at stub dir with fake loaded models."""
    from agents.scoring_agent import ScoringAgent
    agent = ScoringAgent(models_dir=str(stub_dir), model_source="zoo")
    # Force _using_real_models=True so the contract path fires
    # (otherwise agent goes to placeholder because no .pkl files exist)
    agent._using_real_models = True
    agent.models = {}  # No actual models — we test the contract, not the inference
    # Clear caches so each test reads fresh from the stubbed attribution.json
    agent._training_features_cache = None
    agent._model_schema_version_cache = None
    agent._schema_drift_warned = False
    return agent


# ---------------------------------------------------------------------------
# Slice 5.2 Hybrid fail-fast contract tests
# ---------------------------------------------------------------------------

class TestSchemaMatchStrict:
    """STRICT on required features → RuntimeError."""

    def test_missing_required_feature_raises_runtime_error(self, stub_models_dir):
        """Missing team_size_50_200 → RuntimeError with a clear message."""
        agent = _build_agent_with_stub(stub_models_dir)
        features = {
            "industry_saas": 1.0,
            "funding_stage_series_a": 1.0,
            # team_size_50_200 MISSING
        }
        with pytest.raises(RuntimeError, match="missing required features"):
            agent._predict_proba_real(features)

    def test_error_message_names_the_missing_feature(self, stub_models_dir):
        """Error message should list the missing feature name explicitly."""
        agent = _build_agent_with_stub(stub_models_dir)
        features = {
            "industry_saas": 1.0,
            "funding_stage_series_a": 1.0,
        }
        with pytest.raises(RuntimeError) as exc_info:
            agent._predict_proba_real(features)
        assert "team_size_50_200" in str(exc_info.value), (
            f"Error message should name 'team_size_50_200' as missing; "
            f"got: {exc_info.value}"
        )


class TestSchemaMatchLenient:
    """LENIENT on extras → silently dropped, no exception."""

    def test_extra_features_silently_dropped(self, stub_models_dir, caplog):
        """Extra features not in training schema → DEBUG log, no error."""
        agent = _build_agent_with_stub(stub_models_dir)
        features = {
            "industry_saas": 1.0,
            "funding_stage_series_a": 1.0,
            "team_size_50_200": 1.0,
            "an_unknown_extra_feature": 0.5,   # EXTRA — should be dropped
            "another_unknown_feature": 0.1,    # EXTRA — should be dropped
        }
        # Because self.models = {}, prediction will fall through to placeholder
        # after passing the schema check — that's fine, we're testing the
        # contract path, not the inference.
        with caplog.at_level(logging.DEBUG, logger="agents.scoring_agent"):
            try:
                agent._predict_proba_real(features)
            except Exception:
                pass  # placeholder fallback may or may not error; OK
        # The DEBUG log should mention dropping extras
        extras_logged = any(
            "dropping" in rec.message.lower() and "extra features" in rec.message.lower()
            for rec in caplog.records
        )
        assert extras_logged, (
            f"Expected DEBUG log mentioning dropping extras; got records: "
            f"{[r.message for r in caplog.records]}"
        )


class TestSchemaDriftWarning:
    """DRIFT WARNING on schema version mismatch → log + proceed (don't raise)."""

    def test_schema_version_drift_warns_but_proceeds(
        self, stub_models_dir_drifted, caplog,
    ):
        """OLD schema version → WARNING log, no RuntimeError."""
        agent = _build_agent_with_stub(stub_models_dir_drifted)
        features = {
            "industry_saas": 1.0,
            "funding_stage_series_a": 1.0,
            "team_size_50_200": 1.0,
        }
        with caplog.at_level(logging.WARNING, logger="agents.scoring_agent"):
            try:
                agent._predict_proba_real(features)
            except RuntimeError:
                pytest.fail(
                    "Schema drift should NOT raise RuntimeError — the hybrid "
                    "contract warns + proceeds on drift."
                )
            except Exception:
                pass  # placeholder fallback OK
        drift_logged = any(
            "schema drift" in rec.message.lower() for rec in caplog.records
        )
        assert drift_logged, (
            f"Expected WARNING about schema drift; got records: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_exact_schema_match_no_drift_warning(self, stub_models_dir, caplog):
        """Matching schema version → no drift warning."""
        agent = _build_agent_with_stub(stub_models_dir)
        features = {
            "industry_saas": 1.0,
            "funding_stage_series_a": 1.0,
            "team_size_50_200": 1.0,
        }
        with caplog.at_level(logging.WARNING, logger="agents.scoring_agent"):
            try:
                agent._predict_proba_real(features)
            except Exception:
                pass  # placeholder fallback OK
        drift_warnings = [
            rec for rec in caplog.records
            if "schema drift" in rec.message.lower()
        ]
        assert len(drift_warnings) == 0, (
            f"Unexpected drift warnings on matching schema: {drift_warnings}"
        )


class TestSchemaConstants:
    """Smoke tests on the LEADINPUT_SCHEMA_VERSION constant export."""

    def test_schema_version_constant_exported_from_state(self):
        """LEADINPUT_SCHEMA_VERSION is importable, non-empty, and slice-named."""
        assert isinstance(LEADINPUT_SCHEMA_VERSION, str)
        assert len(LEADINPUT_SCHEMA_VERSION) > 0
        assert "slice" in LEADINPUT_SCHEMA_VERSION.lower(), (
            f"Schema version should contain 'slice' marker; got "
            f"'{LEADINPUT_SCHEMA_VERSION}'"
        )
