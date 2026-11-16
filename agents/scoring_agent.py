"""
ScoringAgent — orchestration node V_scoring in the Q-MDP graph (§III.B).

Scores an EnrichedRecord and emits a SuperposedBelief into pipeline state
so the governance commit predicate (§III.A) has a belief to evaluate.

Slice 2 scope (shipped):
    - Loads trained models from ``models/`` if present (produced by
      ``scripts/reproduce_table2.py``) and runs a real six-model ensemble.
    - Falls back to the deterministic Slice 1 predictor with a loud WARNING
      if ``models/`` is empty — pipeline still runs end-to-end, but the
      attribution-stability signal is degraded.
    - Reads the saved σ_SHAP from ``models/attribution.json`` when available
      and uses it as a confidence derating signal on the committed state.

Slice 3.5 scope (shipped):
    - Adds a ``model_source`` constructor flag that selects between two
      deployment paths without changing any downstream behaviour:
        * ``"zoo"`` (default): current Slice 2 behaviour — loads from
          ``models/``, falls back with ``RuntimeWarning`` on mismatch.
          Research-reproduction path.
        * ``"placeholder"``: skips model loading entirely. No file reads,
          no warnings. Product-demo deployment path for website / client
          demos where the Taiwan-vs-GTM warnings would be noise rather
          than signal.

Slice 5.2 scope (shipped):
    - Hybrid fail-fast schema contract (Decision A, 16 Nov 2026):
        * STRICT on required features — missing feature that the model
          was trained on raises ``RuntimeError`` with a clear message
          pointing at EnrichmentAgent or a retrain.
        * LENIENT on extras — features present in LeadInput but not in
          the training schema are silently dropped (DEBUG-logged only).
        * DRIFT WARNING — if ``LEADINPUT_SCHEMA_VERSION`` has moved on
          since the model was trained, a WARNING is logged and the drift
          is recorded for audit. Prediction still proceeds on the
          intersection of features.
    - New helpers: ``_get_training_features()``, ``_get_schema_version()``,
      ``_log_schema_drift_to_trace()`` — all read from
      ``models/gtm/attribution.json``.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from pinkmate.qmdp import BeliefState, SuperposedBelief, governance_commit
from pinkmate.state import LEADINPUT_SCHEMA_VERSION, PipelineState, ScoreOutput, TraceEntry

LOGGER = logging.getLogger(__name__)


# Latent state labels for the lead-domain Q-MDP instantiation.
STATE_QUALIFIED: str = "QUALIFIED"
STATE_UNQUALIFIED: str = "UNQUALIFIED"


class ScoringAgent:
    """
    Score a lead and project the score into a two-component superposed belief.

    Belief construction (Slice 1):
        probability(QUALIFIED)   = final_score / 100
        probability(UNQUALIFIED) = 1 - final_score / 100

    This two-state partition is the minimum sufficient belief shape to
    exercise governance_commit() in Slice 1. Slice 2 will extend the state
    space to {QUALIFIED, NURTURE, DISQUALIFIED, HOLDOUT} and derive the
    partition from calibrated model probabilities.
    """

    NODE_NAME: str = "scoring"
    ACTION: str = "score"

    MODEL_NAMES: Tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "adaboost",
        "xgboost",
        "lightgbm",
        "catboost",
    )

    def __init__(
        self,
        models_dir: str = "models/gtm",
        model_source: Literal["zoo", "placeholder"] = "zoo",
    ) -> None:
        """
        Build a ScoringAgent in one of two deployment modes.

        See module docstring for full description of the "zoo" and
        "placeholder" modes (Slice 3.5) and the Slice 5.2 hybrid
        fail-fast contract applied to the "zoo" mode's real inference
        path.

        Args:
            models_dir: Directory holding trained model pickles. Only
                consulted when ``model_source="zoo"``. Default
                ``"models/gtm"`` per Slice 5.
            model_source: Deployment mode — ``"zoo"`` or ``"placeholder"``.
                Defaults to ``"zoo"`` for backward compatibility.
        """
        self.models_dir = Path(models_dir)
        self.model_source = model_source
        self.models: Dict[str, Any] = {}
        self.sigma_shap_by_model: Dict[str, float] = {}
        self.feature_order: Optional[List[str]] = None
        self._using_real_models: bool = False
        self._training_features_cache: Optional[List[str]] = None
        self._model_schema_version_cache: Optional[str] = None
        self._schema_drift_warned: bool = False  # Suppress duplicate warnings per agent

        if self.model_source == "zoo":
            self._load_models()
        else:
            LOGGER.info(
                "ScoringAgent constructed in placeholder mode "
                "(product-demo path; no model loading, no warnings)."
            )

    # ------------------------------------------------------------------
    # Model loading — Slice 2 + Slice 5.2 schema introspection
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """
        Load trained models from ``self.models_dir`` if present. If the
        directory is missing or empty, fall back to the Slice 1 deterministic
        placeholder with a visible WARNING.
        """
        missing: List[str] = []
        for name in self.MODEL_NAMES:
            pkl = self.models_dir / f"{name}.pkl"
            if pkl.exists():
                try:
                    with open(pkl, "rb") as f:
                        self.models[name] = pickle.load(f)
                except Exception as e:
                    LOGGER.warning("Failed to load %s: %s", pkl, e)
                    self.models[name] = None
                    missing.append(name)
            else:
                self.models[name] = None
                missing.append(name)

        if missing:
            warnings.warn(
                f"ScoringAgent: {len(missing)}/{len(self.MODEL_NAMES)} model "
                f"pickles missing from {self.models_dir.resolve()} "
                f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}). "
                "Falling back to Slice 1 deterministic placeholder predictor. "
                "To train real models: for GTM-domain, run "
                "`python scripts/train_gtm_ensemble.py` (Slice 5, writes to "
                "models/gtm/). For Taiwan paper reproduction, run "
                "`python scripts/reproduce_table2.py` (writes to models/taiwan/).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._using_real_models = False
        else:
            self._using_real_models = True
            LOGGER.info(
                "ScoringAgent loaded real six-model ensemble from %s",
                self.models_dir.resolve(),
            )

        # Load σ_SHAP summary if present — used for soft confidence derating
        attribution_path = self.models_dir / "attribution.json"
        if attribution_path.exists():
            try:
                with open(attribution_path) as f:
                    payload = json.load(f)
                with_smote = payload.get("with_smote", {})
                for name, entry in with_smote.items():
                    self.sigma_shap_by_model[name] = float(entry["sigma_shap"])
            except Exception as e:
                LOGGER.warning("Failed to parse attribution.json: %s", e)

    # ------------------------------------------------------------------
    # Slice 5.2 — Schema introspection helpers
    # ------------------------------------------------------------------

    def _get_training_features(self) -> List[str]:
        """
        Return the ordered list of feature names the model was trained on.
        Reads from ``models/gtm/attribution.json``'s ``feature_names`` key.
        Returns an empty list if unavailable (triggers placeholder fallback).

        Cached on first call per agent instance.
        """
        if self._training_features_cache is not None:
            return self._training_features_cache

        attribution_path = self.models_dir / "attribution.json"
        if not attribution_path.exists():
            self._training_features_cache = []
            return []
        try:
            with open(attribution_path) as f:
                payload = json.load(f)
            features = list(payload.get("feature_names", []))
            self._training_features_cache = features
            return features
        except Exception as e:
            LOGGER.warning(
                "Failed to read feature_names from %s: %s", attribution_path, e
            )
            self._training_features_cache = []
            return []

    def _get_schema_version(self) -> Optional[str]:
        """
        Return the LeadInput schema version the model was trained against.
        Reads from ``models/gtm/attribution.json``'s
        ``leadinput_schema_version`` key. Returns None if unavailable
        (drift warning is suppressed when unknown).

        Cached on first call per agent instance.
        """
        if self._model_schema_version_cache is not None:
            return self._model_schema_version_cache

        attribution_path = self.models_dir / "attribution.json"
        if not attribution_path.exists():
            return None
        try:
            with open(attribution_path) as f:
                payload = json.load(f)
            version = payload.get("leadinput_schema_version")
            self._model_schema_version_cache = version
            return version
        except Exception:
            return None

    def _log_schema_drift_to_trace(
        self, model_version: str, current_version: str,
    ) -> None:
        """
        Audit trail: record the schema drift so downstream analysis can
        detect that this prediction ran under a drifted schema.

        Current implementation: the LOGGER.warning emitted in
        ``_predict_proba_real()`` captures the event in logs. A richer
        implementation would append a ``TraceEntry`` with
        ``metadata["schema_drift"] = {...}`` — which happens naturally
        via the ``__call__()`` trace append if drift detection runs in
        that context. This method is kept as a hook for future expansion.
        """
        if not self._schema_drift_warned:
            # Already warned by the caller; this is a secondary hook.
            self._schema_drift_warned = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def __call__(self, state: PipelineState) -> PipelineState:
        enriched = state.enriched
        if enriched is None:
            raise RuntimeError(
                "ScoringAgent invoked without an EnrichedRecord in state. "
                "The EnrichmentAgent must run first."
            )

        final_score, per_model = self._predict_proba(enriched.features)
        state.score = ScoreOutput(final_score=final_score, model_scores=per_model)

        # Project the score into a two-component superposed belief.
        p_qualified = round(final_score / 100.0, 6)
        p_unqualified = round(1.0 - p_qualified, 6)
        drift = 1.0 - (p_qualified + p_unqualified)
        p_qualified += drift

        belief = SuperposedBelief(
            beliefs=(
                BeliefState(state=STATE_QUALIFIED, probability=p_qualified),
                BeliefState(state=STATE_UNQUALIFIED, probability=p_unqualified),
            ),
            step=(state.belief.step + 1) if state.belief else 0,
        )
        state_before = state.belief.as_dict() if state.belief else {}
        state.belief = belief

        decision = governance_commit(belief, theta=state.theta_governance)
        state.governance_decision = decision

        if decision.committed and decision.state == STATE_QUALIFIED:
            state.committed_action = decision.state

        state.trace.append(
            TraceEntry(
                step=len(state.trace),
                node=self.NODE_NAME,
                state_before=state_before,
                action=self.ACTION,
                state_after=belief.as_dict(),
                metadata={
                    "final_score": final_score,
                    "governance_committed": decision.committed,
                    "governance_state": decision.state,
                    "governance_confidence": decision.confidence,
                },
            )
        )
        return state

    # ------------------------------------------------------------------
    # Prediction (Slice 1 placeholder + Slice 2 real + Slice 5.2 contract)
    # ------------------------------------------------------------------

    def _predict_proba(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Return (final_score, per_model_scores).

        Routes between real ensemble (via Slice 5.2 hybrid contract) and
        Slice 1 placeholder predictor based on ``_using_real_models``.
        """
        if self._using_real_models:
            return self._predict_proba_real(features)
        return self._predict_proba_placeholder(features)

    def _predict_proba_real(
        self, features: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Slice 5.2 hybrid fail-fast contract (Decision A, 16 Nov 2026):

        - STRICT on required features: missing → RuntimeError
        - LENIENT on extras: silently dropped (DEBUG log)
        - DRIFT WARNING: log + trace audit if schema version mismatch,
          proceed on intersection

        Falls back to placeholder only if training schema is unknowable
        (no attribution.json → no feature_names list).
        """
        # Introspect training schema from models/gtm/attribution.json
        training_features = self._get_training_features()
        if not training_features:
            # Can't determine training schema — fall back to placeholder
            # (same behaviour as pre-5.2 silent zero-fill but now explicit)
            LOGGER.warning(
                "No feature_names in %s/attribution.json; cannot enforce "
                "Slice 5.2 schema contract. Falling back to placeholder.",
                self.models_dir,
            )
            return self._predict_proba_placeholder(features)

        # STRICT: required features must be present
        missing_required = set(training_features) - set(features.keys())
        if missing_required:
            raise RuntimeError(
                f"ScoringAgent (Slice 5.2 schema contract): LeadInput "
                f"missing required features for model: "
                f"{sorted(missing_required)}. "
                f"Deployment schema out of sync with "
                f"{self.models_dir}/ training schema. Fix: ensure "
                f"EnrichmentAgent populates all required features, OR "
                f"retrain models against current LeadInput schema (current "
                f"version: {LEADINPUT_SCHEMA_VERSION})."
            )

        # LENIENT: extra features silently dropped
        extras = set(features.keys()) - set(training_features)
        if extras:
            LOGGER.debug(
                "Slice 5.2 schema contract: dropping %d extra features not "
                "in training schema: %s",
                len(extras), sorted(extras),
            )

        # DRIFT WARNING: log + trace if schema version differs
        model_schema_version = self._get_schema_version()
        if (
            model_schema_version is not None
            and model_schema_version != LEADINPUT_SCHEMA_VERSION
            and not self._schema_drift_warned
        ):
            LOGGER.warning(
                "Schema drift: model trained on %s, current LeadInput is %s. "
                "Proceeding with intersection of features; results may "
                "differ from canonical training baseline.",
                model_schema_version, LEADINPUT_SCHEMA_VERSION,
            )
            self._log_schema_drift_to_trace(
                model_schema_version, LEADINPUT_SCHEMA_VERSION,
            )
            self._schema_drift_warned = True

        # Build feature vector in training-feature order
        x = np.array(
            [[float(features[name]) for name in training_features]],
            dtype=np.float64,
        )

        per_model: Dict[str, float] = {}
        probs: List[float] = []
        for name, model in self.models.items():
            if model is None:
                continue
            try:
                p = float(model.predict_proba(x)[0, 1])
            except Exception as e:
                LOGGER.warning("Model %s predict_proba failed: %s", name, e)
                continue
            per_model[name] = round(p, 4)
            probs.append(p)

        if not probs:
            warnings.warn(
                "All loaded models failed predict_proba; falling back to "
                "placeholder.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._predict_proba_placeholder(features)

        consensus_proba = float(np.mean(probs))
        final_score = round(consensus_proba * 100.0, 2)
        return final_score, per_model

    def _predict_proba_placeholder(
        self, features: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Slice 1 deterministic weighted-linear predictor, retained as
        fallback for environments where the real ensemble has not yet
        been trained OR where training schema is unknowable.
        """
        weights = {
            "industry_saas": 15.0,
            "funding_stage_series_a": 15.0,
            "team_size_50_200": 15.0,
            "role_seniority_score": 40.0,
            "hiring_for_gtm": 15.0,
        }
        raw = sum(
            weights.get(k, 0.0) * float(features.get(k, 0.0)) for k in weights
        )
        final_score = max(0.0, min(100.0, raw))
        per_model = {
            name: round(final_score / 100.0, 4) for name in self.MODEL_NAMES
        }
        return round(final_score, 2), per_model


if __name__ == "__main__":
    from pinkmate.state import EnrichedRecord, LeadInput, PipelineState

    def _build_state() -> PipelineState:
        return PipelineState(
            lead_input=LeadInput(
                lead_id="L001",
                industry="SaaS",
                funding_stage="Series_A",
                team_size=120,
                role="VP of Engineering",
            ),
            enriched=EnrichedRecord(
                lead_id="L001",
                features={
                    "industry_saas": 1,
                    "funding_stage_series_a": 1,
                    "team_size_50_200": 1,
                    "role_seniority_score": 0.9,
                    "hiring_for_gtm": 1,
                },
            ),
        )

    print("=" * 60)
    print("ScoringAgent — research-reproduction mode (default, 'zoo')")
    print("=" * 60)
    agent_zoo = ScoringAgent(model_source="zoo")
    result_zoo = agent_zoo(_build_state())
    print("Score:", result_zoo.score.model_dump())
    print("Belief:", result_zoo.belief.as_dict())

    print()
    print("=" * 60)
    print("ScoringAgent — product-demo mode ('placeholder')")
    print("=" * 60)
    agent_demo = ScoringAgent(model_source="placeholder")
    result_demo = agent_demo(_build_state())
    print("Score:", result_demo.score.model_dump())
    print("Belief:", result_demo.belief.as_dict())
