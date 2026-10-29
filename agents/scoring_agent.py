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
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pinkmate.qmdp import BeliefState, SuperposedBelief, governance_commit
from pinkmate.state import PipelineState, ScoreOutput, TraceEntry

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

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.sigma_shap_by_model: Dict[str, float] = {}
        self.feature_order: Optional[List[str]] = None
        self._using_real_models: bool = False
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading — Slice 2
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """
        Load trained models from ``self.models_dir`` if present. If the
        directory is missing or empty, fall back to the Slice 1 deterministic
        placeholder with a visible WARNING.

        Design decision (Slice 2, 2026-10-27): fallback-with-warning so
        the pipeline still runs for demo / smoke-test purposes before the
        reproduce script has been run.
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
                "Run `python scripts/reproduce_table2.py` to train real models "
                "and populate `models/`.",
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
        # on committed beliefs when attribution is unstable.
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
        # Guard rounding: ensure sum == 1 exactly within SuperposedBelief tolerance.
        drift = 1.0 - (p_qualified + p_unqualified)
        p_qualified += drift  # absorb floating remainder into the leading term

        belief = SuperposedBelief(
            beliefs=(
                BeliefState(state=STATE_QUALIFIED, probability=p_qualified),
                BeliefState(state=STATE_UNQUALIFIED, probability=p_unqualified),
            ),
            step=(state.belief.step + 1) if state.belief else 0,
        )
        state_before = state.belief.as_dict() if state.belief else {}
        state.belief = belief

        # Evaluate the governance commit predicate here (not in the routing
        # function) so the resulting GovernanceCommit persists into state.
        # Routers in LangGraph are pure routing functions; node bodies are
        # where state mutations are durable.
        decision = governance_commit(belief, theta=state.theta_governance)
        state.governance_decision = decision

        # Domain semantics: commit onto the downstream outreach path only
        # when the committed state is a progression state (QUALIFIED). A
        # high-confidence UNQUALIFIED commit resolves the governance
        # predicate but is semantically a terminal reject, so execution
        # defers rather than progressing to segmentation / outreach.
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
    # Prediction (Slice 1 placeholder — real ensemble in Slice 2)
    # ------------------------------------------------------------------

    def _predict_proba(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Return (final_score, per_model_scores).

        Slice 2 path (real ensemble present):
            Each loaded model's predict_proba is averaged across the six to
            form the ensemble consensus. The feature vector is pulled from
            ``features`` in a stable order and padded / zero-filled where
            an expected feature is missing.

        Slice 2 fallback path (models missing):
            The Slice 1 deterministic weighted-linear predictor is used.
            A RuntimeWarning was already emitted at construction time.
        """
        if self._using_real_models:
            return self._predict_proba_real(features)
        return self._predict_proba_placeholder(features)

    def _predict_proba_real(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Run the loaded six-model ensemble on the engineered feature vector."""
        # Pull features in a stable order matching training-time columns.
        feature_names = self.feature_order or sorted(features.keys())
        x = np.array(
            [[float(features.get(name, 0.0)) for name in feature_names]],
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
            # Last-ditch: all models failed. Fall back to placeholder.
            warnings.warn(
                "All loaded models failed predict_proba; falling back to placeholder.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._predict_proba_placeholder(features)

        # Ensemble consensus = simple average of per-model probabilities.
        consensus_proba = float(np.mean(probs))
        final_score = round(consensus_proba * 100.0, 2)
        return final_score, per_model

    def _predict_proba_placeholder(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Slice 1 deterministic weighted-linear predictor, retained as fallback
        for environments where the real ensemble has not yet been trained.
        """
        weights = {
            "industry_saas": 15.0,
            "funding_stage_series_a": 15.0,
            "team_size_50_200": 15.0,
            "role_seniority_score": 40.0,
            "hiring_for_gtm": 15.0,
        }
        raw = sum(weights.get(k, 0.0) * float(features.get(k, 0.0)) for k in weights)
        final_score = max(0.0, min(100.0, raw))
        per_model = {name: round(final_score / 100.0, 4) for name in self.MODEL_NAMES}
        return round(final_score, 2), per_model


if __name__ == "__main__":
    # Preserved runnable example (updated for Slice 1 state object).
    from pinkmate.state import EnrichedRecord, LeadInput, PipelineState

    agent = ScoringAgent()
    state = PipelineState(
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
    result = agent(state)
    print("Score:", result.score.model_dump())
    print("Belief:", result.belief.as_dict())
