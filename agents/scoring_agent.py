"""
ScoringAgent — orchestration node V_scoring in the Q-MDP graph (§III.B).

Scores an EnrichedRecord and emits a SuperposedBelief into pipeline state
so the governance commit predicate (§III.A) has a belief to evaluate.

NOTE (Slice 1 scope):
    Per-model scores remain deterministic placeholders in Slice 1 to keep
    the LangGraph integration focused on the Q-MDP formal structure. Real
    model loading + TreeSHAP wiring + σ_SHAP attribution-stability
    measurement land in Slice 2 (see paper §III.D, §V).

    The Slice 2 change is confined to self._predict_proba() — the belief
    construction and governance interface stay stable.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pinkmate.qmdp import BeliefState, SuperposedBelief, governance_commit
from pinkmate.state import PipelineState, ScoreOutput, TraceEntry


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

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading (Slice 1 stub; replaced in Slice 2)
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """
        SLICE 1 STUB. Real pickled model loading lands in Slice 2. The
        placeholder names mirror the six-classifier zoo from paper §III.D:
            Logistic Regression, Random Forest, AdaBoost, XGBoost,
            LightGBM, CatBoost.
        """
        for name in (
            "logistic_regression",
            "random_forest",
            "adaboost",
            "xgboost",
            "lightgbm",
            "catboost",
        ):
            self.models[name] = None

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
        SLICE 1 PLACEHOLDER predictor.

        Deterministic weighted linear combination of the Slice 1 feature set.
        Produces plausible, feature-responsive scores so the Q-MDP governance
        predicate has meaningful variation to gate on during integration
        testing. Replaced by the real six-model ensemble + SMOTE + TreeSHAP
        pipeline in Slice 2 (paper §III.C–D, §V).
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

        # Flat per-model attribution for Slice 1 (all models agree with
        # the ensemble). Slice 2 replaces this with per-model proba output.
        per_model = {
            name: round(final_score / 100.0, 4)
            for name in self.models
        }

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
