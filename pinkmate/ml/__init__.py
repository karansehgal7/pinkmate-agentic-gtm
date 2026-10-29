"""
pinkmate.ml — Slice 2 ML layer.

Implements the imbalance-aware ML components of the Q-MDP paper:
    - §III.C SMOTE-based imbalance mitigation
    - §III.D Ensemble learning framework (LR / RF / AdaBoost / XGBoost /
      LightGBM / CatBoost)
    - §V TreeSHAP attribution + σ_SHAP measurement
    - §VI.A Reproducibility protocol (seed=42, 80/20 stratified split,
      SMOTE on training partition only, TreeSHAP K=50 rotated backgrounds
      on 200 reference inputs)

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Submission #4554.
"""

from pinkmate.ml.imbalance import SMOTE_K_NEIGHBORS, apply_smote
from pinkmate.ml.model_zoo import MODEL_NAMES, build_model_zoo, train_zoo, predict_proba_zoo
from pinkmate.ml.attribution import (
    sigma_shap,
    compute_rotated_shap_variance,
    AttributionResult,
)
from pinkmate.ml.dataset import (
    load_taiwan_bankruptcy,
    preprocess,
    TAIWAN_UCI_ID,
    TAIWAN_N_ROWS,
    TAIWAN_N_FEATURES,
    TAIWAN_MINORITY_RATE,
)

__all__ = [
    # imbalance
    "SMOTE_K_NEIGHBORS",
    "apply_smote",
    # model zoo
    "MODEL_NAMES",
    "build_model_zoo",
    "train_zoo",
    "predict_proba_zoo",
    # attribution
    "sigma_shap",
    "compute_rotated_shap_variance",
    "AttributionResult",
    # dataset
    "load_taiwan_bankruptcy",
    "preprocess",
    "TAIWAN_UCI_ID",
    "TAIWAN_N_ROWS",
    "TAIWAN_N_FEATURES",
    "TAIWAN_MINORITY_RATE",
]
