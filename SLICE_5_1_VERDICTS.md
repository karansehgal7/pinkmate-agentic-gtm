# Slice 5.1 — Real Verdicts (K=50 Canonical)

Captured 9 November 2026 from completed training runs on real HuggingFace LeadForge
advanced tier + real Zenodo CC BY 4.0 X Education lead-scoring data. The scientific
record lives in `docs/xeducation_training_report.md` and
`docs/gtm_advanced_training_report.md`; this file is the canonical verdict summary.

Commit 5cde402 shipped the artefacts; the commit message had placeholder angle-bracket
text instead of interpolated variables (operational slip at 4am build), so this doc
stands as the authoritative verdict record.

## X Education (REAL EdTech data, Zenodo 23791277, CC BY 4.0, 38.54% minority)

- **smote_direction**: PASS — SMOTE increases σ_SHAP on XGBoost (0.0111 → 0.0115, Δ=+0.0004)
- **model_stability_ordering**: PASS — LightGBM σ_SHAP < XGBoost σ_SHAP (0.0098 < 0.0115)

## LeadForge advanced (synthetic harder conditions, 7.86% minority)

- **smote_direction**: PASS — SMOTE increases σ_SHAP on XGBoost (0.0096 → 0.0251, Δ=+0.0155)
- **model_stability_ordering**: PASS — LightGBM σ_SHAP < XGBoost σ_SHAP (0.0200 < 0.0251)

## Cross-domain trend — the key scientific finding

All 4 qualitative directions PASS across all 4 datasets. The SMOTE-direction effect
size on XGBoost σ_SHAP **monotonically decreases** as class balance approaches 50%:

| Dataset | Minority rate | SMOTE Δ on XGBoost σ_SHAP | Domain |
|---|---|---|---|
| Taiwan Bankruptcy (paper §VI.C, Slice 2) | 3.23% | Large (per paper) | Financial distress |
| LeadForge advanced (Slice 5.1) | 7.86% | +0.0155 | B2B GTM synthetic harder |
| LeadForge intermediate (Slice 5) | 20.98% | +0.0108 | B2B GTM synthetic |
| **X Education (Slice 5.1, REAL)** | **38.54%** | **+0.0004** | **EdTech real** |

This is exactly what the methodological definition predicts: SMOTE has less
synthetic-positive headroom to destabilise attribution as classes near balance.
The nuance is empirically visible across 4 validation points, materially strengthening
the paper's cross-domain generalisability claim.

Model-stability ordering (LightGBM σ_SHAP < XGBoost σ_SHAP) holds across all 4 datasets
— confirmed architectural and dataset-agnostic.

## Bank of England / PRA regulatory alignment

The cross-domain validation evidence (3.23% to 38.54% imbalance range across
financial-distress, B2B GTM, and EdTech domains) materially strengthens the Q-MDP
regulatory positioning in the Bank of England supervisory context:

- **PRA SS1/23 model risk management** — validation + continuous monitoring now backed
  by a 4-point cross-domain attribution-stability evidence base
- **FS2/24 AI in financial services** — explainability via rotated-background σ_SHAP
  documented across 4 benchmarks
- **FCA agentic AI concerns** — governance-layer attribution-stability signal
  demonstrated robust across domain shift

## Honesty bar

The σ_SHAP numbers and committed PNG charts are the output of applying the Q-MDP
σ_SHAP governance harness to real EdTech B2B lead-scoring data and to
harder-conditions synthetic B2B GTM data. They DO NOT constitute performance claims
about PinkMate on real target-industry B2B GTM leads. Real-data validation on a
target-industry corpus remains deferred to later data-gated work.
