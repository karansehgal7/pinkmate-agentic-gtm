# PinkMate Slice 5 — GTM-Domain Training Report

- **Generated**: 2026-11-09T11:03:24Z
- **Dataset**: LeadForge Lead Scoring v1 — tier `advanced`
- **Source**: `cached`
- **Rows loaded**: 2 classes; test-partition size fixed by seed=42 split
- **Minority rate (converted_within_90_days)**: 0.0786
- **Seed**: 42
- **Dataset SHA-256**: `06894e82834b4846a41bc289b509fe9c25a55547c019b93b25808fdbf1ffdcdb`

---

## Methodology validation on synthetic B2B-shaped data

The σ_SHAP numbers below and the PNG charts under
`docs/figures/gtm/` are the output of applying the Q-MDP
σ_SHAP governance harness to B2B-shaped tabular data for
the first time. **They DO NOT constitute performance claims**
**about PinkMate on real B2B GTM leads.** Real-data validation
is deferred to Slice 5.1+ once a labelled real corpus is
available.

This honesty bar mirrors Slice 2's Taiwan Bankruptcy framing —
the paper validates the framework on a declared benchmark, it
does not claim the trained models should be used for other
domains without retraining.

---

## 1. Classification metrics (held-out test partition)

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| logistic_regression | 0.0934 | 0.6456 | 0.1632 | 0.5505 |
| random_forest | 0.0000 | 0.0000 | 0.0000 | 0.5690 |
| adaboost | 0.0588 | 0.0127 | 0.0208 | 0.5363 |
| xgboost | 0.2500 | 0.0127 | 0.0241 | 0.5364 |
| lightgbm | 0.0000 | 0.0000 | 0.0000 | 0.5347 |
| catboost | 0.0000 | 0.0000 | 0.0000 | 0.5915 |

![Classification metrics per model](figures/gtm/classification_metrics.png)

---

## 2. Attribution stability — σ_SHAP (paper §V)

| Model | Training regime | σ_SHAP | mean &#124;φ&#124; | K backgrounds | Background size | N reference |
|-------|-----------------|--------|-------------------|---------------|-----------------|-------------|
| xgboost | with SMOTE | 0.0251 | 0.1489 | 50 | 100 | 200 |
| lightgbm | with SMOTE | 0.0200 | 0.0941 | 50 | 100 | 200 |
| xgboost | **without SMOTE (baseline)** | 0.0096 | 0.0945 | 50 | 100 | 200 |

![σ_SHAP per model](figures/gtm/sigma_shap_per_model.png)

---

## 3. Qualitative-direction verdict (vs Taiwan Slice 2)

Per Phase 1 Critique 3 (close-out of Slice 5 scope planning), the
verdict on LeadForge is **qualitative-direction**, not
magnitude-tolerance. Taiwan Slice 2's two §VI.C qualitative
observations are tested for transferability:

**SMOTE direction on XGBoost (Taiwan expected: INCREASE)**
→ PASS — SMOTE increases σ_SHAP on XGBoost (0.0096 → 0.0251, Δ=+0.0155)

**Model stability ordering (Taiwan expected: LightGBM < XGBoost)**
→ PASS — LightGBM σ_SHAP < XGBoost σ_SHAP (0.0200 < 0.0251)

DRIFT verdicts on LeadForge are **expected possible outcomes**,
not failures — LeadForge's ~20% minority class is less imbalanced
than Taiwan's 3.23%, so SMOTE's effect on σ_SHAP has less
headroom to move in the same direction with the same magnitude.

---

## 4. Reproducing

```bash
python scripts/train_gtm_ensemble.py             # Full K=50 run (~10 min)
python scripts/train_gtm_ensemble.py --fast      # K=10 quick check (~2 min)
python scripts/train_gtm_ensemble.py --synthetic # Hermetic synthetic run
python scripts/train_gtm_ensemble.py --tier advanced  # Slice 5.1 preview
```

Trained pickles land under `models/gtm/`. The ScoringAgent default
(`model_source="zoo"`, `models_dir="models/gtm"`) picks them up
automatically on next construction.
