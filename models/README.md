# `models/` — Trained model artefacts

This directory is **gitignored by design** — trained pickles are large (5-20 MB per model), and the reference deployment is the paper-reproduction scripts, not the specific pickle bytes.

## First-run on a fresh clone

A fresh clone has an empty (or absent) `models/` directory. The `ScoringAgent` will fall back to the Slice 1 deterministic placeholder predictor with a `RuntimeWarning` until you populate at least one subdirectory below.

## Subdirectory layout + population scripts

| Subdirectory | Shipped in | Populated by | Contains |
|---|---|---|---|
| `models/taiwan/` | Slice 2 | `python scripts/reproduce_table2.py` | 6 classification models (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost) trained on Taiwan Bankruptcy UCI 572, plus `attribution.json` with σ_SHAP measurements. Used for paper Table I + Table II reproduction. |
| `models/gtm/` | Slice 5 | `python scripts/train_gtm_ensemble.py` | Same 6-model ensemble trained on LeadForge Lead Scoring v1 intermediate tier (MIT-licensed synthetic B2B SaaS data from HuggingFace). Default `models_dir` for `ScoringAgent` post-Slice-5. |
| `models/gtm_advanced/` | Slice 5.1 | `python scripts/train_gtm_ensemble.py --tier advanced` | Same 6-model ensemble trained on LeadForge advanced tier (synthetic B2B GTM with injected missingness and outliers — harder-conditions stress test). |
| `models/xeducation/` | Slice 5.1 | `python scripts/train_xeducation_ensemble.py` | Same 6-model ensemble trained on X Education Lead Scoring v1 (Zenodo 23791277, CC BY 4.0, 9240 real EdTech leads). The first REAL dataset in the build programme. |
| `models/survival/` | Slice 6 | `python scripts/train_survival_ensemble.py` | 6-model survival ensemble (Cox PH + WeibullAFT + LogNormalAFT + LogLogisticAFT + XGBoost `survival:cox` + LightGBM AFT) trained on LeadForge via Decision B=3 synthetic time construction. |
| `models/bayesian/` | Slice 7 | Run `pinkmate.ml.bayesian_survival.fit_bayesian_weibull` directly, or custom script | PyMC NUTS traces (`arviz.InferenceData` objects, ~100 MB each). No dedicated CLI script yet; invoke the module function from a notebook or your own harness. Not required for pytest or ScoringAgent operation — purely for Bayesian posterior-predictive credible interval computation on Weibull survival parameters. |

## Fast-path: minimum setup for paper reproduction

To reproduce the paper's Table I and Table II only (Taiwan Bankruptcy, no GTM data needed):

```bash
# Fresh clone, after pip install -r requirements.txt
python scripts/reproduce_table2.py
```

This writes 6 pickles to `models/taiwan/` plus `models/taiwan/attribution.json` with the σ_SHAP measurements. Expected runtime: ~3-5 minutes on commodity CPU (no GPU required).

## Fast-path: Slice 5 GTM demo

For the product-arc demo path (LeadForge synthetic B2B GTM):

```bash
python scripts/train_gtm_ensemble.py
```

Writes to `models/gtm/` + `docs/gtm_training_report.md` + `docs/figures/gtm/*.png`. Expected runtime: ~5-8 minutes.

## Fast-path: full cross-domain evidence base (Slice 5.1)

```bash
python scripts/reproduce_table2.py                         # Taiwan baseline
python scripts/train_gtm_ensemble.py                       # LeadForge intermediate
python scripts/train_gtm_ensemble.py --tier advanced       # LeadForge advanced
python scripts/train_xeducation_ensemble.py                # X Education real
python scripts/measure_sigma_calibration.py                # σ_calibration across all 4
python scripts/train_survival_ensemble.py                  # Survival ensemble
```

Total runtime: ~30-45 minutes. Produces the full evidence base behind the SLICE_5_1_VERDICTS.md canonical record.

## Storage notes

- Each classification model pickle: ~5-15 MB (CatBoost is the largest at ~15 MB).
- Each survival model pickle: ~3-10 MB (lifelines models are smaller; XGBoost/LightGBM larger).
- PyMC NUTS trace: ~100 MB per dataset for 1000 draws × 4 chains × 2 parameters.
- Expected total `models/` footprint across all 5 subdirectories at full coverage: **300-500 MB**.

## Why gitignored?

1. **Size** — committing trained pickles bloats the repo and slows clones.
2. **Reproducibility** — the scripts are the canonical "how to reproduce" artefact; pickles are the side-effect output.
3. **Non-determinism in some paths** — PyMC NUTS traces depend on RNG state which can drift across OS/BLAS versions; committing them would create false diffs on retraining.

If you need committed artefacts for external review (e.g. paper reviewers who cannot run the scripts), run the scripts on your machine and attach the pickles as supplementary material separately, rather than committing to the public repo.
