# Changelog

All notable changes to PinkMate (reference deployment of the Q-MDP IEEE Cyber-AI 2026 paper) are documented here, in reverse chronological order.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), adapted to the project's weekly slice cadence. Each slice corresponds to a shipped, tested increment against the paper → code map in `docs/architecture.md`.

---

## [Unreleased]

### Planned
- **Slice 7.5** — Conformal prediction overlay on `θ_governance` + PSI/KL distribution-shift sentinels. Production hardening for ongoing-monitoring regulatory conversations. Target ~2 days, ~30 new tests.
- **Paper revisions pass** (parallel to Slice 7.5) — Coordinated edit pass covering Related Work (Yan et al. 2020), Theoretical Foundations (Littman/Cassandra/Kaelbling 1995 POMDP), and Results §VI.D or §VII (cross-domain σ_SHAP transfer evidence).
- **Slice 8+** — Fairness audits (Fairlearn), Kolmogorov-Arnold Networks on lead scoring (Lee et al. 2025), DBpedia-enriched feature engineering (arXiv 2606.28355).

---

## [Housekeeping v3] — 2026-11-23 — `dc346a8`

### Fixed
- Removed duplicate `Slice 7 (planned)` bullet from §6 Roadmap (shipped bullet retained).
- Updated Section 3 tree test-count comment from `(303 total)` to `(359 total)` matching Slice 7 pytest baseline.
- Removed HTML comment fragment `<!-- APPEND this block... -->` leaked during Slice 7 integration.
- Added all 4 Slice 7 code modules, 4 Slice 7 test files, and `docs/probabilistic_framework.md` to Section 3 tree.
- Added Slice 7 status paragraph in Section 3 (was stopping at Slice 6).
- Updated `docs/architecture.md` reference from `(§1-9)` to `(§1-10)` after Slice 7 section renumber.
- Updated `models/` note from "four subdirectories" to "five" + added Bayesian trace storage note.
- Added POMDP grounding reference to ScoringAgent Agent Stack bullet.
- Updated Slice 5.2 MTA formalism reference from "deferred to Slice 7" to "delivered in Slice 7".
- Added Paper revisions pass bullet to §6 Roadmap.
- Added Slice 7.5 effort estimate.

---

## [UTF-8 Housekeeping] — 2026-11-18 — `0e3b9cd`

### Fixed
- Repaired UTF-8 double-encoding corruption across `README.md` (65 corrupted glyphs) and `docs/architecture.md` (71 corrupted glyphs) caused by PowerShell 5.1 `Get-Content -Raw` without `-Encoding UTF8` flag in Slice 5.3-7 integration scripts.
- Em-dashes, arrows (→ and ←), section symbols (§), Greek letters (σ, τ, θ, ρ), box-drawing characters (├─ └─ │), and identity symbol (≡) all restored to clean UTF-8.
- Two edge-case characters (Greek ρ and leftwards arrow ←) required manual replacement because their corruption sequences included bytes undefined in cp1252 (0x81, 0x90).

### Added
- Prevention guidance locked for all future integration guides: use `Copy-Item` from scratchpad files or `[System.IO.File]::ReadAllText` with explicit UTF-8 encoding, never `Get-Content -Raw` + `Set-Content -Encoding utf8`.

---

## [Slice 7] — 2026-11-18 — `4bce54f`

### Added
- **Probabilistic framework formalisation** (bundled per 9 Nov 2026 scope decision).
- `pinkmate/qmdp_formal.py` — POMDP tuple + Bayesian belief filter + Q-MDP commit algorithm. Cites Littman, Cassandra & Kaelbling (1995). Isomorphism adapter proving `SuperposedBelief` ≡ POMDP belief and `governance_commit` ≡ degenerate Q-MDP commit.
- `pinkmate/ml/hmm_belief.py` — 3-state `LatentStateHMM` wrapping `hmmlearn.CategoricalHMM`. Viterbi + forward-backward + Baum-Welch fit. States: MQL, SQL, ClosedWon.
- `pinkmate/ml/mixture_survival.py` — EM-fit Weibull mixture with right-censoring. BIC-based K selection over K ∈ {2, 3, 4}.
- `pinkmate/ml/bayesian_survival.py` — PyMC 5.x+ Bayesian Weibull with HalfNormal priors + NUTS sampling + posterior-predictive credible intervals on S(t).
- `docs/probabilistic_framework.md` — Standalone theoretical reference for paper Section III.A rewrite material.
- New top-level `docs/architecture.md` Section 7 (methodological jump, not sub-slice under §5 or §6).
- 56 new tests across 4 files, pytest total 303 → 359.

### Changed
- `docs/architecture.md` §7 "Slice roadmap", §8 "Running the pipeline", §9 "Citing" renumbered to §8, §9, §10.

### Dependencies
- `pymc>=5.0.0`, `hmmlearn>=0.3.0`.

### Decisions locked (17 Nov 2026)
- A: PyMC over NumPyro (maturity + Windows compatibility).
- B: NUTS sampler (standard Bayesian survival, R-hat + ESS diagnostics).
- C: BIC sweep K ∈ {2, 3, 4} (not committed to K=2, not non-parametric).
- D: hmmlearn with 3 latent states (sklearn API parity).

---

## [README Housekeeping v2] — 2026-11-18 — `a5536c3`

### Changed
- Rebuilt README.md §3 Repository Structure from scratch to match actual repo tree after Slice 5.1–6 additions.
- Moved Slice 5.1 / 5.2 / 5.3 / 6 bullets from end-of-file into §6 Roadmap (correct position).
- Added Slice 5.1 / 5.2 / 5.3 / 6 status paragraphs in §3 (status section previously stopped at Slice 5).

### Fixed
- Removed duplicate `pinkmate/ml/` directory listing from an earlier bad merge.
- Corrected Slice 5.1 roadmap bullet from "deferred" to "shipped 16 Nov 2026".

---

## [Slice 6] — 2026-11-18 — `a131ba9`

### Added
- **Survival analysis extension** — methodological jump from classification to survival models.
- `pinkmate/ml/survival.py` — 6-model survival ensemble (Cox PH primary + WeibullAFT + LogNormalAFT + LogLogisticAFT + XGBoost `survival:cox` + LightGBM AFT) under a unified `_BaseSurvivalWrapper` interface.
- `pinkmate/ml/survival_data.py` — LeadForge → survival schema transform via synthetic `days_since_first_touch` construction (Decision B=3).
- `pinkmate/ml/survival_attribution.py` — σ_SHAP on hazard outputs via Krzyziński et al. 2023 `survshap` package, with finite-difference fallback.
- `scripts/train_survival_ensemble.py` — CLI harness answering three research questions: (1) σ_SHAP transfer classification→survival; (2) Cox PH σ_SHAP vs Weibull σ_SHAP; (3) best parametric family by AIC.
- New top-level `docs/architecture.md` Section 6 (methodological jump).
- 62 new tests across 3 files, pytest total 241 → 303.

### Changed
- `docs/architecture.md` §6 "Slice roadmap", §7 "Running the pipeline", §8 "Citing" renumbered to §7, §8, §9.

### Dependencies
- `lifelines>=0.27.0`, `scikit-survival>=0.22.0`, `survshap>=0.4.0`.

### Regulatory alignment
- Cox PH is the Bank of England PRA SS1/23 convention for survival-style credit-risk modelling; shipping σ_SHAP on both semi-parametric and parametric families gives the RAEng Route 4 evidence pack a regulatory-adjacent methodology line.

---

## [Slice 5.3] — 2026-11-16 — `10949bb`

### Added
- **σ_calibration governance sibling** to σ_SHAP attribution-stability. Both measured under the SAME K=50 rotation protocol.
- `pinkmate/ml/calibration_variance.py` — API mirrors `attribution.py` (`sigma_calibration` + `compute_rotated_calibration_variance` sibling to `sigma_shap` + `compute_rotated_shap_variance`).
- `scripts/measure_sigma_calibration.py` — Reuses Slice 5/5.1 trained model pickles across 4 cross-domain datasets, correlates σ_calibration with σ_SHAP via Spearman + 95% bootstrap CI.
- `docs/architecture.md` Section 5g for σ_calibration (renames Slice 5.1 → 5e, Slice 5.2 → 5f, adds σ_calibration as 5g).
- 26 new tests, pytest total 215 → 241.

### Properties enforced
- Bounded: σ ∈ [0, 0.5].
- K-stable: σ at K=50 within 0.05 of σ at K=10 on well-behaved synthetic data.
- Reproducible: same seed → identical σ.

---

## [Slice 5.2] — 2026-11-16 — `7ed4031`

### Added
- **Deployment-schema alignment** + RFM feature engineering + Clay-style ICP scoring + MTA formalism.
- `pinkmate/ml/feature_engineering.py` — Pure RFM functions (90-day recency decay, log-scaled frequency saturating at 20 touches, monetary ACV band step function, composite) + 5-rule Clay-style ICP scoring with composite 0-100 + A/B/C/D tier.
- `pinkmate/state.py` — Exported `LEADINPUT_SCHEMA_VERSION = "slice52_rfm_icp_v1"` + 12 new optional `LeadInput` fields.
- `docs/architecture.md` Section 5f adds MTA (Multi-Touch Attribution) formalism isomorphism statement to Shao & Li 2011 data-driven MTA model.
- 22 new tests, pytest total 179 → 201.

### Changed
- `ScoringAgent._predict_proba_real()` runs a **hybrid fail-fast schema contract** (STRICT on required features via `RuntimeError`, LENIENT on extras via DEBUG log, WARN on `LEADINPUT_SCHEMA_VERSION` drift) instead of silent zero-fill.
- `EnrichmentAgent` computes RFM + Clay ICP features during enrichment step.

---

## [Slice 5.1 follow-up] — 2026-11-16 — `a3d3606`

### Added
- `SLICE_5_1_VERDICTS.md` canonical cross-domain verdicts record with Bank of England / PRA alignment language.

### Fixed
- Working-tree cleanup after Slice 5.1 ship.

---

## [Slice 5.1] — 2026-11-16 — `5cde402`

### Added
- **Cross-domain validation** on the first REAL dataset: X Education Lead Scoring v1 (Zenodo 23791277, CC BY 4.0, 9240 leads, ~38.54% conversion) + LeadForge advanced tier (synthetic B2B GTM with injected missingness and outliers, harder-conditions stress test).
- `pinkmate/ml/x_education_loader.py` — Zenodo dataset loader with authentication-free fetch.
- `scripts/train_xeducation_ensemble.py` — Trains 6-model ensemble on X Education real data.
- `docs/xeducation_training_report.md` + `docs/gtm_advanced_training_report.md` — Training reports for both datasets.
- Committed PNG charts under `docs/figures/xeducation/` and `docs/gtm_advanced/figures/gtm/`.
- 39 new tests, pytest total 140 → 179.

### Verdict
- **All 4 qualitative directions PASS** across the full 3.23% → 38.54% imbalance spectrum (Taiwan 3.23% → LeadForge advanced 7.86% → LeadForge intermediate 20.98% → X Education 38.54%).
- **Monotone SMOTE-Δ-decrease trend empirically visible:** Δ = +0.0155 → +0.0108 → +0.0004 as minority rate approaches balance, exactly as the methodological definition predicts.

---

## [Slice 5] — 2026-11-09 — `42267e5`

### Added
- **GTM-domain extension** of the Slice 2 ensemble trained on LeadForge Lead Scoring v1 (MIT, synthetic B2B SaaS procurement scenario).
- `pinkmate/ml/leadforge_loader.py` — HuggingFace download + one-hot encoding + leakage column exclusion.
- `scripts/train_gtm_ensemble.py` — Trains 6-model ensemble on LeadForge intermediate tier.
- `docs/gtm_training_report.md` + `docs/figures/gtm/*.png`.
- `pinkmate/state.py` `LeadInput` gains optional `employee_band`, `estimated_revenue_band`, `process_maturity_band` fields.
- 36 new tests, pytest total 104 → 140.

### Changed
- `agents/scoring_agent.py` default `models_dir` changes from `models/` to `models/gtm/` (product-first positioning).
- `scripts/reproduce_table2.py` writes to `models/taiwan/` (subdirectory separation).

### Honesty bar
- σ_SHAP numbers DO NOT constitute performance claims about PinkMate on real B2B GTM leads — this is methodology validation per Slice 2's Taiwan-benchmark honesty bar.

---

## [Slice 4] — 2026-11-02 — `418d6ed`

### Added
- **Empirical validation** of paper §III.B operational claims.
- `pinkmate/benchmarks/latency.py` — Measures scheduler latency vs graph size on parametric chain topology (n ∈ {5, 10, 25, 50, 100, 250}, 10 runs each).
- `pinkmate/benchmarks/replay_fidelity.py` — Reusable cross-process replay fidelity harness.
- `scripts/run_benchmarks.py` — Writes `docs/benchmarks_report.md`, `docs/benchmarks_report.json`, three PNG charts.
- 20 new tests.

### Current empirical result
- Linear fit R² = 0.98 on chain topology.
- Per-step scheduler overhead bounded across the n range (operational signal for O(log n + k) bound).
- 3/3 fidelity threads pass strict replay-fidelity.

---

## [Slice 3.5] — 2026-11-04 — `e282005`

### Added
- `ScoringAgent(model_source="zoo"|"placeholder")` constructor flag — separates research-reproduction path (loads from `models/`, warnings on mismatch) from product-demo path (skips model loading entirely, zero warnings).
- 9 new tests.

---

## [Slice 3] — 2026-10-29 — `815c6e2`

### Added
- **Durable checkpointing** per paper §III.B replay-fidelity claim.
- `pinkmate/persistence.py::make_checkpointer()` — Factory selects between `MemorySaver` (default, in-process) and `SqliteSaver` (on-disk, cross-process).
- `run_pipeline(initial, thread_id, checkpoint_backend="sqlite", ...)` — Persists every post-node `PipelineState` snapshot to disk keyed by `thread_id`.
- 12 new tests covering SQLite persistence, cross-process replay fidelity, thread isolation, and the factory contract.

---

## [Slice 2] — 2026-10-27 — `2903f49`

### Added
- **Imbalance-aware ML layer** per paper §III.C–D and §V.
- `pinkmate/ml/dataset.py` — Taiwan Bankruptcy UCI 572 loader.
- `pinkmate/ml/imbalance.py` — SMOTE wrapper (§III.C).
- `pinkmate/ml/model_zoo.py` — Six-model ensemble (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost).
- `pinkmate/ml/attribution.py` — TreeSHAP + σ_SHAP (§V) measurement harness.
- `pinkmate/ml/pipeline.py` — End-to-end training + measurement runner.
- `scripts/reproduce_table2.py` — Reproduces paper Table I and Table II against UCI 572 with tolerance-based assertions (σ_SHAP ±0.005, recall ±0.03).

### Changed
- `agents/scoring_agent.py` now runs the real six-model ensemble when `models/` is populated, falls back transparently to Slice 1 deterministic placeholder predictor with `RuntimeWarning` when models not yet trained.

---

## [Slice 1] — 2026-05-29 — `18f074`

### Added
- **Q-MDP formal structure** wired into LangGraph with deterministic-replay smoke tests.
- `pinkmate/qmdp.py` — Belief state, superposed belief, governance commit predicate (paper §III.A).
- `pinkmate/state.py` — Pydantic `PipelineState` + `τ_exec` trace (§III.A).
- `workflows/pipeline.py` — LangGraph StateGraph with governance-gated edge.
- `docs/architecture.md` — Initial paper → code map.
- Test harness.
