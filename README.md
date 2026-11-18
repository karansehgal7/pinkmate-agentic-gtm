# PinkMate — Agentic GTM Engine (Technical Architecture Repository)

PinkMate is an early-stage research and engineering project exploring deterministic, multi-agent orchestration for B2B go-to-market workflows. It combines classical interpretable machine learning with structured agent-level routing to investigate how lead-scoring, segmentation, and outreach decisions can be made more reproducible, auditable, and transparent than conventional CRM automation or LLM-only agent loops.

> **Repository scope.** This repository documents the technical architecture, agent stack, and reference skeletons for PinkMate's predictive GTM engine. It is a **research-engineering artefact** — the agent skeletons and orchestration scaffolding are public so the design can be inspected and built upon. Trained model artefacts, production data pipelines, and end-to-end runnable workflows are **work in progress** and not present in this repository in their current form.
>
> **License:** Proprietary — source available for evaluation only. See `LICENSE` file for details.

---

## 1. Technical Overview

**Core thesis.** Conventional CRM automation stores leads and applies static rules. LLM-only agent systems generate fluent action sequences but exhibit non-deterministic execution, weak auditability, and unstable explainability. PinkMate explores a middle path: **classical interpretable ML for scoring, bounded modular agents for orchestration, and explicit state transitions for reproducibility.**

**Design choices.**

- **Multi-agent pipeline:** Ingestion → Enrichment → Predictive Scoring → Segmentation → Outreach → Feedback loop, with each agent responsible for a bounded function and observable state transition.
- **Predictive ML (planned implementation):** classical interpretable models — Naive Bayes, Random Forest, Gradient Boosting — chosen deliberately over opaque deep-learning alternatives to support feature attribution and governance review.
- **Agentic orchestration:** LangChain / LangGraph-style routing between agents, with explicit state schemas and event-triggered transitions.
- **Memory and context (planned):** vector-based memory for company, persona, and interaction history, enabling retrieval-augmented context for downstream agents.
- **Integration surface (planned):** designed to interoperate with Clay, Apollo, PhantomBuster, HubSpot, and Salesforce.

---

## 2. Engineering Positioning

PinkMate is the **reference deployment of the Q-MDP orchestration framework** named in the paper's final section. The architectural choices here — bounded agent responsibilities, superposition-aware belief state, governance-gated conditional transitions, interpretable classical ML for scoring, and explainability-oriented design — directly instantiate the formal constructs of:

> Sehgal, K., & Bhatti, K. N. (2026). *Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure.* **Accepted at IEEE Cyber-AI 2026**, Bucharest, Romania. Submission #4554.

A full paper-to-code map is in [`docs/architecture.md`](./docs/architecture.md). The repository is now in active build against that map on a weekly slice cadence.

---

## 3. Repository Structure

```
pinkmate-agentic-gtm/
│
├─ README.md                          # This file – high-level overview
├─ LICENSE                            # Proprietary-source-available
├─ CODE_OF_CONDUCT.md
├─ SECURITY.md
├─ SLICE_5_1_VERDICTS.md              # Canonical cross-domain verdicts (Slice 5.1)
├─ requirements.txt                   # Python dependencies
│
├─ pinkmate/                          # Core package
│   ├─ __init__.py
│   ├─ qmdp.py                        # Belief state, superposed belief, governance
│   │                                 # commit predicate (paper §III.A) — Slice 1
│   ├─ state.py                       # Pydantic PipelineState + τ_exec trace (§III.A)
│   │                                 # + LEADINPUT_SCHEMA_VERSION constant (Slice 5.2)
│   ├─ persistence.py                 # Checkpointer factory — MemorySaver | SqliteSaver
│   │                                 # (paper §III.B replay fidelity) — Slice 3
│   ├─ benchmarks/                    # Empirical validation harnesses — Slice 4
│   │   ├─ __init__.py
│   │   ├─ latency.py                 # T_fa = O(log n + k) scheduling-bound harness
│   │   └─ replay_fidelity.py         # Reusable cross-process replay fidelity harness
│   └─ ml/                            # ML layer across the Slice arc
│       ├─ __init__.py
│       ├─ dataset.py                 # Taiwan Bankruptcy UCI 572 loader — Slice 2
│       ├─ imbalance.py               # SMOTE wrapper (§III.C) — Slice 2
│       ├─ model_zoo.py               # Six-model ensemble (§III.D) — Slice 2
│       ├─ attribution.py             # TreeSHAP + σ_SHAP (§V) — Slice 2
│       ├─ pipeline.py                # Taiwan training + measurement runner — Slice 2
│       ├─ leadforge_loader.py        # LeadForge Lead Scoring v1 loader — Slice 5
│       ├─ x_education_loader.py      # X Education Zenodo 23791277 loader — Slice 5.1
│       ├─ feature_engineering.py     # RFM + Clay-style ICP pure functions — Slice 5.2
│       ├─ calibration_variance.py    # σ_calibration sibling to σ_SHAP — Slice 5.3
│       ├─ survival.py                # 6-model survival ensemble — Slice 6
│       ├─ survival_data.py           # LeadForge → survival schema transform — Slice 6
│       └─ survival_attribution.py    # SurvSHAP + σ_SHAP_hazard — Slice 6
│
├─ agents/                            # Orchestration-graph vertices (paper §III.B)
│   ├─ __init__.py
│   ├─ ingestion_agent.py
│   ├─ enrichment_agent.py            # RFM + ICP computation step (Slice 5.2)
│   ├─ scoring_agent.py               # Hybrid fail-fast schema contract (Slice 5.2)
│   │                                 # + deployment-mode flag (Slice 3.5)
│   ├─ segmentation_agent.py
│   └─ outreach_agent.py
│
├─ workflows/                         # Orchestrated workflows and routing
│   ├─ pipeline.py                    # LangGraph StateGraph with governance-gated edge
│   │                                 # (Slice 1 + Slice 3 durable checkpointing)
│   └─ pipeline.yaml                  # DEPRECATED; retained for commit-history continuity
│
├─ scripts/                           # Standalone reproduction + training scripts
│   ├─ __init__.py
│   ├─ reproduce_table2.py            # Reproduces paper Table I + Table II from UCI 572,
│   │                                 # writes to models/taiwan/ — Slice 2
│   ├─ run_benchmarks.py              # Latency + replay-fidelity harnesses,
│   │                                 # writes docs/benchmarks_report.md — Slice 4
│   ├─ train_gtm_ensemble.py          # Trains 6-model ensemble on LeadForge intermediate,
│   │                                 # writes models/gtm/ + report + figures — Slice 5
│   ├─ train_xeducation_ensemble.py   # Trains 6-model ensemble on X Education real data,
│   │                                 # writes models/xeducation/ + report — Slice 5.1
│   ├─ measure_sigma_calibration.py   # Reuses Slice 5/5.1 pickles, correlates σ_calibration
│   │                                 # with σ_SHAP across 4 datasets — Slice 5.3
│   └─ train_survival_ensemble.py     # Trains 6 survival models + computes σ_SHAP_hazard
│                                     # + correlates with classification σ_SHAP — Slice 6
│
├─ tests/                             # Reproducibility + determinism tests (303 total)
│   ├─ __init__.py
│   ├─ test_qmdp.py                   # Slice 1 — Q-MDP primitives
│   ├─ test_pipeline.py               # Slice 1 — LangGraph integration + replay
│   ├─ test_imbalance.py              # Slice 2 — SMOTE
│   ├─ test_attribution.py            # Slice 2 — σ_SHAP
│   ├─ test_model_zoo.py              # Slice 2 — six-model ensemble
│   ├─ test_dataset.py                # Slice 2 — Taiwan loader + preprocessing
│   ├─ test_slice2_integration.py    # Slice 2 — full pipeline + §VI.C findings
│   ├─ test_checkpointing.py         # Slice 3 — SqliteSaver + cross-process replay
│   ├─ test_benchmarks.py             # Slice 4 — latency + replay-fidelity harness
│   ├─ test_scoring_agent_modes.py   # Slice 3.5 — ScoringAgent deployment-mode flag
│   ├─ test_leadforge_loader.py      # Slice 5 — LeadForge loader + synthetic fixtures
│   ├─ test_gtm_training.py           # Slice 5 — GTM training qualitative-direction tests
│   ├─ test_x_education_loader.py    # Slice 5.1 — X Education loader + Zenodo fetch
│   ├─ test_xeducation_training.py    # Slice 5.1 — X Education training pipeline
│   ├─ test_feature_engineering.py   # Slice 5.2 — RFM + Clay ICP pure functions
│   ├─ test_scoring_agent_schema.py  # Slice 5.2 — hybrid fail-fast schema contract
│   ├─ test_calibration_variance.py  # Slice 5.3 — σ_calibration 3 properties
│   ├─ test_survival.py               # Slice 6 — 6 survival models correctness
│   ├─ test_survival_data.py          # Slice 6 — LeadForge → survival transform
│   └─ test_survival_attribution.py  # Slice 6 — SurvSHAP + σ_SHAP transfer
│
└─ docs/                              # Technical documentation
    ├─ architecture.md                # Paper ↔ code map, slice roadmap (§1-9)
    ├─ benchmarks_report.md           # Slice 4 empirical validation report
    ├─ benchmarks_report.json         # Machine-readable benchmark payload
    ├─ gtm_training_report.md         # Slice 5 GTM training report
    ├─ xeducation_training_report.md  # Slice 5.1 X Education training report
    ├─ gtm_advanced/                  # Slice 5.1 LeadForge advanced tier artefacts
    │   ├─ gtm_training_report.md
    │   └─ figures/gtm/
    ├─ sigma_calibration_report.md    # Slice 5.3 σ_calibration report (generated)
    ├─ survival_training_report.md    # Slice 6 survival training report (generated)
    └─ figures/                       # Committed PNG charts for in-browser review
        ├─ latency_total_vs_n.png
        ├─ latency_per_step_overhead.png
        ├─ replay_fidelity_summary.png
        ├─ gtm/                       # Slice 5 figures
        │   ├─ classification_metrics.png
        │   └─ sigma_shap_per_model.png
        ├─ xeducation/                # Slice 5.1 figures
        │   ├─ classification_metrics.png
        │   └─ sigma_shap_per_model.png
        ├─ sigma_calibration/         # Slice 5.3 figures (generated)
        └─ survival/                  # Slice 6 figures (generated)
            ├─ aic_comparison.png
            ├─ sigma_shap_per_survival_model.png
            └─ sigma_shap_transfer_scatter.png
```

**Note on `models/` subdirectory structure.** `models/` has four gitignored sibling subdirectories after Slice 6: `models/taiwan/` (Slice 2 paper reproduction), `models/gtm/` (Slice 5 LeadForge intermediate), `models/gtm_advanced/` + `models/xeducation/` (Slice 5.1 cross-domain validation), and `models/survival/` (Slice 6 survival ensemble). Run `python scripts/reproduce_table2.py` to populate `models/taiwan/`; run `python scripts/train_gtm_ensemble.py` for `models/gtm/`; run `python scripts/train_xeducation_ensemble.py` for `models/xeducation/`; run `python scripts/train_survival_ensemble.py` for `models/survival/`. The `ScoringAgent` default `models_dir` is `models/gtm/` post-Slice-5 (product-first positioning); Taiwan reproduction uses explicit `ScoringAgent(models_dir="models/taiwan")`.

**Slice 1 status (shipped).** The Q-MDP formal layer (`pinkmate/qmdp.py`, `pinkmate/state.py`) and the LangGraph orchestration graph (`workflows/pipeline.py`) are live and executable end-to-end on a mock lead with `python -m workflows.pipeline`.

**Slice 2 status (shipped).** The imbalance-aware ML layer is live: six-model ensemble, SMOTE, TreeSHAP σ_SHAP measurement harness, and a reproduction script (`scripts/reproduce_table2.py`) that reproduces the paper's Table I and Table II against the real UCI 572 dataset with tolerance-based assertions (σ_SHAP ±0.005, recall ±0.03). The ScoringAgent now runs the real six-model ensemble when `models/` is populated, and falls back transparently to the Slice 1 deterministic placeholder predictor (with a `RuntimeWarning`) when models are not yet trained.

**Slice 3 status (shipped).** Durable checkpointing is live: a `make_checkpointer()` factory in `pinkmate/persistence.py` selects between `MemorySaver` (default, in-process) and `SqliteSaver` (on-disk, cross-process). `run_pipeline(initial, thread_id, checkpoint_backend="sqlite", ...)` persists every post-node `PipelineState` snapshot to disk keyed by `thread_id`, operationalising the paper's §III.B replay-fidelity claim. Twelve new tests in `tests/test_checkpointing.py` verify SQLite persistence, cross-process replay fidelity, thread isolation, and the factory contract.

**Slice 4 status (shipped).** Empirical validation of the paper's §III.B operational claims is live: `pinkmate/benchmarks/latency.py` measures total pipeline latency across n ∈ {5, 10, 25, 50, 100, 250}; `pinkmate/benchmarks/replay_fidelity.py` generalises Slice 3's cross-process replay proof into a **reusable consumer harness**; `scripts/run_benchmarks.py` writes `docs/benchmarks_report.md`, `docs/benchmarks_report.json`, and three PNG charts under `docs/figures/`. 20 new tests in `tests/test_benchmarks.py`. Current empirical result: linear fit R² = 0.98 on chain topology, per-step scheduler overhead bounded across the n range, 3/3 fidelity threads pass strict replay-fidelity.

**Slice 3.5 status (shipped).** `ScoringAgent(model_source="zoo"|"placeholder")` constructor flag separates the research-reproduction path (loads from `models/`, warnings on mismatch) from the product-demo path (skips model loading entirely). 9 new tests in `tests/test_scoring_agent_modes.py`.

**Slice 5 status (shipped — methodology validation on synthetic B2B-shaped data).** GTM-domain extension trained on LeadForge Lead Scoring v1 (MIT, synthetic B2B SaaS). `scripts/train_gtm_ensemble.py` writes `models/gtm/`, `docs/gtm_training_report.md`, `docs/figures/gtm/*.png`. 36 new tests. **σ_SHAP numbers DO NOT constitute performance claims on real B2B GTM leads** — methodology validation per Slice 2's Taiwan-benchmark honesty bar.

**Slice 5.1 status (shipped 16 Nov 2026).** Cross-domain validation on the first REAL dataset in the build programme: X Education Lead Scoring v1 (Zenodo 23791277, CC BY 4.0, 9240 leads, ~38.54% conversion) + LeadForge advanced tier (synthetic B2B GTM with injected missingness and outliers, harder-conditions stress test). Commits `5cde402` + `a3d3606`. **All 4 qualitative directions PASS** across the full 3.23% → 38.54% imbalance spectrum (Taiwan 3.23% → LeadForge advanced 7.86% → LeadForge intermediate 20.98% → X Education 38.54%). **Monotone SMOTE-Δ-decrease trend empirically visible:** Δ = +0.0155 → +0.0108 → +0.0004 as minority rate approaches balance, exactly as the methodological definition predicts. Canonical verdicts record: [`SLICE_5_1_VERDICTS.md`](./SLICE_5_1_VERDICTS.md).

**Slice 5.2 status (shipped 16 Nov 2026).** Deployment-schema alignment + RFM + Clay-style ICP + MTA formalism. Commit `7ed4031`. `ScoringAgent._predict_proba_real()` now runs a **hybrid fail-fast schema contract** (STRICT on required features via `RuntimeError`, LENIENT on extras via DEBUG log, WARN on `LEADINPUT_SCHEMA_VERSION` drift) instead of silent zero-fill. `pinkmate/ml/feature_engineering.py` adds pure RFM functions (90-day recency decay + log-scaled frequency saturating at 20 touches + monetary ACV band step function + composite) and Clay-style ICP scoring (5 rules + composite 0-100 + A/B/C/D tier). `pinkmate/state.py` exports `LEADINPUT_SCHEMA_VERSION = "slice52_rfm_icp_v1"` + 12 new optional `LeadInput` fields. `docs/architecture.md` Section 5f adds an explicit MTA (Multi-Touch Attribution) formalism showing Q-MDP belief-state transitions implement a Markov chain isomorphic to the Shao & Li 2011 data-driven MTA model (full formal proof deferred to Slice 7). 201 passing tests at shipping.

**Slice 5.3 status (shipped 16 Nov 2026).** σ_calibration = calibration-stability companion to σ_SHAP attribution-stability. Commit `10949bb`. Both measured under the SAME K-rotation protocol (K=50 default, matching paper §V canonical). `pinkmate/ml/calibration_variance.py` mirrors `attribution.py` API (`sigma_calibration` + `compute_rotated_calibration_variance` sibling to `sigma_shap` + `compute_rotated_shap_variance`). `scripts/measure_sigma_calibration.py` reuses Slice 5/5.1 trained model pickles across all 4 cross-domain datasets, correlates σ_calibration with σ_SHAP via Spearman + 95% bootstrap CI, writes `docs/sigma_calibration_report.md`. Three properties enforced by `tests/test_calibration_variance.py`: (1) bounded σ ∈ [0, 0.5]; (2) K-stable (σ at K=50 within 0.05 of σ at K=10); (3) reproducible (same seed → identical σ). 241 passing tests at shipping.

**Slice 6 status (shipped 18 Nov 2026).** Survival analysis extension — Cox PH primary + 3 parametric AFT families (Weibull / LogN / LogL) + XGBoost `survival:cox` + LightGBM AFT. Commit `a131ba9`. Introduces a new top-level `## 6. Survival analysis extension` section in `docs/architecture.md` (methodological jump: different prediction target, different loss, different attribution structure; existing §6/§7/§8 renumbered to §7/§8/§9). `pinkmate/ml/survival.py` exposes a unified `_BaseSurvivalWrapper` interface over lifelines + XGBoost + LightGBM; `pinkmate/ml/survival_data.py` transforms LeadForge columns into `(time, event)` via Decision B=3 synthetic `days_since_first_touch` construction with honesty-bar sensitivity sweep over touch-interval ∈ {5, 7, 14}; `pinkmate/ml/survival_attribution.py` provides `sigma_shap_hazard` + `compute_rotated_survshap_variance` mirroring `attribution.py` API, backed by the Krzyziński et al. 2023 `survshap` package with a finite-difference cumulative-hazard fallback. `scripts/train_survival_ensemble.py` answers three research questions in `docs/survival_training_report.md`: (1) does σ_SHAP transfer from classification to survival (Spearman ρ + 95% bootstrap CI); (2) is Cox PH σ_SHAP lower than Weibull σ_SHAP; (3) which parametric family fits LeadForge best by AIC. **Bank of England PRA SS1/23 credit-risk alignment:** Cox PH is the regulatory convention for survival-style credit-risk modelling; shipping σ_SHAP on both semi-parametric and parametric families gives the RAEng Route 4 evidence pack a regulatory-adjacent methodology line. 62 new tests (303 total). Dependencies added: `lifelines>=0.27.0`, `scikit-survival>=0.22.0`, `survshap>=0.4.0`.

---

## 4. Agent Stack

PinkMate's GTM engine is modelled as a set of independent but coordinated agents:

- **IngestionAgent**
  - Accepts CSV uploads, LinkedIn exports, Apollo data, or direct API feeds.
  - Normalises and validates lead records.

- **EnrichmentAgent**
  - Calls enrichment sources (Clay, Clearbit, Apollo, internal APIs).
  - Adds firmographics (industry, size, location, funding) and buyer signals.
  - Post-Slice-5.2: computes RFM scores + Clay-style ICP rule outputs + composite ICP tier.

- **ScoringAgent**
  - Applies the six-model classical ML ensemble (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost) via a hybrid fail-fast schema contract (Slice 5.2).
  - Produces a probability of conversion with per-model breakdown.

- **SegmentationAgent**
  - Converts scores into tiers (Tier 1 / Tier 2 / Nurture).
  - Applies ICP rules (sector, geography, ARR, role seniority, intent).

- **OutreachAgent**
  - Decides next-best-action: email sequence, LinkedIn DM, nurture cadence.
  - Designed to plug into CRMs and outreach tools.

- **FeedbackLoop / Analytics**
  - Will consume engagement + conversion signals.
  - Will adjust thresholds, weights, and features over time.

Orchestration is via a LangChain / LangGraph-style workflow enabling routing, error handling, and retries.

---

## 5. ScoringAgent — Reference Skeleton

The `ScoringAgent` class in `agents/scoring_agent.py` defines the interface contract for lead scoring: a feature dictionary in, a structured score-plus-breakdown response out.

Core responsibilities:

- Load trained classical ML models from `models/gtm/` (default) or an explicit `models_dir`.
- Accept a dictionary of engineered features (now including RFM + Clay ICP features from Slice 5.2).
- Apply the Slice 5.2 hybrid fail-fast schema contract: STRICT on required training features, LENIENT on extras, WARN on `LEADINPUT_SCHEMA_VERSION` drift.
- Combine model outputs into a final score with per-model breakdown.
- Return a structured response with `final_score` (0–100) and per-model `model_scores`.

Slice 3.5 added the `model_source="zoo"|"placeholder"` constructor flag so the ScoringAgent can be instantiated for product-demo contexts (website, pitch decks) without loading real models or emitting fallback warnings.

---

## 6. Roadmap — Slice Cadence

PinkMate is being built against the paper on a weekly slice cadence. Each slice closes a specific gap between the paper's formal claims and the reference deployment.

- **Slice 1 (shipped).** Q-MDP formal structure wired into LangGraph with deterministic-replay smoke tests. `pinkmate/qmdp.py`, `pinkmate/state.py`, `workflows/pipeline.py`, test harness, paper-to-code map in `docs/architecture.md`.
- **Slice 2 (shipped).** Imbalance-aware ML layer per paper §III.C–D and §V — six-model ensemble (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost), SMOTE on training partition only, TreeSHAP attribution with σ_SHAP measurement harness. `scripts/reproduce_table2.py` reproduces Table I and Table II against UCI 572 with tolerance-based assertions.
- **Slice 3 (shipped).** Durable checkpointing per paper §III.B replay-fidelity claim — `pinkmate/persistence.py::make_checkpointer()` factory, `MemorySaver` → `SqliteSaver`, 12 new tests covering SQLite persistence and cross-process replay fidelity.
- **Slice 4 (shipped).** Empirical validation of paper §III.B operational claims — latency + replay-fidelity harnesses write `docs/benchmarks_report.md` + figures. 20 new tests.
- **Slice 3.5 (shipped).** ScoringAgent deployment-mode flag — product-arc ergonomics. `model_source="zoo"|"placeholder"` constructor flag. 9 new tests.
- **Slice 5 (shipped — methodology validation on synthetic B2B-shaped data).** GTM-domain extension trained on LeadForge Lead Scoring v1 (MIT, synthetic). 36 new tests. σ_SHAP numbers reported as methodology output, not performance claims.
- **Slice 5.1 (shipped 16 Nov 2026).** Cross-domain validation on REAL EdTech data (X Education via Zenodo 23791277 CC BY 4.0) + LeadForge advanced tier (stress test). All 4 qualitative directions PASS across Taiwan 3.23% → LeadForge advanced 7.86% → LeadForge intermediate 20.98% → X Education 38.54%. Monotone SMOTE-Δ-decrease trend empirically visible (+0.0155 → +0.0108 → +0.0004). Commits `5cde402` + `a3d3606`.
- **Slice 5.2 (shipped 16 Nov 2026).** Deployment-schema alignment (hybrid fail-fast ScoringAgent contract) + RFM feature engineering + Clay-style ICP scoring (5 rules + composite + A/B/C/D tier) + MTA formalism in `docs/architecture.md` Section 5f. Commit `7ed4031`. 201 passing tests.
- **Slice 5.3 (shipped 16 Nov 2026).** σ_calibration governance sibling to σ_SHAP (same K=50 rotation protocol). `pinkmate/ml/calibration_variance.py` mirrors `attribution.py` API. Three properties enforced: bounded, K-stable, reproducible. Research question "does σ_calibration correlate with σ_SHAP" reported via Spearman + 95% bootstrap CI in `docs/sigma_calibration_report.md`. Commit `10949bb`. 241 passing tests.
- **Slice 6 (shipped 18 Nov 2026).** Survival analysis extension — Cox PH primary + 3 parametric AFT + XGBoost `survival:cox` + LightGBM AFT + SurvSHAP. New top-level `docs/architecture.md` Section 6 (methodological jump, not sub-slice under §5). Research question "does σ_SHAP transfer from classification to survival" answered via Spearman correlation on matched model-family pairs. Bank of England PRA SS1/23 credit-risk alignment. Commit `a131ba9`. 303 passing tests.
- **Slice 7 (planned).** POMDP formalization + Bayesian + mixture + HMM — probabilistic framework formalisation bundling the theoretical scaffolding referenced from Slice 5.2 MTA section and Slice 6 survival section. Likely a fresh top-level `docs/architecture.md` Section (renumber pattern as Slice 6).
- **Slice 7.5 (planned).** Conformal prediction overlay on θ_governance + PSI/KL distribution-shift sentinels. Production-hardens the ScoringAgent for ongoing-monitoring regulatory conversations.
- **Slice 8+ (optional extensions).** Fairness audits (Fairlearn), Kolmogorov-Arnold Networks comparison on lead scoring (Lee et al. 2025), DBpedia-enriched feature engineering (arXiv 2606.28355).
- **Later (data-gated).** Real B2B conversion data path — `pink-mate.co.uk` landing-page capture over 6-12 months, academic partnership via Kent Business School / Khawar network, or specific industry engagement.
- **Later.** Integration surface — connectors to Clay, PhantomBuster, Apollo, HubSpot, and Salesforce.

Numbers will be reported only once measured against the slices that produce them.

---

## 7. Related Research

The architectural and research thesis underlying PinkMate is developed in academic collaboration with Kent Business School, University of Kent. Peer-reviewed work:

- Sehgal, K., & Bhatti, K. N. (2026). *Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure.* **Accepted** at IEEE Cyber-AI 2026, Bucharest, Romania. Submission #4554 (accepted 18 June 2026).
- Sehgal, K., & Bhatti, K. N. (2026). *Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints.* arXiv:2605.14067 [cs.LG]. https://arxiv.org/abs/2605.14067. Reproducibility artefact on Zenodo: https://doi.org/10.5281/zenodo.20454212 (CC BY 4.0).
- Sehgal, K., & Bhatti, K. N. (2026). *Auditable Climate Risk Intelligence from Fragmented ESG Data: Deterministic Orchestration and Imbalance-Aware Learning for Scope 1–3 Validation.* arXiv:2606.02604 [cs.LG]. https://arxiv.org/abs/2606.02604. Companion artefact on Zenodo: https://doi.org/10.5281/zenodo.20453252 (CC BY 4.0, OpenAIRE-indexed).

An extended methodological manuscript, *A Reproducibility Protocol for Imbalance-Aware Financial Distress Prediction*, is currently with *Intelligent Computing* (AAAS Science Partner Journal with Zhejiang Lab; IF 4.4, Q1) following editorial invitation.

---

## 8. Links and References

- **Product landing page:** https://www.pink-mate.co.uk
- **Deployment mirror (fallback):** https://v0-pink-mate.vercel.app/
- **Public concept page (Notion):** https://iodized-lily-573.notion.site/Hero-Section-205a5e7da1508078b018e8babfa60c61

**Contact / Founder**
Karan Sehgal — Founder, PinkMate
AI-native GTM, predictive automation, and growth strategy
`karansehgal@pink-mate.co.uk`
