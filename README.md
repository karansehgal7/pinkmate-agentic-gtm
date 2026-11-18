# PinkMate â€” Agentic GTM Engine (Technical Architecture Repository)

PinkMate is an early-stage research and engineering project exploring deterministic, multi-agent orchestration for B2B go-to-market workflows. It combines classical interpretable machine learning with structured agent-level routing to investigate how lead-scoring, segmentation, and outreach decisions can be made more reproducible, auditable, and transparent than conventional CRM automation or LLM-only agent loops.

> **Repository scope.** This repository documents the technical architecture, agent stack, and reference skeletons for PinkMate's predictive GTM engine. It is a **research-engineering artefact** â€” the agent skeletons and orchestration scaffolding are public so the design can be inspected and built upon. Trained model artefacts, production data pipelines, and end-to-end runnable workflows are **work in progress** and not present in this repository in their current form.
>
> **License:** Proprietary â€” source available for evaluation only. See `LICENSE` file for details.

---

## 1. Technical Overview

**Core thesis.** Conventional CRM automation stores leads and applies static rules. LLM-only agent systems generate fluent action sequences but exhibit non-deterministic execution, weak auditability, and unstable explainability. PinkMate explores a middle path: **classical interpretable ML for scoring, bounded modular agents for orchestration, and explicit state transitions for reproducibility.**

**Design choices.**

- **Multi-agent pipeline:** Ingestion â†’ Enrichment â†’ Predictive Scoring â†’ Segmentation â†’ Outreach â†’ Feedback loop, with each agent responsible for a bounded function and observable state transition.
- **Predictive ML (planned implementation):** classical interpretable models â€” Naive Bayes, Random Forest, Gradient Boosting â€” chosen deliberately over opaque deep-learning alternatives to support feature attribution and governance review.
- **Agentic orchestration:** LangChain / LangGraph-style routing between agents, with explicit state schemas and event-triggered transitions.
- **Memory and context (planned):** vector-based memory for company, persona, and interaction history, enabling retrieval-augmented context for downstream agents.
- **Integration surface (planned):** designed to interoperate with Clay, Apollo, PhantomBuster, HubSpot, and Salesforce.

---

## 2. Engineering Positioning

PinkMate is the **reference deployment of the Q-MDP orchestration framework** named in the paper's final section. The architectural choices here â€” bounded agent responsibilities, superposition-aware belief state, governance-gated conditional transitions, interpretable classical ML for scoring, and explainability-oriented design â€” directly instantiate the formal constructs of:

> Sehgal, K., & Bhatti, K. N. (2026). *Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure.* **Accepted at IEEE Cyber-AI 2026**, Bucharest, Romania. Submission #4554.

A full paper-to-code map is in [`docs/architecture.md`](./docs/architecture.md). The repository is now in active build against that map on a weekly slice cadence.

---

## 3. Repository Structure

```
pinkmate-agentic-gtm/
â”‚
â”œâ”€ README.md                          # This file â€“ high-level overview
â”œâ”€ LICENSE                            # Proprietary-source-available
â”œâ”€ CODE_OF_CONDUCT.md
â”œâ”€ SECURITY.md
â”œâ”€ SLICE_5_1_VERDICTS.md              # Canonical cross-domain verdicts (Slice 5.1)
â”œâ”€ requirements.txt                   # Python dependencies
â”‚
â”œâ”€ pinkmate/                          # Core package
â”‚   â”œâ”€ __init__.py
â”‚   â”œâ”€ qmdp.py                        # Belief state, superposed belief, governance
â”‚   â”‚                                 # commit predicate (paper Â§III.A) â€” Slice 1
â”‚   â”œâ”€ state.py                       # Pydantic PipelineState + Ï„_exec trace (Â§III.A)
â”‚   â”‚                                 # + LEADINPUT_SCHEMA_VERSION constant (Slice 5.2)
â”‚   â”œâ”€ persistence.py                 # Checkpointer factory â€” MemorySaver | SqliteSaver
â”‚   â”‚                                 # (paper Â§III.B replay fidelity) â€” Slice 3
â”‚   â”œâ”€ benchmarks/                    # Empirical validation harnesses â€” Slice 4
â”‚   â”‚   â”œâ”€ __init__.py
â”‚   â”‚   â”œâ”€ latency.py                 # T_fa = O(log n + k) scheduling-bound harness
â”‚   â”‚   â””â”€ replay_fidelity.py         # Reusable cross-process replay fidelity harness
â”‚   â””â”€ ml/                            # ML layer across the Slice arc
â”‚       â”œâ”€ __init__.py
â”‚       â”œâ”€ dataset.py                 # Taiwan Bankruptcy UCI 572 loader â€” Slice 2
â”‚       â”œâ”€ imbalance.py               # SMOTE wrapper (Â§III.C) â€” Slice 2
â”‚       â”œâ”€ model_zoo.py               # Six-model ensemble (Â§III.D) â€” Slice 2
â”‚       â”œâ”€ attribution.py             # TreeSHAP + Ïƒ_SHAP (Â§V) â€” Slice 2
â”‚       â”œâ”€ pipeline.py                # Taiwan training + measurement runner â€” Slice 2
â”‚       â”œâ”€ leadforge_loader.py        # LeadForge Lead Scoring v1 loader â€” Slice 5
â”‚       â”œâ”€ x_education_loader.py      # X Education Zenodo 23791277 loader â€” Slice 5.1
â”‚       â”œâ”€ feature_engineering.py     # RFM + Clay-style ICP pure functions â€” Slice 5.2
â”‚       â”œâ”€ calibration_variance.py    # Ïƒ_calibration sibling to Ïƒ_SHAP â€” Slice 5.3
â”‚       â”œâ”€ survival.py                # 6-model survival ensemble â€” Slice 6
â”‚       â”œâ”€ survival_data.py           # LeadForge â†’ survival schema transform â€” Slice 6
â”‚       â””â”€ survival_attribution.py    # SurvSHAP + Ïƒ_SHAP_hazard â€” Slice 6
â”‚
â”œâ”€ agents/                            # Orchestration-graph vertices (paper Â§III.B)
â”‚   â”œâ”€ __init__.py
â”‚   â”œâ”€ ingestion_agent.py
â”‚   â”œâ”€ enrichment_agent.py            # RFM + ICP computation step (Slice 5.2)
â”‚   â”œâ”€ scoring_agent.py               # Hybrid fail-fast schema contract (Slice 5.2)
â”‚   â”‚                                 # + deployment-mode flag (Slice 3.5)
â”‚   â”œâ”€ segmentation_agent.py
â”‚   â””â”€ outreach_agent.py
â”‚
â”œâ”€ workflows/                         # Orchestrated workflows and routing
â”‚   â”œâ”€ pipeline.py                    # LangGraph StateGraph with governance-gated edge
â”‚   â”‚                                 # (Slice 1 + Slice 3 durable checkpointing)
â”‚   â””â”€ pipeline.yaml                  # DEPRECATED; retained for commit-history continuity
â”‚
â”œâ”€ scripts/                           # Standalone reproduction + training scripts
â”‚   â”œâ”€ __init__.py
â”‚   â”œâ”€ reproduce_table2.py            # Reproduces paper Table I + Table II from UCI 572,
â”‚   â”‚                                 # writes to models/taiwan/ â€” Slice 2
â”‚   â”œâ”€ run_benchmarks.py              # Latency + replay-fidelity harnesses,
â”‚   â”‚                                 # writes docs/benchmarks_report.md â€” Slice 4
â”‚   â”œâ”€ train_gtm_ensemble.py          # Trains 6-model ensemble on LeadForge intermediate,
â”‚   â”‚                                 # writes models/gtm/ + report + figures â€” Slice 5
â”‚   â”œâ”€ train_xeducation_ensemble.py   # Trains 6-model ensemble on X Education real data,
â”‚   â”‚                                 # writes models/xeducation/ + report â€” Slice 5.1
â”‚   â”œâ”€ measure_sigma_calibration.py   # Reuses Slice 5/5.1 pickles, correlates Ïƒ_calibration
â”‚   â”‚                                 # with Ïƒ_SHAP across 4 datasets â€” Slice 5.3
â”‚   â””â”€ train_survival_ensemble.py     # Trains 6 survival models + computes Ïƒ_SHAP_hazard
â”‚                                     # + correlates with classification Ïƒ_SHAP â€” Slice 6
â”‚
â”œâ”€ tests/                             # Reproducibility + determinism tests (303 total)
â”‚   â”œâ”€ __init__.py
â”‚   â”œâ”€ test_qmdp.py                   # Slice 1 â€” Q-MDP primitives
â”‚   â”œâ”€ test_pipeline.py               # Slice 1 â€” LangGraph integration + replay
â”‚   â”œâ”€ test_imbalance.py              # Slice 2 â€” SMOTE
â”‚   â”œâ”€ test_attribution.py            # Slice 2 â€” Ïƒ_SHAP
â”‚   â”œâ”€ test_model_zoo.py              # Slice 2 â€” six-model ensemble
â”‚   â”œâ”€ test_dataset.py                # Slice 2 â€” Taiwan loader + preprocessing
â”‚   â”œâ”€ test_slice2_integration.py    # Slice 2 â€” full pipeline + Â§VI.C findings
â”‚   â”œâ”€ test_checkpointing.py         # Slice 3 â€” SqliteSaver + cross-process replay
â”‚   â”œâ”€ test_benchmarks.py             # Slice 4 â€” latency + replay-fidelity harness
â”‚   â”œâ”€ test_scoring_agent_modes.py   # Slice 3.5 â€” ScoringAgent deployment-mode flag
â”‚   â”œâ”€ test_leadforge_loader.py      # Slice 5 â€” LeadForge loader + synthetic fixtures
â”‚   â”œâ”€ test_gtm_training.py           # Slice 5 â€” GTM training qualitative-direction tests
â”‚   â”œâ”€ test_x_education_loader.py    # Slice 5.1 â€” X Education loader + Zenodo fetch
â”‚   â”œâ”€ test_xeducation_training.py    # Slice 5.1 â€” X Education training pipeline
â”‚   â”œâ”€ test_feature_engineering.py   # Slice 5.2 â€” RFM + Clay ICP pure functions
â”‚   â”œâ”€ test_scoring_agent_schema.py  # Slice 5.2 â€” hybrid fail-fast schema contract
â”‚   â”œâ”€ test_calibration_variance.py  # Slice 5.3 â€” Ïƒ_calibration 3 properties
â”‚   â”œâ”€ test_survival.py               # Slice 6 â€” 6 survival models correctness
â”‚   â”œâ”€ test_survival_data.py          # Slice 6 â€” LeadForge â†’ survival transform
â”‚   â””â”€ test_survival_attribution.py  # Slice 6 â€” SurvSHAP + Ïƒ_SHAP transfer
â”‚
â””â”€ docs/                              # Technical documentation
    â”œâ”€ architecture.md                # Paper â†” code map, slice roadmap (Â§1-9)
    â”œâ”€ benchmarks_report.md           # Slice 4 empirical validation report
    â”œâ”€ benchmarks_report.json         # Machine-readable benchmark payload
    â”œâ”€ gtm_training_report.md         # Slice 5 GTM training report
    â”œâ”€ xeducation_training_report.md  # Slice 5.1 X Education training report
    â”œâ”€ gtm_advanced/                  # Slice 5.1 LeadForge advanced tier artefacts
    â”‚   â”œâ”€ gtm_training_report.md
    â”‚   â””â”€ figures/gtm/
    â”œâ”€ sigma_calibration_report.md    # Slice 5.3 Ïƒ_calibration report (generated)
    â”œâ”€ survival_training_report.md    # Slice 6 survival training report (generated)
    â””â”€ figures/                       # Committed PNG charts for in-browser review
        â”œâ”€ latency_total_vs_n.png
        â”œâ”€ latency_per_step_overhead.png
        â”œâ”€ replay_fidelity_summary.png
        â”œâ”€ gtm/                       # Slice 5 figures
        â”‚   â”œâ”€ classification_metrics.png
        â”‚   â””â”€ sigma_shap_per_model.png
        â”œâ”€ xeducation/                # Slice 5.1 figures
        â”‚   â”œâ”€ classification_metrics.png
        â”‚   â””â”€ sigma_shap_per_model.png
        â”œâ”€ sigma_calibration/         # Slice 5.3 figures (generated)
        â””â”€ survival/                  # Slice 6 figures (generated)
            â”œâ”€ aic_comparison.png
            â”œâ”€ sigma_shap_per_survival_model.png
            â””â”€ sigma_shap_transfer_scatter.png
```

**Note on `models/` subdirectory structure.** `models/` has four gitignored sibling subdirectories after Slice 6: `models/taiwan/` (Slice 2 paper reproduction), `models/gtm/` (Slice 5 LeadForge intermediate), `models/gtm_advanced/` + `models/xeducation/` (Slice 5.1 cross-domain validation), and `models/survival/` (Slice 6 survival ensemble). Run `python scripts/reproduce_table2.py` to populate `models/taiwan/`; run `python scripts/train_gtm_ensemble.py` for `models/gtm/`; run `python scripts/train_xeducation_ensemble.py` for `models/xeducation/`; run `python scripts/train_survival_ensemble.py` for `models/survival/`. The `ScoringAgent` default `models_dir` is `models/gtm/` post-Slice-5 (product-first positioning); Taiwan reproduction uses explicit `ScoringAgent(models_dir="models/taiwan")`.

**Slice 1 status (shipped).** The Q-MDP formal layer (`pinkmate/qmdp.py`, `pinkmate/state.py`) and the LangGraph orchestration graph (`workflows/pipeline.py`) are live and executable end-to-end on a mock lead with `python -m workflows.pipeline`.

**Slice 2 status (shipped).** The imbalance-aware ML layer is live: six-model ensemble, SMOTE, TreeSHAP Ïƒ_SHAP measurement harness, and a reproduction script (`scripts/reproduce_table2.py`) that reproduces the paper's Table I and Table II against the real UCI 572 dataset with tolerance-based assertions (Ïƒ_SHAP Â±0.005, recall Â±0.03). The ScoringAgent now runs the real six-model ensemble when `models/` is populated, and falls back transparently to the Slice 1 deterministic placeholder predictor (with a `RuntimeWarning`) when models are not yet trained.

**Slice 3 status (shipped).** Durable checkpointing is live: a `make_checkpointer()` factory in `pinkmate/persistence.py` selects between `MemorySaver` (default, in-process) and `SqliteSaver` (on-disk, cross-process). `run_pipeline(initial, thread_id, checkpoint_backend="sqlite", ...)` persists every post-node `PipelineState` snapshot to disk keyed by `thread_id`, operationalising the paper's Â§III.B replay-fidelity claim. Twelve new tests in `tests/test_checkpointing.py` verify SQLite persistence, cross-process replay fidelity, thread isolation, and the factory contract.

**Slice 4 status (shipped).** Empirical validation of the paper's Â§III.B operational claims is live: `pinkmate/benchmarks/latency.py` measures total pipeline latency across n âˆˆ {5, 10, 25, 50, 100, 250}; `pinkmate/benchmarks/replay_fidelity.py` generalises Slice 3's cross-process replay proof into a **reusable consumer harness**; `scripts/run_benchmarks.py` writes `docs/benchmarks_report.md`, `docs/benchmarks_report.json`, and three PNG charts under `docs/figures/`. 20 new tests in `tests/test_benchmarks.py`. Current empirical result: linear fit RÂ² = 0.98 on chain topology, per-step scheduler overhead bounded across the n range, 3/3 fidelity threads pass strict replay-fidelity.

**Slice 3.5 status (shipped).** `ScoringAgent(model_source="zoo"|"placeholder")` constructor flag separates the research-reproduction path (loads from `models/`, warnings on mismatch) from the product-demo path (skips model loading entirely). 9 new tests in `tests/test_scoring_agent_modes.py`.

**Slice 5 status (shipped â€” methodology validation on synthetic B2B-shaped data).** GTM-domain extension trained on LeadForge Lead Scoring v1 (MIT, synthetic B2B SaaS). `scripts/train_gtm_ensemble.py` writes `models/gtm/`, `docs/gtm_training_report.md`, `docs/figures/gtm/*.png`. 36 new tests. **Ïƒ_SHAP numbers DO NOT constitute performance claims on real B2B GTM leads** â€” methodology validation per Slice 2's Taiwan-benchmark honesty bar.

**Slice 5.1 status (shipped 16 Nov 2026).** Cross-domain validation on the first REAL dataset in the build programme: X Education Lead Scoring v1 (Zenodo 23791277, CC BY 4.0, 9240 leads, ~38.54% conversion) + LeadForge advanced tier (synthetic B2B GTM with injected missingness and outliers, harder-conditions stress test). Commits `5cde402` + `a3d3606`. **All 4 qualitative directions PASS** across the full 3.23% â†’ 38.54% imbalance spectrum (Taiwan 3.23% â†’ LeadForge advanced 7.86% â†’ LeadForge intermediate 20.98% â†’ X Education 38.54%). **Monotone SMOTE-Î”-decrease trend empirically visible:** Î” = +0.0155 â†’ +0.0108 â†’ +0.0004 as minority rate approaches balance, exactly as the methodological definition predicts. Canonical verdicts record: [`SLICE_5_1_VERDICTS.md`](./SLICE_5_1_VERDICTS.md).

**Slice 5.2 status (shipped 16 Nov 2026).** Deployment-schema alignment + RFM + Clay-style ICP + MTA formalism. Commit `7ed4031`. `ScoringAgent._predict_proba_real()` now runs a **hybrid fail-fast schema contract** (STRICT on required features via `RuntimeError`, LENIENT on extras via DEBUG log, WARN on `LEADINPUT_SCHEMA_VERSION` drift) instead of silent zero-fill. `pinkmate/ml/feature_engineering.py` adds pure RFM functions (90-day recency decay + log-scaled frequency saturating at 20 touches + monetary ACV band step function + composite) and Clay-style ICP scoring (5 rules + composite 0-100 + A/B/C/D tier). `pinkmate/state.py` exports `LEADINPUT_SCHEMA_VERSION = "slice52_rfm_icp_v1"` + 12 new optional `LeadInput` fields. `docs/architecture.md` Section 5f adds an explicit MTA (Multi-Touch Attribution) formalism showing Q-MDP belief-state transitions implement a Markov chain isomorphic to the Shao & Li 2011 data-driven MTA model (full formal proof deferred to Slice 7). 201 passing tests at shipping.

**Slice 5.3 status (shipped 16 Nov 2026).** Ïƒ_calibration = calibration-stability companion to Ïƒ_SHAP attribution-stability. Commit `10949bb`. Both measured under the SAME K-rotation protocol (K=50 default, matching paper Â§V canonical). `pinkmate/ml/calibration_variance.py` mirrors `attribution.py` API (`sigma_calibration` + `compute_rotated_calibration_variance` sibling to `sigma_shap` + `compute_rotated_shap_variance`). `scripts/measure_sigma_calibration.py` reuses Slice 5/5.1 trained model pickles across all 4 cross-domain datasets, correlates Ïƒ_calibration with Ïƒ_SHAP via Spearman + 95% bootstrap CI, writes `docs/sigma_calibration_report.md`. Three properties enforced by `tests/test_calibration_variance.py`: (1) bounded Ïƒ âˆˆ [0, 0.5]; (2) K-stable (Ïƒ at K=50 within 0.05 of Ïƒ at K=10); (3) reproducible (same seed â†’ identical Ïƒ). 241 passing tests at shipping.

**Slice 6 status (shipped 18 Nov 2026).** Survival analysis extension â€” Cox PH primary + 3 parametric AFT families (Weibull / LogN / LogL) + XGBoost `survival:cox` + LightGBM AFT. Commit `a131ba9`. Introduces a new top-level `## 6. Survival analysis extension` section in `docs/architecture.md` (methodological jump: different prediction target, different loss, different attribution structure; existing Â§6/Â§7/Â§8 renumbered to Â§7/Â§8/Â§9). `pinkmate/ml/survival.py` exposes a unified `_BaseSurvivalWrapper` interface over lifelines + XGBoost + LightGBM; `pinkmate/ml/survival_data.py` transforms LeadForge columns into `(time, event)` via Decision B=3 synthetic `days_since_first_touch` construction with honesty-bar sensitivity sweep over touch-interval âˆˆ {5, 7, 14}; `pinkmate/ml/survival_attribution.py` provides `sigma_shap_hazard` + `compute_rotated_survshap_variance` mirroring `attribution.py` API, backed by the KrzyziÅ„ski et al. 2023 `survshap` package with a finite-difference cumulative-hazard fallback. `scripts/train_survival_ensemble.py` answers three research questions in `docs/survival_training_report.md`: (1) does Ïƒ_SHAP transfer from classification to survival (Spearman Ï + 95% bootstrap CI); (2) is Cox PH Ïƒ_SHAP lower than Weibull Ïƒ_SHAP; (3) which parametric family fits LeadForge best by AIC. **Bank of England PRA SS1/23 credit-risk alignment:** Cox PH is the regulatory convention for survival-style credit-risk modelling; shipping Ïƒ_SHAP on both semi-parametric and parametric families gives the RAEng Route 4 evidence pack a regulatory-adjacent methodology line. 62 new tests (303 total). Dependencies added: `lifelines>=0.27.0`, `scikit-survival>=0.22.0`, `survshap>=0.4.0`.

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

## 5. ScoringAgent â€” Reference Skeleton

The `ScoringAgent` class in `agents/scoring_agent.py` defines the interface contract for lead scoring: a feature dictionary in, a structured score-plus-breakdown response out.

Core responsibilities:

- Load trained classical ML models from `models/gtm/` (default) or an explicit `models_dir`.
- Accept a dictionary of engineered features (now including RFM + Clay ICP features from Slice 5.2).
- Apply the Slice 5.2 hybrid fail-fast schema contract: STRICT on required training features, LENIENT on extras, WARN on `LEADINPUT_SCHEMA_VERSION` drift.
- Combine model outputs into a final score with per-model breakdown.
- Return a structured response with `final_score` (0â€“100) and per-model `model_scores`.

Slice 3.5 added the `model_source="zoo"|"placeholder"` constructor flag so the ScoringAgent can be instantiated for product-demo contexts (website, pitch decks) without loading real models or emitting fallback warnings.

---

## 6. Roadmap â€” Slice Cadence

PinkMate is being built against the paper on a weekly slice cadence. Each slice closes a specific gap between the paper's formal claims and the reference deployment.

- **Slice 1 (shipped).** Q-MDP formal structure wired into LangGraph with deterministic-replay smoke tests. `pinkmate/qmdp.py`, `pinkmate/state.py`, `workflows/pipeline.py`, test harness, paper-to-code map in `docs/architecture.md`.
- **Slice 2 (shipped).** Imbalance-aware ML layer per paper Â§III.Câ€“D and Â§V â€” six-model ensemble (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost), SMOTE on training partition only, TreeSHAP attribution with Ïƒ_SHAP measurement harness. `scripts/reproduce_table2.py` reproduces Table I and Table II against UCI 572 with tolerance-based assertions.
- **Slice 3 (shipped).** Durable checkpointing per paper Â§III.B replay-fidelity claim â€” `pinkmate/persistence.py::make_checkpointer()` factory, `MemorySaver` â†’ `SqliteSaver`, 12 new tests covering SQLite persistence and cross-process replay fidelity.
- **Slice 4 (shipped).** Empirical validation of paper Â§III.B operational claims â€” latency + replay-fidelity harnesses write `docs/benchmarks_report.md` + figures. 20 new tests.
- **Slice 3.5 (shipped).** ScoringAgent deployment-mode flag â€” product-arc ergonomics. `model_source="zoo"|"placeholder"` constructor flag. 9 new tests.
- **Slice 5 (shipped â€” methodology validation on synthetic B2B-shaped data).** GTM-domain extension trained on LeadForge Lead Scoring v1 (MIT, synthetic). 36 new tests. Ïƒ_SHAP numbers reported as methodology output, not performance claims.
- **Slice 5.1 (shipped 16 Nov 2026).** Cross-domain validation on REAL EdTech data (X Education via Zenodo 23791277 CC BY 4.0) + LeadForge advanced tier (stress test). All 4 qualitative directions PASS across Taiwan 3.23% â†’ LeadForge advanced 7.86% â†’ LeadForge intermediate 20.98% â†’ X Education 38.54%. Monotone SMOTE-Î”-decrease trend empirically visible (+0.0155 â†’ +0.0108 â†’ +0.0004). Commits `5cde402` + `a3d3606`.
- **Slice 5.2 (shipped 16 Nov 2026).** Deployment-schema alignment (hybrid fail-fast ScoringAgent contract) + RFM feature engineering + Clay-style ICP scoring (5 rules + composite + A/B/C/D tier) + MTA formalism in `docs/architecture.md` Section 5f. Commit `7ed4031`. 201 passing tests.
- **Slice 5.3 (shipped 16 Nov 2026).** Ïƒ_calibration governance sibling to Ïƒ_SHAP (same K=50 rotation protocol). `pinkmate/ml/calibration_variance.py` mirrors `attribution.py` API. Three properties enforced: bounded, K-stable, reproducible. Research question "does Ïƒ_calibration correlate with Ïƒ_SHAP" reported via Spearman + 95% bootstrap CI in `docs/sigma_calibration_report.md`. Commit `10949bb`. 241 passing tests.
- **Slice 6 (shipped 18 Nov 2026).** Survival analysis extension â€” Cox PH primary + 3 parametric AFT + XGBoost `survival:cox` + LightGBM AFT + SurvSHAP. New top-level `docs/architecture.md` Section 6 (methodological jump, not sub-slice under Â§5). Research question "does Ïƒ_SHAP transfer from classification to survival" answered via Spearman correlation on matched model-family pairs. Bank of England PRA SS1/23 credit-risk alignment. Commit `a131ba9`. 303 passing tests.
- **Slice 7 (planned).** POMDP formalization + Bayesian + mixture + HMM â€” probabilistic framework formalisation bundling the theoretical scaffolding referenced from Slice 5.2 MTA section and Slice 6 survival section. Likely a fresh top-level `docs/architecture.md` Section (renumber pattern as Slice 6).
<!-- APPEND this block to README.md Section 6 Roadmap, inserted BEFORE the "Slice 7 (planned)" bullet. -->
<!-- Integration guide Step 4 uses regex-replace to insert at the correct position, NOT Add-Content EOF (per 18 Nov 2026 housekeeping lesson). -->

- **Slice 7 (shipped 18 Nov 2026).** Probabilistic framework formalisation â€” bundled per the 9 November 2026 scope decision. New top-level `docs/architecture.md` Section 7 (methodological jump: fresh theoretical scaffolding, not survival evidence); existing Â§7/Â§8/Â§9 renumbered to Â§8/Â§9/Â§10. Four modules shipped: `pinkmate/qmdp_formal.py` (POMDP tuple + Bayesian belief filter + Q-MDP commit algorithm with Littman, Cassandra & Kaelbling 1995 citation + isomorphism adapter proving `SuperposedBelief` â‰¡ POMDP belief and `governance_commit` â‰¡ degenerate Q-MDP); `pinkmate/ml/hmm_belief.py` (3-state `LatentStateHMM` wrapping `hmmlearn.CategoricalHMM` with Viterbi + forward-backward + Baum-Welch fit); `pinkmate/ml/mixture_survival.py` (EM-fit Weibull mixture with right-censoring + BIC-based K selection across K âˆˆ {2, 3, 4}); `pinkmate/ml/bayesian_survival.py` (PyMC 5.x+ Bayesian Weibull with HalfNormal priors + NUTS sampling + posterior-predictive credible intervals on S(t)). New one-stop theoretical reference `docs/probabilistic_framework.md`. Four decisions locked 17 November 2026: A=PyMC over NumPyro, B=NUTS sampler, C=BIC sweep K âˆˆ {2,3,4}, D=hmmlearn with 3 latent states (MQL/SQL/ClosedWon). Dependencies added: `pymc>=5.0.0`, `hmmlearn>=0.3.0`. 56 new tests (359 total).


- **Slice 7.5 (planned).** Conformal prediction overlay on Î¸_governance + PSI/KL distribution-shift sentinels. Production-hardens the ScoringAgent for ongoing-monitoring regulatory conversations.
- **Slice 8+ (optional extensions).** Fairness audits (Fairlearn), Kolmogorov-Arnold Networks comparison on lead scoring (Lee et al. 2025), DBpedia-enriched feature engineering (arXiv 2606.28355).
- **Later (data-gated).** Real B2B conversion data path â€” `pink-mate.co.uk` landing-page capture over 6-12 months, academic partnership via Kent Business School / Khawar network, or specific industry engagement.
- **Later.** Integration surface â€” connectors to Clay, PhantomBuster, Apollo, HubSpot, and Salesforce.

Numbers will be reported only once measured against the slices that produce them.

---

## 7. Related Research

The architectural and research thesis underlying PinkMate is developed in academic collaboration with Kent Business School, University of Kent. Peer-reviewed work:

- Sehgal, K., & Bhatti, K. N. (2026). *Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure.* **Accepted** at IEEE Cyber-AI 2026, Bucharest, Romania. Submission #4554 (accepted 18 June 2026).
- Sehgal, K., & Bhatti, K. N. (2026). *Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints.* arXiv:2605.14067 [cs.LG]. https://arxiv.org/abs/2605.14067. Reproducibility artefact on Zenodo: https://doi.org/10.5281/zenodo.20454212 (CC BY 4.0).
- Sehgal, K., & Bhatti, K. N. (2026). *Auditable Climate Risk Intelligence from Fragmented ESG Data: Deterministic Orchestration and Imbalance-Aware Learning for Scope 1â€“3 Validation.* arXiv:2606.02604 [cs.LG]. https://arxiv.org/abs/2606.02604. Companion artefact on Zenodo: https://doi.org/10.5281/zenodo.20453252 (CC BY 4.0, OpenAIRE-indexed).

An extended methodological manuscript, *A Reproducibility Protocol for Imbalance-Aware Financial Distress Prediction*, is currently with *Intelligent Computing* (AAAS Science Partner Journal with Zhejiang Lab; IF 4.4, Q1) following editorial invitation.

---

## 8. Links and References

- **Product landing page:** https://www.pink-mate.co.uk
- **Deployment mirror (fallback):** https://v0-pink-mate.vercel.app/
- **Public concept page (Notion):** https://iodized-lily-573.notion.site/Hero-Section-205a5e7da1508078b018e8babfa60c61

**Contact / Founder**
Karan Sehgal â€” Founder, PinkMate
AI-native GTM, predictive automation, and growth strategy
`karansehgal@pink-mate.co.uk`
