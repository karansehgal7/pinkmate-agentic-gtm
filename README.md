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
├─ README.md                  # This file – high-level overview
├─ requirements.txt           # Python dependencies
│
├─ pinkmate/                  # Core package
│   ├─ __init__.py
│   ├─ qmdp.py                # Belief state, superposed belief, governance
│   │                         # commit predicate (paper §III.A) — Slice 1
│   ├─ state.py               # Pydantic PipelineState + τ_exec trace (§III.A) — Slice 1
│   ├─ persistence.py         # Checkpointer factory — MemorySaver | SqliteSaver
│   │                         # (paper §III.B replay fidelity) — Slice 3
│   ├─ benchmarks/            # Empirical validation harnesses — Slice 4
│   │   ├─ __init__.py
│   │   ├─ latency.py         # T_fa = O(log n + k) scheduling-bound harness (§III.B)
│   │   └─ replay_fidelity.py # Reusable cross-process replay fidelity harness (§III.B)
│   └─ ml/                    # Imbalance-aware ML layer (Slice 2 + Slice 5)
│       ├─ dataset.py              # Taiwan Bankruptcy UCI 572 loader (§VI.A, §VIII) — Slice 2
│       ├─ leadforge_loader.py     # LeadForge Lead Scoring v1 loader — Slice 5 GTM domain
│       ├─ imbalance.py            # SMOTE wrapper (§III.C)
│       ├─ model_zoo.py            # Six-model ensemble (§III.D)
│       ├─ attribution.py          # TreeSHAP + σ_SHAP (§V)
│       └─ pipeline.py             # Taiwan training + measurement runner
│       ├─ __init__.py
│       ├─ dataset.py         # Taiwan Bankruptcy UCI 572 loader (§VI.A, §VIII)
│       ├─ imbalance.py       # SMOTE wrapper (§III.C)
│       ├─ model_zoo.py       # Six-model ensemble (§III.D)
│       ├─ attribution.py     # TreeSHAP + σ_SHAP (§V)
│       └─ pipeline.py        # End-to-end training + measurement runner
│
├─ agents/                    # Orchestration-graph vertices (paper §III.B)
│   ├─ __init__.py
│   ├─ ingestion_agent.py
│   ├─ enrichment_agent.py
│   ├─ scoring_agent.py       # Loads real six-model ensemble from models/
│   │                         # if present; falls back to Slice 1 placeholder
│   │                         # predictor with a RuntimeWarning if not.
│   ├─ segmentation_agent.py
│   └─ outreach_agent.py
│
├─ workflows/                 # Orchestrated workflows and routing
│   ├─ pipeline.py            # LangGraph StateGraph with governance-gated edge
│   │                         # (Slice 1 orchestration + Slice 3 durable
│   │                         #  checkpoint-backend plumbing)
│   └─ pipeline.yaml          # DEPRECATED; retained for commit-history continuity
│
├─ scripts/                   # Standalone reproduction scripts
│   ├─ reproduce_table2.py    # Reproduces paper Table I + Table II from UCI 572,
│   │                         #   writes to models/taiwan/ (Slice 5 subdirectory structure)
│   ├─ train_gtm_ensemble.py  # Slice 5 — trains six-model ensemble on LeadForge
│   │                         #   intermediate tier, writes to models/gtm/,
│   │                         #   writes docs/gtm_training_report.md + docs/figures/gtm/
│   └─ run_benchmarks.py      # Slice 4 — runs latency + replay-fidelity harnesses,
│                             #   writes docs/benchmarks_report.md + docs/figures/*.png
│
├─ tests/                     # Reproducibility + determinism tests
│   ├─ test_qmdp.py                 # Slice 1 — Q-MDP primitives
│   ├─ test_pipeline.py             # Slice 1 — LangGraph integration + replay
│   ├─ test_imbalance.py            # Slice 2 — SMOTE
│   ├─ test_attribution.py          # Slice 2 — σ_SHAP
│   ├─ test_model_zoo.py            # Slice 2 — six-model ensemble
│   ├─ test_dataset.py              # Slice 2 — Taiwan loader + preprocessing
│   ├─ test_slice2_integration.py   # Slice 2 — full pipeline + §VI.C findings
│   ├─ test_checkpointing.py        # Slice 3 — SqliteSaver + cross-process
│   │                               #   replay fidelity (§III.B)
│   ├─ test_benchmarks.py           # Slice 4 — latency + replay-fidelity harness
│   │                               #   correctness tests (§III.B)
│   ├─ test_scoring_agent_modes.py  # Slice 3.5 — ScoringAgent deployment-mode
│   │                               #   flag ("zoo" vs "placeholder") tests
│   ├─ test_leadforge_loader.py     # Slice 5 — LeadForge loader + synthetic
│   │                               #   fixture correctness
│   └─ test_gtm_training.py         # Slice 5 — GTM training pipeline qualitative-
│                                   #   direction tests (§VI.C transferability)
│
└─ docs/                      # Technical documentation
    ├─ architecture.md        # Paper ↔ code map, slice roadmap
    ├─ benchmarks_report.md   # Slice 4 empirical validation report (committed)
    ├─ benchmarks_report.json # Machine-readable benchmark payload
    ├─ gtm_training_report.md # Slice 5 GTM training report (generated locally)
    └─ figures/               # Committed PNG charts for in-browser review
        ├─ latency_total_vs_n.png
        ├─ latency_per_step_overhead.png
        ├─ replay_fidelity_summary.png
        └─ gtm/
            ├─ classification_metrics.png
            └─ sigma_shap_per_model.png
```

**Note on `models/` subdirectory structure (Slice 5):** `models/` now has two sibling subdirectories, `models/taiwan/` and `models/gtm/`, both gitignored. Run `python scripts/reproduce_table2.py` to populate `models/taiwan/` with Taiwan-trained models (paper reproduction); run `python scripts/train_gtm_ensemble.py` to populate `models/gtm/` with LeadForge-trained GTM ensemble. The `ScoringAgent` default `models_dir` is `models/gtm/` post-Slice-5 (product-first positioning); Taiwan reproduction uses explicit `ScoringAgent(models_dir="models/taiwan")`.

**Slice 1 status (shipped).** The Q-MDP formal layer (`pinkmate/qmdp.py`, `pinkmate/state.py`) and the LangGraph orchestration graph (`workflows/pipeline.py`) are live and executable end-to-end on a mock lead with `python -m workflows.pipeline`.

**Slice 2 status (shipped).** The imbalance-aware ML layer is live: six-model ensemble, SMOTE, TreeSHAP σ_SHAP measurement harness, and a reproduction script (`scripts/reproduce_table2.py`) that reproduces the paper's Table I and Table II against the real UCI 572 dataset with tolerance-based assertions (σ_SHAP ±0.005, recall ±0.03). The ScoringAgent now runs the real six-model ensemble when `models/` is populated, and falls back transparently to the Slice 1 deterministic placeholder predictor (with a `RuntimeWarning`) when models are not yet trained.

**Slice 3 status (shipped).** Durable checkpointing is live: a `make_checkpointer()` factory in `pinkmate/persistence.py` selects between `MemorySaver` (default, in-process) and `SqliteSaver` (on-disk, cross-process). `run_pipeline(initial, thread_id, checkpoint_backend="sqlite", ...)` persists every post-node `PipelineState` snapshot to disk keyed by `thread_id`, operationalising the paper's §III.B replay-fidelity claim. Twelve new tests in `tests/test_checkpointing.py` verify SQLite persistence, cross-process replay fidelity (a fresh `SqliteSaver` instance pointed at the same DB file recovers the terminal state and full execution trace), thread isolation, and the factory contract. PostgreSQL swap for team/multi-instance deployments is the next natural extension.

**Slice 4 status (shipped).** Empirical validation of the paper's §III.B operational claims is live: `pinkmate/benchmarks/latency.py` measures total pipeline latency across n ∈ {5, 10, 25, 50, 100, 250} on a parametric chain-topology StateGraph with near-no-op pass-through nodes, isolating the per-step scheduler cost from per-node execution. `pinkmate/benchmarks/replay_fidelity.py` generalises Slice 3's cross-process replay proof into a **reusable consumer harness** that any Q-MDP deployment (PinkMate, the ESG companion, future instantiations) can run against its own checkpointed audit trails to emit a structured completeness-and-divergence report. `scripts/run_benchmarks.py` executes both and commits the artefacts to the repo for in-browser review: `docs/benchmarks_report.md`, `docs/benchmarks_report.json`, and three PNG charts under `docs/figures/`. 20 new tests in `tests/test_benchmarks.py` verify harness correctness (not empirical fit numbers — those legitimately wobble under OS jitter). Current empirical result: linear fit R² = 0.98 on chain topology (expected — sequential dependencies force linear scaling), per-step scheduler overhead stays bounded across the n range (the operational signal for the `O(log n + k)` bound), 3/3 fidelity threads pass strict replay-fidelity.

**Slice 3.5 status (shipped).** ScoringAgent deployment-mode flag — product-arc ergonomics. The `ScoringAgent(model_source="zoo"|"placeholder")` constructor flag selects between two deployment paths without changing any downstream behaviour: `"zoo"` (default, backward-compatible) preserves Slice 2 behaviour (load models from `models/`, fall back with `RuntimeWarning` on mismatch) for research reproduction and RAEng/academic review; `"placeholder"` skips model loading entirely (no file reads, no warnings) for the product-demo surface (website, pitch decks, client demos) where the Taiwan-vs-GTM warnings would be noise rather than signal. 9 new tests in `tests/test_scoring_agent_modes.py` verify both paths + their equivalence contract when `models/` is empty. Becomes effectively equivalent to `"zoo"` mode once Slice 5 ships GTM-domain models matching the deployment feature schema.

**Slice 5 status (shipped — methodology validation on synthetic B2B-shaped data).** GTM-domain extension of the Slice 2 ensemble + σ_SHAP governance harness, applied to LeadForge Lead Scoring v1 (MIT licence, synthetic B2B SaaS procurement scenario on HuggingFace). `pinkmate/ml/leadforge_loader.py` handles the HuggingFace download + one-hot-encoding of nine categorical firmographic + contact + source columns + explicit exclusion of the documented `total_touches_all` leakage trap + identity/timing columns. `scripts/train_gtm_ensemble.py` trains the six-model ensemble on the intermediate-tier (canonical difficulty per LeadForge docs), measures classification metrics on the held-out test partition, measures σ_SHAP for XGBoost + LightGBM under SMOTE + XGBoost-without-SMOTE baseline (direction control), and writes committed artefacts: `models/gtm/*.pkl`, `docs/gtm_training_report.md`, `docs/figures/gtm/*.png`. `agents/scoring_agent.py` default `models_dir` changes to `models/gtm/`; `scripts/reproduce_table2.py` writes to `models/taiwan/` (subdirectory separation per Slice 5 Option A). `pinkmate/state.py` `LeadInput` gets optional `employee_band`, `estimated_revenue_band`, `process_maturity_band` fields (backward-compatible defaults = None); `funding_stage` retained in API but dropped from the Slice 5 training schema (LeadForge does not include it). 36 new tests in `tests/test_leadforge_loader.py` + `tests/test_gtm_training.py` cover loader correctness, synthetic fixture shape + class balance, HF-unreachable fallback, and qualitative-direction verdicts (SMOTE INCREASES σ_SHAP on XGBoost; LightGBM σ_SHAP < XGBoost σ_SHAP). **Honesty bar**: σ_SHAP numbers reported here are the output of applying the Q-MDP σ_SHAP governance harness to B2B-shaped data for the first time; they DO NOT constitute performance claims about PinkMate on real B2B GTM leads. Real-data validation deferred to Slice 5.1+ (X Education sanity check + LeadForge advanced tier + any real B2B corpus that accumulates via pink-mate.co.uk landing-page capture / academic partnership / industry engagement).

---

## 4. Agent Stack

PinkMate's GTM engine is modelled as a set of independent but coordinated agents:

- **IngestionAgent**
  - Accepts CSV uploads, LinkedIn exports, Apollo data, or direct API feeds.
  - Normalises and validates lead records.

- **EnrichmentAgent**
  - Calls enrichment sources (Clay, Clearbit, Apollo, internal APIs).
  - Adds firmographics (industry, size, location, funding) and buyer signals.

- **ScoringAgent**
  - Designed to apply classical ML models (Naive Bayes, Random Forest, Gradient Boosting) — model-loading logic and training artefacts are pending.
  - Will produce a probability of conversion and a confidence score.

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

Core responsibilities (planned implementation):

- Load trained classical ML models from `models/`.
- Accept a dictionary of engineered features.
- Combine model outputs into a final score.
- Return a structured response with `final_score` (0–100) and per-model `model_scores`.

The current implementation contains explicit placeholder return values (e.g. `nb_score = 0.78  # simulated`) so the interface contract is inspectable before the real model-loading logic is added. This is intentional — the file is a **reference skeleton**, not a deployed scoring service.

---

## 6. Roadmap — Slice Cadence

PinkMate is being built against the paper on a weekly slice cadence. Each slice closes a specific gap between the paper's formal claims and the reference deployment.

- **Slice 1 (shipped).** Q-MDP formal structure wired into LangGraph with deterministic-replay smoke tests. `pinkmate/qmdp.py`, `pinkmate/state.py`, `workflows/pipeline.py`, test harness, paper-to-code map in `docs/architecture.md`.
- **Slice 2 (shipped).** Imbalance-aware ML layer per paper §III.C–D and §V — six-model ensemble (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost), SMOTE on training partition only, TreeSHAP attribution with σ_SHAP measurement harness. `scripts/reproduce_table2.py` reproduces Table I and Table II against UCI 572 with tolerance-based assertions. `ScoringAgent` now runs real pickled models when present, with a transparent fallback to the Slice 1 placeholder.
- **Slice 3 (shipped).** Durable checkpointing per paper §III.B replay-fidelity claim — `pinkmate/persistence.py::make_checkpointer()` factory, `MemorySaver` → `SqliteSaver` with per-`thread_id` audit trail on disk, `run_pipeline()` helper threading the durability contract from caller to compiled graph, 12 new tests covering SQLite persistence, cross-process replay fidelity, thread isolation, and the factory contract. PostgreSQL swap is a single-call-site change against this factory in a later slice.
- **Slice 4 (shipped).** Empirical validation of paper §III.B operational claims — `pinkmate/benchmarks/latency.py` measures scheduler latency vs graph size on a parametric chain topology (n ∈ {5, 10, 25, 50, 100, 250}, 10 runs each). `pinkmate/benchmarks/replay_fidelity.py` generalises the cross-process replay proof into a reusable consumer harness emitting structured completeness-and-divergence reports (reusable across deployments including the ESG companion). `scripts/run_benchmarks.py` writes committed artefacts to `docs/benchmarks_report.md` + `docs/figures/*.png` for in-browser review. 20 new harness-correctness tests in `tests/test_benchmarks.py`; empirical fit numbers reported in markdown rather than asserted in tests to avoid flakiness from OS scheduler jitter.
- **Slice 3.5 (shipped).** ScoringAgent deployment-mode flag — product-arc ergonomics, not a paper claim. The `model_source="zoo"|"placeholder"` constructor flag separates the research-reproduction path (loads models from `models/`, warnings on mismatch) from the product-demo path (skips model loading entirely, zero warnings). Backward-compatible default is `"zoo"` so all existing callers continue to work unchanged. 9 new tests in `tests/test_scoring_agent_modes.py` verify both paths + an equivalence contract between them when `models/` is empty.
- **Slice 5 (shipped — methodology validation on synthetic B2B-shaped data).** GTM-domain extension trained on LeadForge Lead Scoring v1 (MIT, synthetic B2B SaaS procurement scenario). `pinkmate/ml/leadforge_loader.py` + `scripts/train_gtm_ensemble.py` train the six-model ensemble on intermediate tier, measure σ_SHAP (with-SMOTE + XGBoost-without-SMOTE baseline), write artefacts to `models/gtm/`, `docs/gtm_training_report.md`, `docs/figures/gtm/*.png`. Subdirectory restructure: `models/` now holds `models/gtm/` and `models/taiwan/` as siblings; `ScoringAgent` default `models_dir` changes to `models/gtm/`; `reproduce_table2.py` writes to `models/taiwan/`. `LeadInput` gains optional `employee_band`, `estimated_revenue_band`, `process_maturity_band` fields; `funding_stage` retained in API but dropped from training schema. 36 new tests cover loader correctness + qualitative-direction verdicts. **σ_SHAP numbers DO NOT constitute performance claims about PinkMate on real B2B GTM leads** — this is methodology validation per Slice 2's Taiwan-benchmark honesty bar.
- **Slice 5.1 (deferred).** Secondary sanity check — train on X Education Lead Scoring (real but EdTech-domain, feature-schema divergent) AND on LeadForge advanced tier (stress-test σ_SHAP harness under higher noise/missingness). Both are "did the methodology survive harder/different conditions?" validations.
- **Later (deferred, data-gated).** Real B2B conversion data path — pink-mate.co.uk landing-page capture over 6-12 months, academic partnership via Kent Business School / Khawar network, or specific industry engagement. When this corpus arrives, retrain Slice 5 ensemble on real labels, re-add `funding_stage` to the training schema, promote to `models/gtm/`.
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
<!-- APPEND this block to README.md — paste at the end of the "Slices shipped" roadmap section, before Section 7 (Related Research). -->

### Slice 5.1 — X Education real-data + LeadForge advanced validation (shipped 16 Nov 2026)

- **Commits:** `5cde402` (scientific artefacts) + `a3d3606` (canonical verdicts record + working-tree cleanup)
- **X Education Lead Scoring v1** (Zenodo 23791277, CC BY 4.0) — the first REAL EdTech B2B lead-scoring dataset in the PinkMate build programme. 9240 leads, ~38.54% conversion rate.
- **LeadForge advanced tier** — synthetic B2B GTM with injected missingness and outliers (harder-conditions stress test).
- **All 4 qualitative directions PASS** across the full 3.23%→38.54% imbalance spectrum (Taiwan / LeadForge advanced / LeadForge intermediate / X Education).
- **Monotone SMOTE-Δ-decrease trend empirically visible:** Δ=+0.0155 at 7.86% minority → +0.0108 at 20.98% → +0.0004 at 38.54%. Exactly as the methodological definition predicts — SMOTE has less synthetic-positive headroom to destabilise attribution as classes approach balance.
- **Canonical verdicts record:** [`SLICE_5_1_VERDICTS.md`](SLICE_5_1_VERDICTS.md)

### Slice 5.2 — Deployment-schema alignment + RFM + Clay ICP + MTA formalism (shipped <date> 2026)

- **ScoringAgent hybrid fail-fast schema contract** (Decision A, 16 Nov 2026): STRICT on required training features (missing → `RuntimeError`) + LENIENT on extras (silently dropped, DEBUG-logged) + WARN on `LEADINPUT_SCHEMA_VERSION` drift (log + proceed on intersection).
- **RFM feature engineering** (Decision B, pure functions in `pinkmate/ml/feature_engineering.py`): recency (90-day linear decay) + frequency (log-scaled, saturates at 20 touches) + monetary (ACV band step function) + composite.
- **Clay-style ICP scoring layer** (Decision C, BOTH rules + composite): 5 rule-based binary features (employees-band × industry × seniority × country × non-free-mail) + composite 0-100 + A/B/C/D tier assignment.
- **MTA (Multi-Touch Attribution) formalism** (Decision D, MEDIUM depth): Section 5e of `docs/architecture.md` shows Q-MDP belief-state transitions implement a Markov chain isomorphic to the data-driven MTA model (Shao & Li 2011). Full formal proof deferred to Slice 7 probabilistic framework formalisation.
- **Test count:** 201 passing (179 Slice 5.1 + 17 feature_engineering + 5 schema contract).
<!-- APPEND this block to README.md at the end of the roadmap bullets section. -->

### Slice 5.3 — σ_calibration governance sibling (shipped <date> 2026)

- **σ_calibration** = calibration-stability companion to σ_SHAP attribution-stability. Both measured under the SAME K-rotation protocol (K=50 default, matching paper §V canonical).
- **Research question answered:** does σ_calibration correlate with σ_SHAP? Spearman correlation reported in `docs/sigma_calibration_report.md` with 95% bootstrap CI — either positive (cheap-proxy hypothesis supported), negative (opposing failure modes), or null (complementary signals; both must be reported).
- **Three properties enforced by tests** (`tests/test_calibration_variance.py`): (1) Bounded σ ∈ [0, 0.5]; (2) K-stable (σ at K=50 within noise of σ at K=10); (3) Reproducible (same seed → identical σ).
- **New module:** `pinkmate/ml/calibration_variance.py` — sibling to `attribution.py`, same API shape (`sigma_calibration` / `compute_rotated_calibration_variance` mirror `sigma_shap` / `compute_rotated_shap_variance`).
- **Measurement script:** `scripts/measure_sigma_calibration.py` — reuses Slice 5/5.1 trained model pickles across the 4 cross-domain validation datasets (Taiwan + LeadForge advanced/intermediate + X Education), no retraining required.
- **Test count:** 241 passing (215 Slice 5.2 baseline + 26 calibration variance tests).
<!-- APPEND this block to README.md at the end of the roadmap bullets section. -->

### Slice 6 — Survival analysis extension (shipped <date> 2026)

- **σ_SHAP transfer research question answered:** does attribution-stability transfer from classification (`predict_proba`) to survival (`predict_cumulative_hazard`)? Reported in `docs/survival_training_report.md` via Spearman rank correlation between Slice 5.1 classification σ_SHAP and new survival σ_SHAP, with 95% bootstrap CI.
- **6-model survival ensemble:** Cox PH (primary, semi-parametric) + 3 parametric AFT families (Weibull, LogNormal, LogLogistic) + XGBoost `survival:cox` + LightGBM AFT (log-time regression). Unified wrapper interface in `pinkmate/ml/survival.py` so the training harness treats all models identically regardless of backing library.
- **SurvSHAP attribution** via the Krzyziński et al. 2023 package, with a finite-difference cumulative-hazard fallback when `survshap` is unavailable. Mirrors `pinkmate/ml/attribution.py` API — `sigma_shap_hazard` + `compute_rotated_survshap_variance` match the sibling conventions established in Slice 5.3.
- **LeadForge → survival schema transform** (`pinkmate/ml/survival_data.py`) constructs `(time, event)` from LeadForge's `touch_count` + `days_since_last_touch` + `converted` columns via Decision B=3. Honesty bar surfaced in every training-report run: time construction is synthetic, verdict stability is enforced across touch-interval sensitivity sweep (5 / 7 / 14 days).
- **Bank of England credit-risk alignment:** Cox PH is the PRA SS1/23 model-risk-management convention for survival-style credit-risk modelling. Shipping σ_SHAP on both semi-parametric (Cox) and parametric (Weibull / LogN / LogL) families gives the RAEng Route 4 evidence pack a regulatory-adjacent methodology line.
- **Dependencies added to `requirements.txt`:** `lifelines>=0.27.0`, `scikit-survival>=0.22.0`, `survshap>=0.4.0`.
- **Test count:** 303 passing (241 Slice 5.3 baseline + 62 new survival tests across `test_survival.py` 24 + `test_survival_attribution.py` 15 + `test_survival_data.py` 23).
