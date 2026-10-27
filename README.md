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
├─ pinkmate/                  # Q-MDP formal constructs (Slice 1)
│   ├─ __init__.py
│   ├─ qmdp.py                # BeliefState, SuperposedBelief, transition(),
│   │                         # governance_commit() — paper §III.A
│   └─ state.py               # Pydantic PipelineState, TraceEntry — paper §III.A τ_exec
│
├─ agents/                    # Orchestration-graph vertices (paper §III.B)
│   ├─ __init__.py
│   ├─ ingestion_agent.py
│   ├─ enrichment_agent.py
│   ├─ scoring_agent.py       # Emits SuperposedBelief; placeholder predictor
│   │                         # until Slice 2 wires the real six-model ensemble
│   ├─ segmentation_agent.py
│   └─ outreach_agent.py
│
├─ workflows/                 # Orchestrated workflows and routing
│   ├─ pipeline.py            # LangGraph StateGraph with governance-gated edge
│   │                         # (Slice 1 — authoritative orchestration layer)
│   └─ pipeline.yaml          # DEPRECATED; retained for commit-history continuity
│
├─ tests/                     # Reproducibility + determinism tests
│   ├─ test_qmdp.py           # Unit tests on belief / commit / transition
│   └─ test_pipeline.py       # Integration + deterministic-replay smoke tests
│
└─ docs/                      # Technical documentation
    └─ architecture.md        # Paper ↔ code map, slice roadmap
```

**Slice 1 status.** The Q-MDP formal layer (`pinkmate/qmdp.py`, `pinkmate/state.py`) and the LangGraph orchestration graph (`workflows/pipeline.py`) are **live** and executable end-to-end on a mock lead with `python -m workflows.pipeline`. The ScoringAgent predictor remains a deterministic placeholder in Slice 1 so the Q-MDP orchestration layer can be verified independently; the real six-model ensemble + SMOTE + TreeSHAP wiring lands in Slice 2.

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
- **Slice 2.** Real ML layer per paper §III.C–D and §V — six-model ensemble (LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost), SMOTE on training partition only, TreeSHAP attribution with σ_SHAP measurement harness, replacing the Slice 1 placeholder predictor in `ScoringAgent`.
- **Slice 3.** Durable checkpointing per paper §III.B replay-fidelity claim — `MemorySaver` → `SqliteSaver` → PostgreSQL checkpointer for cross-process audit-trail persistence.
- **Slice 4.** Empirical validation — latency benchmarks against the `T_fa = O(log n + k)` scheduling claim (§III.B), replay-fidelity harness, audit-trace-completeness metric aligned with the ESG companion deployment's measurement.
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
