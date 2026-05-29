PinkMate — Agentic GTM Engine (Technical Architecture Repository)
PinkMate is an early-stage research and engineering project exploring deterministic, multi-agent orchestration for B2B go-to-market workflows. It combines classical interpretable machine learning with structured agent-level routing to investigate how lead-scoring, segmentation, and outreach decisions can be made more reproducible, auditable, and transparent than conventional CRM automation or LLM-only agent loops.

Repository scope. This repository documents the technical architecture, agent stack, and reference skeletons for PinkMate's predictive GTM engine. It is a research-engineering artefact — the agent skeletons and orchestration scaffolding are public so the design can be inspected and built upon. Trained model artefacts, production data pipelines, and end-to-end runnable workflows are work in progress and not present in this repository in their current form.
License: Proprietary — source available for evaluation only. See LICENSE file for details.


1. Technical Overview
Core thesis. Conventional CRM automation stores leads and applies static rules. LLM-only agent systems generate fluent action sequences but exhibit non-deterministic execution, weak auditability, and unstable explainability. PinkMate explores a middle path: classical interpretable ML for scoring, bounded modular agents for orchestration, and explicit state transitions for reproducibility.
Design choices.

Multi-agent pipeline: Ingestion → Enrichment → Predictive Scoring → Segmentation → Outreach → Feedback loop, with each agent responsible for a bounded function and observable state transition.
Predictive ML (planned implementation): classical interpretable models — Naive Bayes, Random Forest, Gradient Boosting — chosen deliberately over opaque deep-learning alternatives to support feature attribution and governance review.
Agentic orchestration: LangChain / LangGraph-style routing between agents, with explicit state schemas and event-triggered transitions.
Memory and context (planned): vector-based memory for company, persona, and interaction history, enabling retrieval-augmented context for downstream agents.
Integration surface (planned): designed to interoperate with Clay, Apollo, PhantomBuster, HubSpot, and Salesforce.


2. Engineering Positioning
The intellectual direction of PinkMate connects to ongoing research into deterministic, governance-aware orchestration for agentic AI systems — a thesis pursued in parallel through academic peer-reviewed work co-authored with the University of Kent.
Specifically, the architectural choices here — bounded agent responsibilities, explicit state transitions, interpretable classical ML for scoring, and explainability-oriented design — anticipate integration with the Q-MDP (superposition-aware Markov Decision Process) formulation described in:

Sehgal, K. and Bhatti, K. N. (2026). Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure. IEEE Cyber-AI 2026 (under peer review).

This repository is the practical scaffolding that the broader research thesis will be instantiated against once the trained-model artefacts and end-to-end LangGraph workflows are built out.

3. Repository Structure
pinkmate-agentic-gtm/
│
├─ README.md                  # This file – high-level overview
├─ requirements.txt           # Python dependencies (placeholder)
│
├─ agents/                    # Individual agent skeletons in the GTM pipeline
│   ├─ __init__.py
│   ├─ ingestion_agent.py     # Ingest CSV / LinkedIn / Apollo data
│   ├─ enrichment_agent.py    # Enrich company + persona signals
│   ├─ scoring_agent.py       # Predictive ML scoring (skeleton; placeholder logic)
│   ├─ segmentation_agent.py  # Tiering, ICP buckets & readiness
│   └─ outreach_agent.py      # Next-best-action decisions
│
├─ workflows/                 # Orchestrated workflows and routing
│   └─ pipeline.yaml          # Declarative pipeline definition
│
└─ docs/                      # Additional technical documentation
    └─ architecture.md        # Detailed architecture notes
Status note. At this stage, many files are intentional skeletons. The scoring_agent.py in particular contains placeholder return values (clearly marked in code with TODO and # Placeholder comments) so the agent's interface contract can be inspected before the model-loading logic is finalised. The repository's purpose at this point is to document the engineering design and agentic architecture, not to provide a production-ready or benchmark-validated system.

4. Agent Stack
PinkMate's GTM engine is modelled as a set of independent but coordinated agents:

IngestionAgent

Accepts CSV uploads, LinkedIn exports, Apollo data, or direct API feeds.
Normalises and validates lead records.


EnrichmentAgent

Calls enrichment sources (Clay, Clearbit, Apollo, internal APIs).
Adds firmographics (industry, size, location, funding) and buyer signals.


ScoringAgent

Designed to apply classical ML models (Naive Bayes, Random Forest, Gradient Boosting) — model-loading logic and training artefacts are pending.
Will produce a probability of conversion and a confidence score.


SegmentationAgent

Converts scores into tiers (Tier 1 / Tier 2 / Nurture).
Applies ICP rules (sector, geography, ARR, role seniority, intent).


OutreachAgent

Decides next-best-action: email sequence, LinkedIn DM, nurture cadence.
Designed to plug into CRMs and outreach tools.


FeedbackLoop / Analytics

Will consume engagement + conversion signals.
Will adjust thresholds, weights, and features over time.



Orchestration is via a LangChain / LangGraph-style workflow enabling routing, error handling, and retries.

5. ScoringAgent — Reference Skeleton
The ScoringAgent class in agents/scoring_agent.py defines the interface contract for lead scoring: a feature dictionary in, a structured score-plus-breakdown response out.
Core responsibilities (planned implementation):

Load trained classical ML models from models/.
Accept a dictionary of engineered features.
Combine model outputs into a final score.
Return a structured response with final_score (0–100) and per-model model_scores.

The current implementation contains explicit placeholder return values (e.g. nb_score = 0.78  # simulated) so the interface contract is inspectable before the real model-loading logic is added. This is intentional — the file is a reference skeleton, not a deployed scoring service.

6. Roadmap
Planned next stages of development:

Real model artefacts — train Naive Bayes, Random Forest, and Gradient Boosting on a public B2B / lead-scoring dataset and commit the artefacts to models/.
End-to-end LangGraph workflow — replace the YAML pipeline scaffolding with an executable LangGraph state machine wiring the five agents.
Real benchmark harness — instrument latency, replay consistency, audit-trace size, and explainability stability against a baseline heuristic orchestration loop. Numbers will be reported only when measured.
SHAP integration — wire TreeSHAP-based attribution stability monitoring into the ScoringAgent for governance-oriented inspection.
Integration surface — connectors to Clay, PhantomBuster, Apollo, HubSpot, and Salesforce.


7. Related Research
The architectural and research thesis underlying PinkMate is developed in academic collaboration with Kent Business School, University of Kent. Related and ongoing peer-reviewed work:

Sehgal, K., & Bhatti, K. N. (2026). Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure. Submitted to IEEE Cyber-AI 2026 (under peer review; decision expected July 2026).
Sehgal, K., & Bhatti, K. N. (2026). Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints. arXiv:2605.14067 [cs.LG]. https://arxiv.org/abs/2605.14067
Sehgal, K., & Bhatti, K. N. (2026). Auditable Climate Risk Intelligence from Fragmented ESG Data: Deterministic Orchestration and Imbalance-Aware Learning for Scope 1–3 Validation. arXiv preprint (under moderation review).


8. Links and References

Product landing page: https://www.pink-mate.co.uk
Deployment mirror (fallback): https://v0-pink-mate.vercel.app/
Public concept page (Notion): https://iodized-lily-573.notion.site/Hero-Section-205a5e7da1508078b018e8babfa60c61

Contact / Founder
Karan Sehgal — Founder, PinkMate
AI-native GTM, predictive automation, and growth strategy
karansehgal@pink-mate.co.uk
