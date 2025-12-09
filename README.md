# PinkMate – Agentic GTM Engine (Technical MVP)

PinkMate is an AI-native, agentic GTM engine that transforms raw lead lists into scored, segmented and outreach-ready pipeline. It combines classical ML models with multi-agent orchestration (via LangChain / LangGraph) to automate enrichment, predictive lead scoring, ICP segmentation and agentic outreach workflows.

> This repository documents the technical architecture, agent stack and early implementation skeleton for PinkMate's predictive GTM engine.

---

## 1. Technical Overview

**Core idea:**  
Traditional CRMs store leads. PinkMate predicts, prioritises and executes using an AI-native GTM engine.

**Key capabilities:**

- Multi-agent pipeline:
  - Ingestion → Enrichment → Predictive Scoring → Segmentation → Outreach → Feedback loop
- Predictive ML scoring:
  - Naive Bayes, Random Forest, Gradient Boosting (classical ML)
- Agentic orchestration:
  - LangChain / LangGraph style routing between agents
- Memory & context:
  - Vector-based memory (embeddings) for company, persona and interaction history
- Automation-ready:
  - Designed to integrate later with Clay, Apollo, PhantomBuster, HubSpot / Salesforce

---

## 2. Repository Structure

```text
pinkmate-agentic-gtm/
│
├─ README.md                  # This file – high-level overview
├─ requirements.txt           # Python dependencies (placeholder)
│
├─ agents/                    # Individual agents in the GTM pipeline
│   ├─ __init__.py
│   ├─ ingestion_agent.py     # Ingest CSV / LinkedIn / Apollo data
│   ├─ enrichment_agent.py    # Enrich company + persona signals
│   ├─ scoring_agent.py       # Predictive ML scoring logic
│   ├─ segmentation_agent.py  # Tiering, ICP buckets & readiness
│   └─ outreach_agent.py      # Trigger email / LinkedIn sequences
│
├─ workflows/                 # Orchestrated workflows and routing
│   ├─ pipeline.yaml          # Declarative pipeline definition (MVP)
│   └─ routing_logic.py       # Agent routing / decision logic
│
├─ models/                    # Model artefacts & configs
│   ├─ naive_bayes_model.pkl  # Placeholder
│   ├─ random_forest.pkl      # Placeholder
│   └─ boosting_config.json   # Placeholder
│
└─ docs/                      # Additional technical documentation
    ├─ architecture.md        # Detailed architecture notes
    ├─ agent_stack.png        # Architecture diagram (to be added)
    └─ workflow_diagram.txt   # Text-based workflow description
