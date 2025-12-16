# PinkMate – Agentic GTM Engine (Technical MVP)

PinkMate is an AI-native, agentic GTM engine that transforms raw lead lists into scored, segmented and outreach-ready pipeline. It combines classical ML models with multi-agent orchestration (via LangChain / LangGraph) to automate enrichment, predictive lead scoring, ICP segmentation and agentic outreach workflows.

> This repository documents the technical architecture, agent stack and early implementation skeleton for PinkMate's predictive GTM engine.
> License: Proprietary — source available for evaluation only. See LICENSE file for details.


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

At this stage, many files are placeholders or skeletons. The emphasis is on showing the technical design and agentic architecture, rather than a production-ready system.

---

## 3. Agent Stack

PinkMate's GTM engine is modelled as a set of independent but coordinated agents:

- **IngestionAgent**
  - Accepts CSV uploads, LinkedIn exports, Apollo data or direct API feeds.
  - Normalises and validates lead records.

- **EnrichmentAgent**
  - Calls enrichment sources (Clay, Clearbit, Apollo, internal APIs).
  - Adds firmographics (industry, size, location, funding) and buyer signals.

- **ScoringAgent**
  - Applies classical ML models (Naive Bayes, Random Forest, Gradient Boosting).
  - Produces a probability of conversion and a confidence score.

- **SegmentationAgent**
  - Converts scores into tiers (e.g. Tier 1, Tier 2, Nurture).
  - Applies ICP rules (sector, geography, ARR, role seniority, intent).

- **OutreachAgent**
  - Decides next-best-action: email sequence, LinkedIn DM, nurture cadence.
  - Designed to plug into CRMs and outreach tools.

- **FeedbackLoop / Analytics**
  - Consumes engagement + conversion signals.
  - Adjusts thresholds, weights and model features over time.

The agents are orchestrated via a LangChain / LangGraph-style workflow, enabling routing, error handling and retries.

---

## 4. ScoringAgent – Example Skeleton

The ScoringAgent is responsible for taking engineered features for a lead and returning a probability-like score and breakdown.

See agents/scoring_agent.py for a simple class skeleton.

Core responsibilities:

- Load trained models (classical ML) from models/
- Accept a dictionary of features for a given lead
- Combine model outputs into a final score
- Return a structured response, including:
  - final_score (0–100)
  - model_scores (per-model scores)
  - optional explanation fields

This is intentionally kept as a minimal starting point for further extension.

---

## 5. Future Work (V1+ Roadmap)

Planned enhancements include:

- LLM-enhanced scoring layers
  - Use LLMs to interpret unstructured signals (job descriptions, posts, news).

- RAG-based context grounding
  - Use embeddings + vector search for account history and prior touchpoints.

- Autonomous outreach sequencing
  - Allow the OutreachAgent to autonomously schedule and adapt multi-step sequences.

- Deeper integrations
  - Clay, PhantomBuster, Apollo, HubSpot / Salesforce, custom webhooks.

- Experimentation & evaluation
  - Track uplift in pipeline quality and conversion vs. baseline GTM flows.

---

## 6. Links & References

Product landing page (primary):   https://www.pink-mate.co.uk
  Deployment mirror:   https://v0-pink-mate.vercel.app/ (fallback access if DNS propagation/caching delays access)

PinkMate public MVP (Notion – concept & UI):
https://iodized-lily-573.notion.site/Hero-Section-205a5e7da1508078b018e8babfa60c61?pvs=143

Contact / founder:  
Karan Sehgal – Founder, PinkMate  
AI-native GTM, predictive automation & growth strategy  
karansehgal@pink-mate.co.uk
