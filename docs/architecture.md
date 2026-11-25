# PinkMate — Q-MDP Reference Deployment Architecture

PinkMate is the reference deployment of the **Q-MDP orchestration framework** described in:

> Sehgal, K., & Bhatti, K. N. (2026). *Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure.* Accepted at IEEE Cyber-AI 2026, Bucharest, Romania. Submission #4554.

The paper's final paragraph names this repository (`github.com/karansehgal7/pinkmate-agentic-gtm`) as the LangGraph-based reference deployment of the orchestration layer. This document maps every formal construct of the paper to the file that implements it in this repository.

---

## 1. Paper → Code Map

| Paper reference | Formal construct | Implementation |
|---|---|---|
| §III.A, eq. for `b_t(s)` | Belief state `b_t(s) = P(s_t = s \| o_{1:t}, a_{1:t})` | `pinkmate/qmdp.py` → `BeliefState` |
| §III.A, `B_t = {b_1, …, b_n}` | Superposed belief set | `pinkmate/qmdp.py` → `SuperposedBelief` |
| §III.A, transition dynamics | `P(s_{t+1}\|s_t, a_t) = Σ_i P(s_{t+1}\|s_{t,i})·P(a_t\|s_{t,i})` | `pinkmate/qmdp.py` → `transition()` |
| §III.A, execution trace | `τ_exec = [(s_0, a_0, r_0, s_1), …]` | `pinkmate/state.py` → `PipelineState.trace` (list of `TraceEntry`) |
| §III.A, commit predicate | `commit(i*) ← argmax_i P(s_t\|o_{1:t}, a_{1:t}) s.t. P > θ_gov` | `pinkmate/qmdp.py` → `governance_commit()` |
| §III.B, orchestration graph | `G = (V, E)` — asynchronous event-driven | `workflows/pipeline.py` → LangGraph `StateGraph` |
| §III.B, scheduling bound | `T_fa = O(log n + k)` | Enforced by LangGraph priority scheduler; **empirical validation in `pinkmate/benchmarks/latency.py`** (Slice 4), results in `docs/benchmarks_report.md` + `docs/figures/latency_*.png` |
| §III.B, replay fidelity | Durable τ_exec persistence across orchestration restarts | `pinkmate/persistence.py` → `make_checkpointer()` factory + `SqliteSaver`; **reusable consumer harness in `pinkmate/benchmarks/replay_fidelity.py`** (Slice 4); unit-level proof in `tests/test_checkpointing.py` |
| §III.C, SMOTE | `x_new = x_i + λ(x_nn − x_i)` | `pinkmate/ml/imbalance.py` → `apply_smote()` |
| §III.D, model zoo | LR / RF / AdaBoost / XGBoost / LightGBM / CatBoost | `pinkmate/ml/model_zoo.py` → `build_model_zoo()`, `train_zoo()` |
| §IV.B, risk-sensitive loss | `L_risk = αFN + βFP, α > β` | Metric scaffolding in `pinkmate/ml/pipeline.py::ModelMetrics`; full `L_risk` landing in Slice 3 |
| §V, attribution stability | `σ_SHAP = √((1/m) Σ_j Var_k \|φ_{j,k}\|)` | `pinkmate/ml/attribution.py` → `sigma_shap()`, `compute_rotated_shap_variance()` |
| §VI.A, reproducibility protocol | seed=42, 80/20 stratified, SMOTE on train only, TreeSHAP K=50 × 100 × 200 | `pinkmate/ml/pipeline.py` → `run_slice_two_pipeline()`; Taiwan loader `pinkmate/ml/dataset.py`; LeadForge loader `pinkmate/ml/leadforge_loader.py` |
| §VI.B / §VI.C, Table I / Table II | Per-model metrics + σ_SHAP comparison | `scripts/reproduce_table2.py` (Taiwan, magnitude tolerance); `scripts/train_gtm_ensemble.py` (LeadForge, qualitative-direction) — Slice 5 |

---

## 2. Orchestration graph (§III.B)

The graph is directed and acyclic:

```
START
  │
  ▼
ingestion ──► enrichment ──► scoring ──┬──► segmentation ──► outreach ──► END
                                       │                                   
                                       └──► defer ─────────────────────► END
                                            (governance predicate did not fire)
```

The edge from `scoring` to `segmentation` is **conditional** on the governance commit predicate (§III.A). When the predicate fires, execution proceeds through `segmentation` → `outreach`. When it does not fire, execution routes to the `defer` node, which records the uncommitted belief and governance decision into the trace and halts without emitting a downstream action — the **deferred-commitment semantics** of the paper.

---

## 3. Deployment domain

The paper is state-domain agnostic; PinkMate instantiates Q-MDP for the **B2B GTM lead-scoring domain**. Latent enterprise states correspond to lead qualification stages (currently `QUALIFIED` and `UNQUALIFIED` in Slice 1; extended to `{QUALIFIED, NURTURE, DISQUALIFIED, HOLDOUT}` in Slice 2). The governance threshold `θ_governance` gates progression from scoring to downstream outreach actions.

The paper's ESG climate-risk deployment ([Sehgal & Bhatti, 2026b, arXiv:2606.02604](https://arxiv.org/abs/2606.02604)) applies the same Q-MDP framework to a different regulated domain. PinkMate and the ESG deployment share the formal orchestration layer; only the state-domain instantiation differs.

---

## 4. Governance threshold

`θ_governance` is a configuration parameter. The paper references it symbolically without fixing a numeric value, because the appropriate threshold depends on deployment context (high-stakes financial decisions warrant tighter thresholds than low-stakes B2B outreach). PinkMate's default is **0.75** — committed action requires at least 75% posterior belief in the leading latent state. Operators may raise this at deployment time for stricter governance regimes.

---

## 5. Determinism contract (§III.B)

The paper's §III.B claims deterministic execution and reproducible replay. Operationally, PinkMate commits to:

- **State is typed.** `PipelineState` is a Pydantic model; no untyped dicts flow between nodes.
- **Trace is complete.** Every orchestration node appends a `TraceEntry` with `(step, node, action, state_before, state_after, metadata)`. The trace is the paper's `τ_exec`.
- **Replay is testable.** `tests/test_pipeline.py::TestDeterministicReplay` asserts that two independent runs on identical inputs produce identical final state (score, belief, governance decision, segment, next-best action, trace signature).
- **No wall-clock reads inside the graph.** All sequencing uses the integer `step` counter so a replayed trace is identical regardless of execution speed.
- **Durability (Slice 3).** The `SqliteSaver` checkpointer persists every post-node `PipelineState` snapshot to disk, keyed by `thread_id`, so the trace survives orchestration restarts. `tests/test_checkpointing.py::TestCrossProcessReplayFidelity` asserts that a fresh `SqliteSaver` instance pointed at the same on-disk file reconstructs the terminal state and the full execution trace τ_exec identical to the original run — the paper's cross-process replay contract operationalised.

---

## 5a. Durable checkpointing architecture (§III.B — Slice 3)

The checkpointer is a swappable backend behind one factory call:

```python
from pinkmate.persistence import make_checkpointer

# In-process, Slice 1 behaviour (non-durable, fast).
cp = make_checkpointer(backend="memory")

# On-disk, Slice 3 behaviour (durable, cross-process).
cp = make_checkpointer(backend="sqlite", path="checkpoints/pinkmate.sqlite")
```

**Lifecycle.** Each `make_checkpointer(backend="sqlite", ...)` call opens its own `sqlite3.Connection` and calls `SqliteSaver.setup()` to create the `checkpoints` and `writes` tables idempotently. The connection stays open for the lifetime of the compiled LangGraph app. For tests, `tmp_path`-scoped files give per-test isolation without an external database.

**Thread model.** `thread_id` is the audit-trail key. Distinct lead runs use distinct `thread_id` values; re-invoking the compiled graph with the same `thread_id` resumes from (or appends to) that thread's persisted history. Thread isolation is verified at the raw-SQL level in `TestThreadIsolation`.

**Serialisation.** `PipelineState` is a Pydantic v2 model and its nested `SuperposedBelief`, `GovernanceCommit`, and `TraceEntry` are either Pydantic models or `frozen=True` dataclass-like Pydantic models. LangGraph's built-in `JsonPlusSerializer` round-trips these cleanly through the SQLite blob columns.

**Later-slice extension.** PostgreSQL is a single call-site change: `make_checkpointer(backend="postgres", dsn=...)` with a `PostgresSaver` instantiated inside the factory. No other code moves. This is the pattern the Slice 3 factory was shaped to support.

---

## 5b. Empirical validation harnesses (§III.B — Slice 4)

Slice 4 ships two measurement harnesses under `pinkmate/benchmarks/`, both consumed by `scripts/run_benchmarks.py` which writes committed artefacts to `docs/`.

### Latency harness (`latency.py`)

**Measurement protocol.** The harness builds a parametric chain-topology `StateGraph` of size n using the real `PipelineState` schema, where each node is a near-no-op pass-through (appends one `TraceEntry`, returns). Total `app.invoke()` latency is measured with `time.perf_counter` at microsecond resolution across R runs per n (default R=10). Warm-up runs (default 2) are discarded.

**Why chain topology.** A chain has sequential dependencies, so total latency necessarily scales at least linearly in n — this is a known constraint of the topology, not a scheduler bug. The paper's `T_fa = O(log n + k)` bound governs **per-step scheduler cost**, not total execution time. The chain benchmark validates the bound by showing per-step overhead stays **bounded** as n grows (not by showing total latency being logarithmic, which would be impossible for a chain).

**Fitting.** Both linear (`y = a·n + b`) and log-linear (`y = a·log(n) + b`) models are fit via ordinary least-squares to the mean-latency series. R² is reported for both. Pure-Python regression — no SciPy dependency. The verdict categorises the result as `LINEAR CHAIN AS EXPECTED` / `LOG-LINEAR OBSERVED` / `AMBIGUOUS`.

**Scope note.** Parallel-dispatch (fan-out / DAG) topologies that admit concurrent super-step dispatch — where the log(n) factor would be directly observable in total latency — are a follow-up refinement. The chain benchmark + per-step overhead stability is sufficient to validate the paper's per-step bound.

### Replay-fidelity harness (`replay_fidelity.py`)

**Reusable consumer design.** The harness takes `(thread_id, sqlite_path, reference_state)` and does not re-run the original pipeline. It opens a FRESH `SqliteSaver` instance against the on-disk SQLite file (simulating a cold process start) via a configurable `graph_builder` callback, reconstructs the state history, and emits a structured `FidelityReport` with:

- **completeness**: fraction of expected super-step snapshots recovered;
- **trace_integrity**: boolean — does the recovered τ_exec match the reference at the `(step, node, action)` tuple level?
- **divergences**: list of `FieldDivergence(field_path, reference_value, recovered_value, delta)` for every PipelineState field that differs between reference and recovered.

**Why reusable.** The `graph_builder` parameter lets other Q-MDP deployments (the ESG companion [arXiv:2606.02604], future instantiations) call the harness against their own compiled graphs without a hard dependency on the PinkMate pipeline. The default builder is PinkMate's own; replacing it is a one-line change.

**Tolerance semantics.** `FidelityReport.passed(tolerance)` lets each deployment set its own acceptable numeric-delta threshold per field. Strict replay fidelity (tolerance=0.0) requires exact equality on all fields; a deployment may choose tolerance > 0 for drift-detection scenarios where floating-point reserialisation introduces negligible deltas.

### Reporting

`scripts/run_benchmarks.py` executes both harnesses and writes to the repo:

- `docs/benchmarks_report.md` — markdown report with measurements tables, R² values, verdict, embedded chart references
- `docs/benchmarks_report.json` — machine-readable payload for downstream tooling
- `docs/figures/latency_total_vs_n.png` — total latency curve with both fits overlaid
- `docs/figures/latency_per_step_overhead.png` — per-step overhead stability curve with marginal-per-node reference line
- `docs/figures/replay_fidelity_summary.png` — bar chart of per-thread completeness with pass/fail colouring

All five artefacts are **committed** to the repo so RAEng / academic / grant reviewers can inspect the empirical validation in-browser without cloning or installing dependencies.

### Running

```bash
python scripts/run_benchmarks.py              # Full run — n ∈ {5,10,25,50,100,250}, 10 runs/n (~5-10 min)
python scripts/run_benchmarks.py --fast       # Quick check — n ∈ {5,10,25,50}, 4 runs/n (~2 min)
python scripts/run_benchmarks.py --no-charts  # Markdown + JSON only
```

### Test philosophy

`tests/test_benchmarks.py` (20 tests) covers **harness correctness**: builders produce valid graphs, measurements have the expected structure, curve-fit returns sensible outputs on controlled synthetic data (y = 2x+1 ⇒ linear R² ≈ 1.0), fidelity harness detects both identical replays (completeness 1.0, zero divergence) and injected divergence (score mutation → `score.final_score` divergence flagged), runner script produces well-formed artefacts.

Empirical numerical outcomes (actual R² values on real runs, verdict distribution across runs) are **not** asserted — they wobble ±5-15% run-to-run under OS scheduler jitter and GC pauses; a numerical assertion would make the test suite flaky without reflecting a real regression. The quantitative assessment lives in the committed markdown report, which is deterministically regeneratable but intentionally not CI-asserted.

---

## 5c. ScoringAgent deployment-mode flag (Slice 3.5)

Slice 3.5 is **product-arc ergonomics, not a paper claim.** The `ScoringAgent` constructor gains a `model_source` parameter that selects between two deployment paths without changing any downstream scoring, belief-emission, or governance behaviour.

### Three deployment paths

| Path | Construction | Audience | Warnings |
|------|-------------|----------|----------|
| Research reproduction | `ScoringAgent(model_source="zoo")` (default) | RAEng reviewers, academic grant panels, anyone reproducing paper claims | **Yes, transparently** — the Taiwan-vs-GTM warning is the signal, not noise |
| Product demo (Slice 3.5) | `ScoringAgent(model_source="placeholder")` | Website screenshots, pitch decks, client demos, ClickUp tiles, prospects | **None** — clean output |
| Production (Slice 5+) | `ScoringAgent(model_source="zoo")` with GTM-trained models in `models/` | Real B2B scoring when labelled data is available | None — real predictions |

### Why the flag is decoupled from Slice 5

Slice 5 (data-gated GTM retrain) is the long-term fix: once the ensemble is trained on domain-matching labels, the warnings disappear organically in `"zoo"` mode and `"placeholder"` becomes effectively equivalent. Slice 3.5 is the near-term fix that unblocks product-arc work (website, ClickUp, integrations) **without waiting on the data-gated Slice 5**, because the product-demo surface shouldn't show warnings to prospects regardless of research timeline.

### Backward compatibility

The default is `"zoo"` — all existing callers (`workflows/pipeline.py`, every test, every script) continue to work unchanged. Slice 3.5 is purely additive.

### Equivalence contract

When `models/` is empty (as on a fresh checkout before `scripts/reproduce_table2.py` has been run), `"zoo"` mode falls back to the same Slice 1 deterministic placeholder predictor that `"placeholder"` mode uses directly. The two modes therefore produce **identical scores on identical input** in that state, pinned by `tests/test_scoring_agent_modes.py::TestZooAndPlaceholderEquivalence`.

---

## 5d. GTM-domain extension (Slice 5)

Slice 5 extends the Slice 2 six-model ensemble + σ_SHAP governance harness to the B2B GTM domain, trained on LeadForge Lead Scoring v1 (HuggingFace: `shaypal5/leadforge-lead-scoring-v1`, MIT licence, synthetic B2B SaaS procurement scenario). This is a **product-arc domain extension**, and like Slice 2's Taiwan-benchmark validation, it is a methodology-validation slice — NOT a performance claim about real B2B GTM leads.

### Three deployment domains, one framework

The Q-MDP framework (§III) is domain-agnostic. Slice 5 operationalises this by running the same ensemble + σ_SHAP harness on a second dataset:

| Domain | Dataset | Training script | Writes to | Purpose |
|--------|---------|-----------------|-----------|---------|
| Financial distress (paper benchmark) | Taiwan Bankruptcy UCI 572 (6819 rows, 95 features, 3.23% minority) | `scripts/reproduce_table2.py` | `models/taiwan/` | Reproduces paper Table I + Table II within tolerance; RAEng / academic review artefact |
| B2B GTM (product arc) | LeadForge Lead Scoring v1 (5000 leads, 51 encoded features, ~20% minority) | `scripts/train_gtm_ensemble.py` | `models/gtm/` | Methodology validation on B2B-shaped data; populates the ScoringAgent's default `models_dir` |
| ESG climate-risk (companion deployment) | Cross-deployment at SustainGRC / DCarbon Global | External | External | Paper §VI companion arc, not in this repo |

### Subdirectory structure (Slice 5 Option A)

Prior to Slice 5, `models/` held Taiwan models directly. Post-Slice-5:

```
models/
├─ taiwan/              # Reproduction artefact (Slice 2)
│   ├─ logistic_regression.pkl
│   ├─ ... (6 models)
│   ├─ metrics.json
│   └─ attribution.json
└─ gtm/                 # Product-arc ensemble (Slice 5)
    ├─ logistic_regression.pkl
    ├─ ... (6 models)
    ├─ metrics.json
    └─ attribution.json
```

Both subdirectories are `.gitignored` — researchers populate them locally by running the appropriate script.

**ScoringAgent default**: `models_dir="models/gtm"` (product-first positioning). Taiwan reproduction becomes explicit: `ScoringAgent(models_dir="models/taiwan")`.

### LeadForge feature schema

After `pinkmate/ml/leadforge_loader.py` preprocesses the raw dataset, the feature matrix has ~26 columns (before one-hot encoding) → ~51 columns (after). Explicitly excluded:

- **Identity**: `lead_id`, `account_id`, `contact_id` (foreign keys, not features)
- **Timing**: `lead_created_at` (simulation clock, would leak)
- **Leakage trap**: `total_touches_all` (documented by LeadForge as counting touches across the full 90-day horizon — must not be a training feature)
- **Target**: `converted_within_90_days` (becomes `y`)

Included feature families:

- **Firmographics** (one-hot): `industry`, `region`, `employee_band`, `estimated_revenue_band`, `process_maturity_band`
- **Contact** (one-hot): `role_function`, `seniority`, `buyer_role`
- **Lead meta** (one-hot): `lead_source`
- **Engagement** (numeric): `touch_count`, `inbound/outbound_touch_count`, `session_count`, `pricing_page_views`, `demo_page_views`, `total_session_duration_seconds`, `touches_days_0_7`, `touches_last_7_days`, `days_since_first_touch`, `days_since_last_touch`
- **Funnel** (numeric + boolean): `activity_count`, `opportunity_created`, `has_open_opportunity`, `opportunity_estimated_acv`, `expected_acv`

### LeadInput schema evolution

`pinkmate.state.LeadInput` gains three optional band fields in Slice 5:

- `employee_band: Optional[str]` — maps to LeadForge `employee_band` categorical
- `estimated_revenue_band: Optional[str]` — maps to LeadForge `estimated_revenue_band` categorical
- `process_maturity_band: Optional[str]` — maps to LeadForge `process_maturity_band` categorical

`funding_stage` is retained in the `LeadInput` API (backward-compat) but is **dropped from the Slice 5 training schema** because LeadForge does not include it. Rationale: training a model on a feature that's 100% missing adds zero information; synthesising a proxy from other bands would introduce a correlation that doesn't exist in reality (research-ethics violation). The field re-enters the training schema in Slice 5.1+ when a real B2B corpus with funding-stage labels becomes available.

### Qualitative-direction verdicts (per Phase 1 Critique 3)

Slice 2 reproduction uses magnitude-tolerance verdicts (σ_SHAP ±0.005, recall ±0.03 vs paper values). Slice 5 has **no paper-declared target values** for LeadForge, so uses **qualitative-direction verdicts** instead:

- **SMOTE direction on XGBoost**: Taiwan Slice 2 showed SMOTE INCREASES σ_SHAP. Expect direction to transfer (may have weaker magnitude on LeadForge's ~20% minority vs Taiwan's 3.23%).
- **Model stability ordering**: Taiwan Slice 2 showed LightGBM σ_SHAP < XGBoost σ_SHAP under SMOTE. Expect ordering to transfer (this is a model-architectural property, less dataset-dependent).

DRIFT verdicts on LeadForge are **legitimate scientific outcomes**, not test failures. They would be reported in `docs/gtm_training_report.md` and interpreted in context (likely due to milder imbalance reducing SMOTE's effect size).

### Honesty bar — methodology validation only

The σ_SHAP numbers reported in `docs/gtm_training_report.md` and the committed PNG charts under `docs/figures/gtm/` are the output of applying the Q-MDP σ_SHAP governance harness to B2B-shaped tabular data for the first time. **They DO NOT constitute performance claims about PinkMate on real B2B GTM leads.**

This matches Slice 2's framing — the paper validates the framework on Taiwan Bankruptcy (the declared benchmark); it never claims Taiwan-trained models should be used for B2B scoring. LeadForge plays the same methodology-validation role for the product arc. Real-data validation is deferred to:

- **Slice 5.1**: X Education secondary sanity check + LeadForge advanced tier
- **Later (data-gated)**: pink-mate.co.uk landing-page capture over 6-12 months, academic partnership via Kent Business School / Khawar network, or specific industry engagement

---

<!-- APPEND this block to docs/architecture.md. -->
<!-- Insert BEFORE the current "## 6. Slice roadmap" (which gets renamed to ## 7 as part of Step 4a in SLICE_6_INTEGRATION.md). -->
<!-- After insertion, the file layout is:
       ## 5a-5g (classification ensemble evidence line)
       ## 6. Survival analysis extension (Slice 6)      <- this new section
       <!-- APPEND this block to docs/architecture.md. -->
<!-- Insert BEFORE the current "## 7. Slice roadmap" (which gets renamed to ## 8 as part of Step 4a). -->
<!-- After insertion, the file layout is:
       ## 5a-5g (classification ensemble evidence line)
       ## 6. Survival analysis extension (Slice 6)
       ## 7. Probabilistic framework formalisation (Slice 7)   <- this new section
       ## 8. Slice roadmap                                     (was ## 7)
       ## 9. Running the pipeline                              (was ## 8)
       ## 10. Citing                                           (was ## 9)
-->

---

## 5e. Slice 5.1 — Cross-domain validation evidence base

Shipped 16 November 2026 (commits `5cde402` + `a3d3606`).

### 4-point validation table

| Dataset | Minority rate | Domain | Source | smote_direction | model_stability_ordering |
|---|---|---|---|---|---|
| Taiwan Bankruptcy | 3.23% | Financial distress | Real (UCI) | PASS (paper §VI.C) | PASS (paper §VI.C) |
| LeadForge advanced | 7.86% | B2B GTM synthetic (harder) | Synthetic (HF) | PASS (Δ+0.0155) | PASS (LGB 0.0200 < XGB 0.0251) |
| LeadForge intermediate | 20.98% | B2B GTM synthetic | Synthetic (HF) | PASS (Δ+0.0108) | PASS (LGB 0.0147 < XGB 0.0189) |
| X Education | 38.54% | EdTech lead scoring | **REAL** (Zenodo CC BY 4.0) | PASS (Δ+0.0004) | PASS (LGB 0.0098 < XGB 0.0115) |

### Monotone SMOTE-Δ decrease trend

The SMOTE-direction effect size on XGBoost σ_SHAP **monotonically decreases**
as class balance approaches 50%: Δ=+0.0155 at 7.86% minority → +0.0108 at
20.98% → +0.0004 at 38.54%. This is exactly what the methodological
definition predicts — SMOTE has less synthetic-positive headroom to
destabilise attribution as classes near balance. The nuance is empirically
visible across 4 validation points, materially strengthening the paper's
cross-domain generalisability claim.

Full canonical verdicts record: [`SLICE_5_1_VERDICTS.md`](../SLICE_5_1_VERDICTS.md).

### Honesty bar

The σ_SHAP numbers and PNG charts committed under `docs/figures/xeducation/`
and `docs/gtm_advanced/figures/gtm/` are the output of applying the Q-MDP
σ_SHAP governance harness to real EdTech B2B lead-scoring data and to
harder-conditions synthetic B2B GTM data. They DO NOT constitute
performance claims about PinkMate on real target-industry B2B GTM leads.
Real-data validation on a target-industry corpus remains deferred to
later data-gated work (pink-mate.co.uk landing-page capture over 6-12
months, academic partnership via Kent Business School / Khawar network,
specific industry engagement).

---

## 5f. Slice 5.2 — Deployment-schema alignment + MTA formalism

Shipped <date> 2026.

### 5f.1 — ScoringAgent hybrid fail-fast schema contract

**Decision A** (16 Nov 2026) locked a hybrid contract between training-time
schema and deployment-time `LeadInput`:

- **STRICT on required features** — if a feature the model was trained on
  is missing from the deployment feature dict, `_predict_proba_real()`
  raises `RuntimeError` with a message pointing at EnrichmentAgent
  population or a retrain.
- **LENIENT on extras** — features present in `LeadInput` but not in the
  training schema are silently dropped (DEBUG-logged only).
- **DRIFT WARNING** — if `LEADINPUT_SCHEMA_VERSION` in `pinkmate/state.py`
  has moved on since the model was trained, a `WARNING` is logged and
  the drift is recorded for audit. Prediction still proceeds with the
  intersection of features (no hard failure on version mismatch — the
  model's existing feature set is still usable).

This replaces the Slice 2 silent zero-fill behaviour that was identified
as a deployment risk during Slice 5 integration. Zero-fill is a quiet
failure mode: the model runs on a feature vector it wasn't trained on,
producing nominally valid but semantically wrong predictions. The hybrid
contract fails loudly on real misalignment while remaining forgiving
toward forward-compatible schema evolution.

### 5f.2 — RFM feature-engineering layer

**Decision B** (16 Nov 2026) locked pure-function implementation in
`pinkmate/ml/feature_engineering.py` — minimal machinery, maximum
testability, no configuration object required.

| Score | Source LeadInput field | Function | Range |
|---|---|---|---|
| Recency | `days_since_last_touch` | Linear decay | 1.0 at 0 days → 0.0 at 90 days |
| Frequency | `touch_count` | `log1p(count) / log1p(20)` | 0.0 at 0 touches → 1.0 at ≥20 touches |
| Monetary | `opportunity_estimated_acv` | Step function over USD bands | 0.1 (<10k) → 0.3 (10-50k) → 0.5 (50-200k) → 0.75 (200k-1M) → 1.0 (>1M) |
| Composite | derived | Weighted average (default equal thirds) | [0, 1] |

All functions return 0.0 on `None` or invalid input. Weights on the
composite are configurable per-call; defaults are symmetric.

### 5f.3 — Clay-style ICP scoring layer

**Decision C** (16 Nov 2026) locked **BOTH rule-based binary features
AND composite score + tier** — interpretability wins, model
regularisation handles whichever is less useful at inference time.

Five rules (from `AazanM/clay-hubspot-enrichment` reference, Phase 2
research finding):

| Rule | Points | Match criterion |
|---|---|---|
| Employees band match | 30 | `employee_band` ∈ {"50-199", "200-999"} |
| Industry match | 25 | `industry.lower()` ∈ {"manufacturing", "logistics", "professional_services"} |
| Seniority match | 25 | `seniority.lower()` ∈ {"director", "vp", "c_suite"} |
| Country match | 10 | `country` ∈ {"US", "UK"} |
| Non-free-mail | 10 | Email domain NOT in {"gmail.com", "yahoo.com", ...} |

Composite = sum of matched rule-points (0-100). Tier mapping: ≥80 → A,
≥60 → B, ≥40 → C, <40 → D.

**Known limitations** (documented in `pinkmate/ml/feature_engineering.py`
module docstring, not blocking):

- Western-SaaS-biased defaults for employee bands, industries, countries
- Free-mail domain list lacks APAC providers (rediffmail.com, qq.com,
  163.com, naver.com)
- Monetary band step function; smooth log-linear alternative is a
  straightforward swap

Parameterise `ICP_TARGET_*` + `ICP_FREE_MAIL_DOMAINS` per deployment ICP.

### 5f.4 — MTA (Multi-Touch Attribution) formalism

**Decision D** (16 Nov 2026) locked **MEDIUM depth** — section with
notation, no full proof. Full proof of equivalence deferred to Slice 7
probabilistic framework formalisation (POMDP + Bayesian + mixture + HMM
bundled slice).

#### Isomorphism statement

Q-MDP's belief-state transitions over the discrete state space
S = {MQL, SQL, Closed_Won} implement a Markov chain that is **isomorphic
to the data-driven Multi-Touch Attribution (MTA) model** (Shao & Li,
2011).

#### Notation

Let `s_t ∈ S` be the lead's latent qualification state at step `t`,
`a_t` the action taken by the pipeline (score / govern / outreach), and
`o_t` the observation vector (touches, engagement signals derived from
EnrichmentAgent + RFM + ICP features).

The Q-MDP belief update (§III.A):

```
b_{t+1}(s') = η · P(o_{t+1} | s', a_t) · Σ_{s ∈ S} P(s' | s, a_t) · b_t(s)
```

is isomorphic to the MTA attribution-weight update on the same state
space S when `P(s' | s, a_t)` is the data-driven transition matrix
estimated from touch sequences in the corpus.

The composite `governance_commit(belief, θ)` rule (paper §III.A)
formalises a commit threshold on MTA-derived state posteriors —
equivalent to the standard MTA "credit allocation" decision at the
final state Closed_Won.

**Formal proof of equivalence is deferred to Slice 7.** This section
establishes the isomorphism statement for the current build so
downstream work can build on the MTA framing; Slice 7 provides the full
POMDP formalisation including the measurement model
`P(o_t | s_t, a_{t-1})` and the resulting belief-update derivation.

#### Reference

Shao, X., & Li, L. (2011). *Data-driven multi-touch attribution models.*
In Proceedings of the 17th ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD '11), pp. 258–264.

### 5f.5 — LeadInput schema version tagging

`pinkmate/state.py` now exports `LEADINPUT_SCHEMA_VERSION =
"slice52_rfm_icp_v1"`. The ScoringAgent compares this against the version
persisted in `models/gtm/attribution.json`'s `leadinput_schema_version`
key when a model was trained. Mismatch → WARNING, not failure.

This versioning policy will be extended by Slice 6 (survival-time fields
bump to `slice6_survival_v1`), Slice 7 (probabilistic framework fields
bump to `slice7_probabilistic_v1`), and so on. Downstream tooling can
detect version drift and recommend retrains without blocking inference.
## 5g. Slice 5.3 — σ_calibration governance sibling

Shipped <date> 2026.

### 5g.1 Motivation

Paper §V defines σ_SHAP as an **attribution-stability** signal: the standard deviation
of per-feature mean-absolute SHAP values across K rotated SHAP backgrounds. A low
σ_SHAP means audit cycles produce consistent attribution rankings; a high σ_SHAP
flags a model whose attribution flips under perturbation and therefore cannot
support a governance commit.

σ_calibration is the **calibration-stability** companion: the standard deviation
of a calibration metric (ECE or Brier) across K bootstrap rotations of the test
set. A low σ_calibration means predicted probabilities remain reliable under
audit perturbation; a high σ_calibration flags a model whose probability outputs
drift between audit cycles even if its attribution rankings are stable.

### 5g.2 Definition

σ_calibration (two parallel metrics, both reported):

```
σ_ECE   = sqrt( Var_k( ECE_k ) )
σ_Brier = sqrt( Var_k( Brier_k ) )
```

where ECE_k and Brier_k are computed on the k-th bootstrap resample of the test
set, k ∈ {1, …, K}.

Operational protocol (matches paper §V σ_SHAP for direct comparability):

- K = 50 rotations (paper's canonical choice)
- Seeded bootstrap resampling of the full test set, with replacement
- ECE binning: 10 equal-width bins on [0, 1]
- Reproducible: identical seed produces identical σ

### 5g.3 Three required properties (test-enforced)

1. **Bounded**: σ_ECE ∈ [0, 0.5] and σ_Brier ∈ [0, 0.5]. Theoretical cap derived
   from max sample std of values in [0, 1].
2. **K-stable**: σ at K=50 close to σ at K=10 on well-behaved synthetic data.
   K-dependence beyond bootstrap noise would indicate pathological rotation
   coupling.
3. **Reproducible**: same seed → identical σ, per-rotation trajectories, and
   mean metrics.

Enforced by `tests/test_calibration_variance.py` (26 tests total).

### 5g.4 Research question and correlation with σ_SHAP

> Does σ_calibration correlate with σ_SHAP?

Spearman rank correlation + 95% bootstrap CI reported in
`docs/sigma_calibration_report.md` across all models × all 4 cross-domain
validation datasets (Taiwan + LeadForge advanced + LeadForge intermediate +
X Education).

Three possible verdicts:

- **Positive correlation** (CI > 0): σ_calibration is a cheap proxy for σ_SHAP.
  Useful when SHAP computation is prohibitive; governance commit can be gated
  on σ_calibration alone.
- **Negative correlation** (CI < 0): attribution stability and calibration
  stability measure opposing failure modes. Both must be reported; a model
  that is attribution-stable but calibration-unstable (or vice versa) exists.
- **No correlation** (CI contains 0): σ_calibration and σ_SHAP are
  complementary governance signals. Both must be reported; governance commit
  requires both to be below their respective thresholds.

Because the correlation is bootstrapped from a modest sample (~16-24 model
× dataset pairs), the verdict is directional evidence rather than a definitive
methodological claim. A large-N replication is on the Slice 8+ queue.

### 5g.5 API shape (sibling to attribution.py)

```python
from pinkmate.ml.calibration_variance import (
    compute_rotated_calibration_variance,
    sigma_calibration,
    sweep_k_stability,
)

result = compute_rotated_calibration_variance(
    X_test=X_test,
    y_test=y_test,
    predict_proba_fn=model.predict_proba,
    k_rotations=50,
    seed=42,
)
# result.sigma_ece, result.sigma_brier, result.per_rotation_ece, ...
```

The signatures mirror `pinkmate.ml.attribution.compute_rotated_shap_variance`
and `pinkmate.ml.attribution.sigma_shap` to make the sibling-structure explicit
to anyone reading both modules.

### 5g.6 What Slice 5.3 does NOT claim

σ_calibration is a stability signal, not a calibration-quality claim. A model
can be well-calibrated (low mean_ECE) and calibration-unstable (high σ_ECE);
conversely a model can be mis-calibrated and stable. Both mean and σ are
reported in `sigma_calibration_report.md` so the distinction is explicit.

Correlation with σ_SHAP is reported as evidence, not as a prescription that
either metric should replace the other. Governance commit in `ScoringAgent`
remains gated on the full (σ_SHAP, σ_calibration, θ_governance) triple.
## 6. Survival analysis extension (Slice 6)

Shipped <date> 2026.

### 6.1 Why a top-level section rather than §5h

Slices 5a-5g are operational and evidence extensions of the SAME
classification ensemble + governance contract: same `predict_proba` target,
same binary cross-entropy loss, same attribution structure (class-probability
gradient magnitudes). Slice 6 introduces:

- A different prediction target: `predict_cumulative_hazard(X, t)` and
  `predict_survival_function(X, t)`, both indexed by time.
- A different loss family: Cox partial likelihood (semi-parametric),
  Weibull / LogNormal / LogLogistic parametric AFT, XGBoost
  `survival:cox` custom objective, LightGBM AFT-style log-time regression.
- A different attribution structure: SurvSHAP on hazard gradients at a
  fixed time grid, aggregated over K rotated backgrounds.
- A different regulatory body of standards: Bank of England PRA SS1/23 and
  FS2/24 model-risk-management guidance explicitly reference survival
  analysis as a credit-risk-adjacent methodology.

For these reasons Slice 6 is reported as a new top-level §6 rather than
appended under §5. Future methodological jumps (Slice 7's POMDP
formalization, Slice 7.5's conformal prediction overlay) will follow
the same pattern.

### 6.2 Six-model survival ensemble (Decision A=3)

Unified `_BaseSurvivalWrapper` interface over four libraries:

| Model key         | Library       | Family                                     | Role                           |
|-------------------|---------------|--------------------------------------------|--------------------------------|
| `cox_ph`          | lifelines     | Semi-parametric (Cox partial likelihood)   | PRIMARY. BoE/PRA convention.   |
| `weibull_aft`     | lifelines     | Parametric AFT, monotone hazard            | Reliability engineering std    |
| `lognormal_aft`   | lifelines     | Parametric AFT, non-monotone hazard        | Pharma survival std            |
| `loglogistic_aft` | lifelines     | Parametric AFT, non-monotone hazard        | Mortality modelling            |
| `xgb_survival`    | XGBoost       | Gradient-boosted Cox (`survival:cox`)      | Ensemble member, non-linear    |
| `lgb_aft`         | LightGBM      | Log-time regression (AFT-style)            | Ensemble member, non-linear    |

ExponentialAFT with covariates is NOT included because lifelines >=0.30
does not ship `ExponentialAFTFitter` (only univariate `ExponentialFitter`
and `PiecewiseExponentialRegressionFitter` exist). WeibullAFT's shape
parameter approaching 1 is the AFT equivalent — the training report
surfaces this shape parameter explicitly for diagnostic purposes.

Every wrapper exposes the same interface:

```python
fit(X, time, event) -> self
predict_survival_function(X, times) -> ndarray (n, len(times))
predict_cumulative_hazard(X, times) -> ndarray (n, len(times))
predict_median_survival_time(X) -> ndarray (n,)
concordance_index(X, time, event) -> float
log_likelihood() -> float
aic() -> float
```

### 6.3 Time + event construction on LeadForge (Decision B=3)

LeadForge has `touch_count`, `days_since_last_touch`, and binary
`converted` columns but no native `days_since_first_touch`. The
`transform_leadforge_to_survival` function constructs the survival schema
via:

```
days_since_first_touch = touch_count * touch_interval_days
                         + days_since_last_touch
time  = min(days_since_first_touch, max_followup_days)
event = (converted == 1) AND (time <= max_followup_days)
```

Defaults: `max_followup_days = 90` (matches LeadForge 90-day sales-cycle
convention), `touch_interval_days = 7.0` (weekly cadence assumption).

Honesty bar baked into every training run: a sensitivity sweep over
`touch_interval_days ∈ {5.0, 7.0, 14.0}` is reported, and the σ_SHAP
transfer verdict is only considered valid if it is stable across that range.

### 6.4 SurvSHAP attribution (Decision C=1)

Primary backend: Krzyziński, Spytek, Baniecki, Biecek (2023)
`survshap` package. Falls back to finite-difference cumulative-hazard
attribution when `survshap` cannot import or does not support a given
model wrapper (XGBoost survival:cox and LightGBM AFT wrappers use the
fallback by default because they expose non-standard prediction
interfaces).

σ_SHAP on hazard outputs uses the SAME paper §V rotated-background
protocol as classification σ_SHAP and Slice 5.3 σ_calibration:

```
sigma_SHAP_hazard = sqrt( (1/m) * sum_j Var_k( |phi_{j,k,hazard}| ) )
```

where φ_{j,k,hazard} is the mean-abs SurvSHAP attribution for feature j
on the reference set under background k, averaged over the time grid
`(30, 60, 90)` days (monthly points inside the 90-day censoring window).

### 6.5 Three research questions (Slice 6 headline)

1. **Does σ_SHAP transfer from classification to survival?**
   Spearman rank correlation + 95% bootstrap CI between classification
   σ_SHAP (loaded from Slice 5.1 `attribution.json` files) and survival
   σ_SHAP across matched model family pairs:
   `xgboost -> xgb_survival`, `lightgbm -> lgb_aft`,
   `random_forest -> cox_ph`, etc. Positive CI: transfer works
   (attribution-stability is a model-level property, not target-specific).
   Null CI: complementary signals, report both. Negative CI: opposing
   behaviours, flag for investigation.

2. **Is Cox PH σ_SHAP lower than Weibull σ_SHAP?**
   Direct comparison in the per-model σ_SHAP table. Hypothesis:
   semi-parametric Cox (no distributional assumption on baseline hazard)
   should be more attribution-stable than parametric Weibull (commits to
   a shape parameter). Verdict reported honestly whichever way it lands.

3. **Which parametric distribution fits LeadForge best?**
   AIC ranking + log-likelihood comparison over `{weibull_aft,
   lognormal_aft, loglogistic_aft}`. Best-fitting family informs the
   choice of parametric baseline if Cox PH itself is unavailable in a
   deployment setting.

### 6.6 API shape (sibling to attribution.py and calibration_variance.py)

```python
from pinkmate.ml.survival import build_survival_model, fit_model
from pinkmate.ml.survival_data import transform_leadforge_to_survival
from pinkmate.ml.survival_attribution import compute_rotated_survshap_variance

sf = transform_leadforge_to_survival(leadforge_df)
model = build_survival_model("cox_ph").fit(sf.X, sf.time, sf.event)
attr = compute_rotated_survshap_variance(
    model=model,
    model_key="cox_ph",
    X_train=sf.X,
    X_reference=sf.X.iloc[:200],
    feature_names=sf.feature_names,
    k_backgrounds=50,
)
# attr.sigma_shap_hazard, attr.per_feature_variance, attr.time_grid, ...
```

Signatures mirror `pinkmate.ml.attribution.compute_rotated_shap_variance`
and `pinkmate.ml.calibration_variance.compute_rotated_calibration_variance`
to make the sibling structure explicit across modules.

### 6.7 What Slice 6 does NOT claim

- σ_SHAP transfer is a stability claim, not a prediction-quality claim.
  A model can have stable attribution and still be poorly discriminative.
  Concordance index, log-likelihood, and AIC are reported alongside σ_SHAP
  for the full picture.
- The LeadForge survival analysis is on SYNTHETIC time construction
  (Decision B=3). Real-time-to-conversion data is deferred to Slice 8+
  pending an external dataset (pink-mate.co.uk landing-page capture OR
  academic partnership).
- Bank of England PRA SS1/23 alignment is methodological signalling, not
  a certification claim. PinkMate is not a regulated model; the Cox PH
  primary choice establishes vocabulary compatibility for future BoE /
  PRA / FCA agentic-AI evidence-pack conversations.


<!-- APPEND this block to docs/architecture.md. -->
<!-- Insert BEFORE the current "## 7. Slice roadmap" (which gets renamed to ## 8 as part of Step 4a). -->
<!-- After insertion, the file layout is:
       ## 5a-5g (classification ensemble evidence line)
       ## 6. Survival analysis extension (Slice 6)
       ## 7. Probabilistic framework formalisation (Slice 7)   <- this new section
       ## 8. Slice roadmap                                     (was ## 7)
## 7. Probabilistic framework formalisation (Slice 7)

Shipped 18 November 2026.

### 7.1 Why a top-level section (not sub-slice under Section 5 or 6)

Slice 7 is a **theoretical scaffolding slice**, not evidence or survival
extension. It formalises the mathematical objects that Slices 1-6
operationalised and makes them cite-able in the paper's Theoretical
Foundations rewrite. Per the methodological-jump pattern established for
Slice 6: fresh theoretical scaffolding → new top-level section. Existing
§7 / §8 / §9 renumber to §8 / §9 / §10 to make room.

### 7.2 POMDP formal mapping (Littman, Cassandra & Kaelbling 1995)

`pinkmate/qmdp_formal.py` provides the canonical POMDP tuple
`(S, A, T, O, R, γ)`, the Bayesian belief filter (predictive step + 
observation step + normalisation), and the Q-MDP commit rule
`π_QMDP(b) = argmax_a Σ_s b(s) · Q_MDP(s, a)`.

The module proves two isomorphisms via `tests/test_qmdp_formal.py`:

1. `pinkmate/qmdp.py::SuperposedBelief` (dict representation) IS a POMDP
   belief distribution (ndarray representation). The adapter
   `pinkmate_superposed_belief_as_pomdp_belief()` is a shape conversion,
   not a mathematical change.
2. `SuperposedBelief.governance_commit(θ)` IS a degenerate Q-MDP commit
   over `A' = {commit, defer}` where `Q_MDP(s, commit) âˆ ðŸ™[max_s b(s) ≥ θ]`.

This isomorphism is the formal justification for citing Littman et al. 1995
in the paper's Theoretical Foundations section.

### 7.3 Hidden Markov Model (Rabiner 1989)

`pinkmate/ml/hmm_belief.py` wraps `hmmlearn.CategoricalHMM` with a
PinkMate-flavoured API: latent states `{MQL, SQL, ClosedWon}`, discretised
touch-count observations, Viterbi decoding, forward-backward smoothing,
Baum-Welch re-estimation.

An HMM is a POMDP with `|A| = 1` and `R ≡ 0` — the agent has no choice,
only observes the latent-state trajectory. This reduction is useful for
batch post-hoc analytics on historical touch sequences, complementing the
live governance loop (which does have actions and uses the full POMDP).

Default parameters encode PinkMate's prior: slow left-to-right transition
(`p_self = 0.7`, `p_forward = 0.3`) with absorbing final state;
emission matrix biases higher states to emit more touches on average.
Overridable via Baum-Welch EM re-estimation (`LatentStateHMM.fit()`) once
real touch-sequence data accumulates.

### 7.4 Bayesian Weibull survival (Gelman et al. 2013)

`pinkmate/ml/bayesian_survival.py` adds posterior inference on the Slice 6
Weibull AFT parameters via PyMC 5.x+ NUTS sampling. Priors:
`shape ~ HalfNormal(2.0)`, `scale ~ HalfNormal(2 · max t)`. Likelihood
hand-rolled with right-censoring: `log pdf` for events, `log S(t)` for
censored observations, summed via `pm.Potential`.

Primary output: posterior-predictive 95% credible intervals on
`S(t) = P(T > t)` at caller-specified times. This is **stronger than
Slice 6's point-estimate Weibull AFT** because it quantifies parameter
uncertainty — for regulatory evidence lines, "P(convert by day 30) is
0.42 with 95% CI [0.35, 0.49]" is more defensible than a single point.

Diagnostics: R-hat (Gelman-Rubin) and ESS (Effective Sample Size) via
arviz, surfaced on the result dataclass.

### 7.5 Weibull mixture for latent buyer regimes (McLachlan & Peel 2000)

`pinkmate/ml/mixture_survival.py` fits a `K`-component Weibull mixture
to right-censored survival data via the EM algorithm. Each component
corresponds to a latent buyer regime — typically a fast-converter regime
(low scale, high shape) and a slow-converter regime (high scale, moderate
shape) at K=2.

E-step uses responsibilities with right-censoring contributions. M-step
uses weighted MLE via Nelder-Mead (scipy.optimize) on the weighted
log-likelihood. Convergence: `|ΔL| < 10^{-5}` or 200 iterations.

**Decision C (17 Nov 2026):** K selected by BIC sweep across K ∈ {2, 3, 4}
via `select_k_by_bic`. Not committed to K=2 upfront (avoids overfitting
the "fast vs slow converters" narrative); not non-parametric (Dirichlet
Process would be Slice 8+ complexity).

### 7.6 Four locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| A: Bayesian engine | PyMC ≥5.0 (not NumPyro) | Maturity + Windows compatibility; NumPyro migration can be Slice 8+ if speed becomes bottleneck |
| B: MCMC sampler | NUTS (PyMC default) | Standard in Bayesian survival; R-hat + ESS diagnostics well-understood; Variational ADVI sacrifices exactness we don't need |
| C: Mixture K | BIC sweep K ∈ {2, 3, 4} | Not upfront commitment to K=2; not Dirichlet Process overkill |
| D: HMM library | hmmlearn ≥0.3, 3 states | hmmlearn maturity + sklearn-style API; 3 states matches paper's belief-state terminology |

### 7.7 What Slice 7 does NOT do

- **Does not change the production governance path.** `ScoringAgent` and
  `workflows/pipeline.py` continue to use the Slice 1 `pinkmate/qmdp.py`
  primitives. Slice 7 is formal grounding, not a replacement.
- **Does not retrain Slice 6 survival models.** Bayesian Weibull and
  Weibull mixture are additional estimators alongside the Slice 6
  ensemble, not substitutes.
- **Does not rewrite the paper's Section III.A.** That's queued as part of
  the coordinated paper revisions pass. The module code and
  `docs/probabilistic_framework.md` are the SOURCE MATERIAL for the paper
  rewrite; the rewrite itself is a separate deliverable.

For the one-stop theoretical reference with full math derivations, see
`docs/probabilistic_framework.md`.


## 8. Production hardening + PRA SS1/23 alignment (Slice 7.5)

Shipped 24 November 2026. PinkMate is the GTM reference deployment of the
Q-MDP framework (Sehgal & Bhatti 2026). This slice brings the deployment
to Bank of England Prudential Regulation Authority **SS1/23 Principles 3
and 4** compliance via three complementary production-hardening layers:

* **Conformal prediction coverage guarantees** (§8.1)
* **Drift sentinels** (§8.2)
* **Fairness diagnostics** (§8.3)

Taken together with σ_SHAP (Slice 2) and σ_calibration (Slice 5.3), these
signals satisfy the Financial Conduct Authority's **FS2/24 agentic AI
explainability** requirements for the measured-attribution axis. §8.6
maps each component to the specific regulatory principle it answers.

### 8.1 Conformal prediction overlay on θ_governance

The Slice 1 governance primitive is

    commit_point(X)       :=    point_score(X) >= theta

which fires whenever the ensemble's mean class-1 probability crosses a
scalar threshold. This is sufficient for internal demos but it is NOT
sufficient under PRA SS1/23 Principle 3 (conservatism in model use for
credit, risk, and decisioning), because a 0.76 point score with a wide
underlying posterior is operationally indistinguishable from a 0.76
point score with a narrow one.

Slice 7.5a installs a strictly stricter conservative commit rule:

    commit_conservative(X) :=    lower_bound(X) >= theta

where `lower_bound(X)` is the lower end of a split-conformal prediction
interval at miscoverage rate α (default α = 0.10, giving 90% empirical
coverage on the calibration set).

**Construction.** Given a trained ensemble of classifiers, an adapter
(`_EnsembleProbabilityRegressor`) exposes the ensemble's mean
`predict_proba(X)[:, 1]` as a `sklearn.base.RegressorMixin`. The adapter
is wrapped in MAPIE's `SplitConformalRegressor` (version ≥ 1.0.0, uses
the `confidence_level = 1 - α` API introduced in MAPIE 1.0) with
`prefit=True` and conformity score `"absolute"`. Calibration is performed
on the held-out 20% test split from training-time `preprocess()`,
persisted to `models/<dataset>/calibration.npz` alongside `attribution.json`.
On inference, `predict_interval(X)` returns both a point estimate and a
`(lower, upper)` pair clipped to the probability simplex [0, 1].

**Coverage guarantee.** Under exchangeability of calibration and
production samples, conformal prediction gives the frequentist
guarantee

    P( true P(Y=1|X) ∈ [lower_bound(X), upper_bound(X)] )  >=  1 - α

marginalising over the calibration-plus-test joint distribution. This
is a distribution-free guarantee: no parametric assumption about the
scoring model is required. Lei et al. (2018) JASA give the canonical
exposition; Angelopoulos & Bates (2023) give the modern gentle
introduction.

**What breaks it.** The guarantee is on exchangeability. If the
production distribution drifts away from calibration, coverage silently
degrades. §8.2 is the companion drift sentinel that fires at the feature
level when PSI exceeds the credit-risk convention threshold of 0.25.

### 8.2 PSI + KL + KS drift sentinels

Slice 7.5b installs three complementary scalar drift metrics between a
reference distribution (the training-time feature snapshot, stored as
`models/<dataset>/train_distribution_snapshot.json`) and the current
inference-time distribution. All three are implemented in pure
numpy/scipy; no new dependencies are added.

**PSI (Population Stability Index).** The industry-standard credit-risk
metric (Siddiqi 2017, Yurdakul 2018), computed as

    PSI = Σ_i (current_i - reference_i) * ln(current_i / reference_i)

over equal-frequency buckets (default 10) of the reference sample. PRA
SS1/23 Section 5 convention:

    PSI <= 0.10               stable
    0.10 < PSI <= 0.25         shift warning
    PSI > 0.25                 significant shift (RuntimeWarning raised)

Both thresholds are exposed via `DriftThresholds(warn=…, alert=…)` for
deployments that prefer stricter cutoffs (e.g. 0.05 / 0.15 for more
conservative stacks).

**KL divergence (nats, current ∥ reference).** Asymmetric by
construction. Informative when PSI fires but direction of shift matters
for diagnosis. A large KL with a small PSI indicates concentrated mass
has moved between buckets without changing marginal shape.

**Two-sample Kolmogorov-Smirnov.** Returns `(statistic, p_value)`. The D
statistic is the max absolute difference between the two empirical CDFs.
The p-value alone is prone to firing on large-sample noise and is
therefore reported alongside PSI rather than as the primary sentinel.

**Snapshot persistence.** `FeatureSnapshot.from_array(name, values)`
subsamples (up to `max_sample_size=5000`) and computes summary stats.
`save_distribution_snapshot(path, snapshots)` writes a JSON envelope
keyed by schema version; `ScoringAgent` calls `load_distribution_snapshot`
at startup. A missing snapshot is a graceful-degradation case: the agent
logs once and proceeds without drift checks (first-run-on-fresh-clone).

**Batch-vs-single-lead semantics.** PSI/KL/KS are batch metrics; they
need ≥ ~50 samples to produce a stable value. Single-lead scoring should
NOT trigger drift checks on a 1-element array — this produces spurious
red flags. The agent collects feature-value batches on a rolling window
(default: last 100 scored leads) and runs the drift check against the
window.

### 8.3 Fairness diagnostics via Fairlearn

Slice 7.5c installs a Fairlearn-based fairness audit that is **purely
diagnostic** — it does not modify predictions. The audit runs across
the protected attributes declared on `LeadInput.protected_attributes:
Dict[str, Any]` (schema extension); the default primary audit axis is
`employee_band`, matching the EU AI Act / FCA DP24/1 informal
convention that SMB-vs-enterprise disparate impact is a stated
regulatory interest.

**Three metrics reported per protected attribute:**

* `selection_rate` per group — fraction of leads in each group with a
  positive prediction. Large disparities suggest the scorer is
  systematically ranking one segment higher than another.
* `demographic_parity_difference` — max group selection rate minus min.
  EU AI Act Article 10 cites this as a baseline bias check.
* `equalized_odds_difference` — max of `|TPR_g - TPR_g'|` and
  `|FPR_g - FPR_g'|` across groups, conditioning on the ground-truth
  label. A stricter criterion than demographic parity (Hardt, Price &
  Srebro 2016 NeurIPS).

**Interpretation threshold.** The informal EU AI Act / FCA convention:

    diff <= 0.10               acceptable
    0.10 < diff <= 0.20         requires attention
    diff > 0.20                 requires intervention or documented
                                 mitigation

configurable via `FairnessThresholds(warn=…, alert=…)`.

**Live-inference mode.** Without `y_true`, only
`demographic_parity_difference` is computed; `equalized_odds_difference`
is reported as `None`. Live-inference deployments should periodically
back-fill `y_true` from closed-won/lost outcomes and re-run the audit
on the retrospective batch.

**Mitigation.** A detected red fairness level is NOT auto-remediated.
Fairlearn's `reductions` and `postprocessing` APIs provide mitigation
primitives; adding a mitigation pass is deferred to Slice 8+. AIF360
and aequitas-ai are deferred to the separate cyber-AI finance product
(SS3/19 climate financial risk), not PinkMate Slice 8.

### 8.4 Composition into the scoring path

The three layers compose behind a single `ProductionHardeningPipeline`
class, instantiated once per `ScoringAgent` at startup with:

* a pre-conformalised `ConformalScoringWrapper` (needs calibration data
  at construction, loaded from `models/<dataset>/calibration.npz`);
* a dict of `FeatureSnapshot` objects loaded from
  `<models_dir>/train_distribution_snapshot.json` (or empty when
  running on a fresh clone);
* `DriftThresholds` and `FairnessThresholds` configurable per
  deployment.

The scoring path then calls two pipeline methods:

* `pipeline.score_lead(features, feature_dict_for_drift, theta)`
  returns a `HardenedScore` carrying the conformal score plus drift
  flags for one lead.
* `pipeline.audit_batch_fairness(y_pred, protected_attributes, y_true)`
  runs the fairness audit once per batch.

The `ScoringAgent` writes both outputs to its trace. The conservative
commit flag is reported as a sibling signal in trace metadata alongside
the Slice 1 commit; downstream policy chooses which to honour.

### 8.5 Trace schema extensions

`LeadInput` gains one optional field:

* `protected_attributes: Dict[str, Any]` — maps attribute name to the
  lead's group label, e.g. `{"employee_band": "SMB"}`. Defaults to an
  empty dict so the schema is additive and all existing callers are
  unaffected.

`ScoreOutput` gains three optional fields:

* `conformal_lower: Optional[float]` — Slice 7.5a lower bound (0-100 scale).
* `conformal_upper: Optional[float]` — Slice 7.5a upper bound (0-100 scale).
* `drift_flags: List[str]` — sorted list of `'<level>:<feature>'` flags
  from Slice 7.5b.

The `LEADINPUT_SCHEMA_VERSION` constant bumps to
`"slice75_production_hardening_v1"`.

### 8.6 Regulatory mapping

| Regulator reference | PinkMate component (slice) |
|---|---|
| PRA SS1/23 Principle 1 (model identification) | Model registry via `models/<dataset>/` subdirectory convention (Slice 5.1) |
| PRA SS1/23 Principle 2 (governance framework) | Agentic commit protocol with trace journal (Slice 1, Slice 3) |
| PRA SS1/23 Principle 3 (conservatism in model use) | Conservative commit rule: `lower_bound >= theta` (Slice 7.5a) |
| PRA SS1/23 Principle 4 (independent model validation) | Fairness diagnostics as independent diagnostic signal (Slice 7.5c) |
| PRA SS1/23 Principle 5 (ongoing monitoring) | PSI + KL + KS drift sentinels on every feature (Slice 7.5b) |
| FCA DP24/1 § on agentic AI explainability | σ_SHAP attribution-stability (Slice 2), σ_calibration (Slice 5.3), σ_SHAP_hazard (Slice 6) |
| EU AI Act Article 10 (data and data governance) | `LeadInput.protected_attributes` schema + fairness audit (Slice 7.5c) |
| EU AI Act Article 13 (transparency) | `docs/probabilistic_framework.md` (Slice 7) + this architecture document |

Alignment is at the architectural level — a mapping between PinkMate's
DESIGN and the regulators' STATED PRINCIPLES. It is not a certification
claim; certification requires an external auditor and sector-specific
scope that this reference deployment does not have. A production FinServ
deployment of the Q-MDP framework (the planned cyber-AI finance product
spin-off) is where formal certification becomes relevant.

### 8.7 References

* Angelopoulos, A. N. & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction*. Foundations and Trends in Machine Learning 16(4), 494-591.
* Bank of England PRA (2023). *SS1/23 Model Risk Management Principles for Banks*. Policy Statement PS6/23, May 2023.
* European Union (2024). *Artificial Intelligence Act*, Regulation (EU) 2024/1689.
* Financial Conduct Authority (2024). *DP24/1 AI in Financial Services*, April 2024.
* Hardt, M., Price, E., Srebro, N. (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS 29.
* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. (2018). *Distribution-free predictive inference for regression*. JASA 113(523).
* Sehgal, K. & Bhatti, K. (2026). *Q-MDP: Superposition-aware state orchestration for deterministic governance of agentic systems in regulated financial and ESG infrastructure*. IEEE Cyber-AI 2026, Submission #4554.
* Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd edition, Wiley, ch. 7.
* Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
* Yurdakul, B. (2018). *Statistical Properties of Population Stability Index*. PhD thesis, Western Michigan University.

---

## 9. Slice roadmap

- **Slice 1 (shipped):** Q-MDP formal structure wired into LangGraph with deterministic replay smoke tests.
- **Slice 2 (shipped):** Imbalance-aware ML layer — six-model ensemble (§III.D), SMOTE (§III.C), TreeSHAP + σ_SHAP measurement (§V), full `scripts/reproduce_table2.py` reproducing Table I + Table II against UCI 572 with tolerance assertions (σ_SHAP ±0.005, recall ±0.03). `ScoringAgent` now loads real pickled models with a transparent fallback to the Slice 1 deterministic predictor when models are not yet trained.
- **Slice 3 (shipped):** Durable checkpointing — `pinkmate/persistence.py::make_checkpointer()` factory selects between `MemorySaver` and `SqliteSaver`. `run_pipeline(initial, thread_id, checkpoint_backend="sqlite", ...)` persists per-node snapshots to disk under `thread_id`. 12 new tests in `tests/test_checkpointing.py` verify SQLite persistence, cross-process replay fidelity (fresh `SqliteSaver` recovers terminal state + full τ_exec), thread isolation, and the factory contract. PostgreSQL swap is a single-call-site change against the factory.
- **Slice 4 (shipped):** Empirical validation of paper §III.B operational claims. `pinkmate/benchmarks/latency.py` measures total latency across n ∈ {5, 10, 25, 50, 100, 250} on a parametric chain topology with near-no-op nodes, isolating per-step scheduler cost. `pinkmate/benchmarks/replay_fidelity.py` generalises Slice 3's cross-process replay proof into a reusable consumer harness (completeness, trace integrity, per-field divergence). `scripts/run_benchmarks.py` executes both and writes committed artefacts: `docs/benchmarks_report.md`, `docs/benchmarks_report.json`, `docs/figures/latency_total_vs_n.png`, `docs/figures/latency_per_step_overhead.png`, `docs/figures/replay_fidelity_summary.png`. 20 new harness-correctness tests in `tests/test_benchmarks.py` (empirical fits not asserted — those wobble under OS scheduler jitter; the committed markdown report carries the quantitative numbers for reviewer inspection).
- **Slice 3.5 (shipped, product-arc ergonomics):** `ScoringAgent` deployment-mode flag — `model_source="zoo"|"placeholder"` separates the research-reproduction path (loads models, warnings on mismatch) from the product-demo path (skips model loading entirely, zero warnings). Backward-compatible default is `"zoo"`. Unblocks product-arc work (website, ClickUp, integrations) without waiting on Slice 5. 9 new tests in `tests/test_scoring_agent_modes.py`.
- **Slice 5 (shipped — methodology validation on synthetic B2B-shaped data):** GTM-domain extension trained on LeadForge Lead Scoring v1 (MIT licence, synthetic B2B SaaS procurement scenario). `pinkmate/ml/leadforge_loader.py` handles the HF download + preprocessing; `scripts/train_gtm_ensemble.py` trains the six-model ensemble + σ_SHAP harness + writes committed artefacts to `models/gtm/`, `docs/gtm_training_report.md`, `docs/figures/gtm/*.png`. Subdirectory restructure: `models/{gtm,taiwan}/` siblings. `ScoringAgent` default `models_dir` → `models/gtm/`. `LeadInput` gains optional `employee_band`, `estimated_revenue_band`, `process_maturity_band` fields. Qualitative-direction verdicts (SMOTE INCREASES σ_SHAP; LightGBM σ_SHAP < XGBoost σ_SHAP), NOT magnitude-tolerance. 36 new tests. σ_SHAP numbers validate the framework runs end-to-end on B2B-shaped data; **DO NOT constitute performance claims about PinkMate on real B2B GTM leads.** See §5d above for full detail.
- **Slice 5.1 (deferred):** Secondary sanity check — X Education Lead Scoring (real but EdTech-domain, feature-schema divergent) + LeadForge advanced tier (stress-test under higher noise/missingness).
- **Later (deferred, data-gated):** Real B2B conversion data path — pink-mate.co.uk landing-page capture over 6-12 months, academic partnership via Kent Business School / Khawar network, or specific industry engagement. When this arrives, retrain Slice 5 ensemble on real labels, re-add `funding_stage` to the training schema, promote to `models/gtm/`.

---

## 10. Running the pipeline

### Smoke-test the Q-MDP orchestration layer (fast, no ML training required)

```bash
pip install -r requirements.txt
python -m workflows.pipeline
```

Expected output: a demo lead ingested, scored, committed through governance, segmented into a tier, and routed to a next-best action, with the full execution trace printed. The ScoringAgent falls back to its deterministic placeholder predictor (a Slice 1 behaviour retained in Slice 2) when real trained models are not yet present on disk, and emits a `RuntimeWarning` noting this.

### Run against the durable SQLite checkpointer (Slice 3)

```python
from pathlib import Path
from pinkmate.state import LeadInput, PipelineState
from workflows.pipeline import run_pipeline

initial = PipelineState(
    lead_input=LeadInput(
        lead_id="DEMO-SQLITE",
        company="Acme SaaS",
        role="VP of Engineering",
        industry="SaaS",
        funding_stage="Series_A",
        team_size=120,
        source="demo",
    )
)

final = run_pipeline(
    initial,
    thread_id="demo-sqlite-thread",
    checkpoint_backend="sqlite",
    checkpoint_path=Path("checkpoints/pinkmate.sqlite"),
)
```

The on-disk SQLite file persists every per-node `PipelineState` snapshot under `thread_id`. A fresh Python process can later load the same file and call `app.get_state(config)` on a freshly-compiled graph to recover the terminal state and the full execution trace — the paper's §III.B replay-fidelity contract. See `tests/test_checkpointing.py::TestCrossProcessReplayFidelity` for the asserted guarantees.

### Run the Slice 4 empirical benchmarks

```bash
python scripts/run_benchmarks.py             # Full run (~5-10 min)
python scripts/run_benchmarks.py --fast      # Quick check (~2 min)
```

Regenerates the committed artefacts under `docs/`: `benchmarks_report.md`, `benchmarks_report.json`, and three PNG charts in `docs/figures/`. The committed version lives in the repo; running the script against a different machine will produce different absolute numbers (due to CPU, OS, Python version) but the empirical pattern — bounded per-step overhead, linear total-latency scaling on chain topology, 3/3 replay-fidelity threads passing strict — should hold on any reasonable modern system.

### Reproduce the paper's Table I and Table II (requires UCI network access)

```bash
python scripts/reproduce_table2.py             # Full K=50 run (~10 min)
python scripts/reproduce_table2.py --fast      # K=10 quick check (~2 min)
python scripts/reproduce_table2.py --synthetic # Hermetic synthetic run (CI)
```

The script:
- Fetches Taiwan Bankruptcy UCI 572 via `ucimlrepo`, caches to `data/taiwan/`.
- Trains the six-model zoo on the SMOTE-oversampled training partition.
- Computes `σ_SHAP` for XGBoost+LightGBM with SMOTE and XGBoost without SMOTE.
- Prints a side-by-side comparison against the paper's declared Table II values with tolerance thresholds (σ_SHAP ±0.005, recall ±0.03) and PASS/DRIFT status.
- Persists trained models + metrics + attribution summary to `models/` (gitignored).

### Run the test suite

```bash
pytest -q
```

Current totals: Slice 1 Q-MDP + pipeline tests + Slice 2 ML + integration tests. All run on synthetic data for hermeticity.

---

## 11. Citing

If you build on this reference deployment, please cite the paper:

```bibtex
@inproceedings{sehgal_bhatti_qmdp_2026,
  author    = {Karan Sehgal and Khawar Naveed Bhatti},
  title     = {{Q-MDP}: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and {ESG} Infrastructure},
  booktitle = {Proceedings of the 2nd IEEE International Conference on Cybersecurity and AI-Based Systems (Cyber-AI 2026)},
  year      = {2026},
  address   = {Bucharest, Romania},
  note      = {Submission \#4554}
}
```
<!-- APPEND this block to docs/architecture.md — paste BEFORE Section 8 (Citing). -->

