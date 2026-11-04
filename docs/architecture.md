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
| §VI.A, reproducibility protocol | seed=42, 80/20 stratified, SMOTE on train only, TreeSHAP K=50 × 100 × 200 | `pinkmate/ml/pipeline.py` → `run_slice_two_pipeline()`; dataset in `pinkmate/ml/dataset.py` |
| §VI.B / §VI.C, Table I / Table II | Per-model metrics + σ_SHAP comparison | `scripts/reproduce_table2.py` |

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

## 6. Slice roadmap

- **Slice 1 (shipped):** Q-MDP formal structure wired into LangGraph with deterministic replay smoke tests.
- **Slice 2 (shipped):** Imbalance-aware ML layer — six-model ensemble (§III.D), SMOTE (§III.C), TreeSHAP + σ_SHAP measurement (§V), full `scripts/reproduce_table2.py` reproducing Table I + Table II against UCI 572 with tolerance assertions (σ_SHAP ±0.005, recall ±0.03). `ScoringAgent` now loads real pickled models with a transparent fallback to the Slice 1 deterministic predictor when models are not yet trained.
- **Slice 3 (shipped):** Durable checkpointing — `pinkmate/persistence.py::make_checkpointer()` factory selects between `MemorySaver` and `SqliteSaver`. `run_pipeline(initial, thread_id, checkpoint_backend="sqlite", ...)` persists per-node snapshots to disk under `thread_id`. 12 new tests in `tests/test_checkpointing.py` verify SQLite persistence, cross-process replay fidelity (fresh `SqliteSaver` recovers terminal state + full τ_exec), thread isolation, and the factory contract. PostgreSQL swap is a single-call-site change against the factory.
- **Slice 4 (shipped):** Empirical validation of paper §III.B operational claims. `pinkmate/benchmarks/latency.py` measures total latency across n ∈ {5, 10, 25, 50, 100, 250} on a parametric chain topology with near-no-op nodes, isolating per-step scheduler cost. `pinkmate/benchmarks/replay_fidelity.py` generalises Slice 3's cross-process replay proof into a reusable consumer harness (completeness, trace integrity, per-field divergence). `scripts/run_benchmarks.py` executes both and writes committed artefacts: `docs/benchmarks_report.md`, `docs/benchmarks_report.json`, `docs/figures/latency_total_vs_n.png`, `docs/figures/latency_per_step_overhead.png`, `docs/figures/replay_fidelity_summary.png`. 20 new harness-correctness tests in `tests/test_benchmarks.py` (empirical fits not asserted — those wobble under OS scheduler jitter; the committed markdown report carries the quantitative numbers for reviewer inspection).
- **Slice 3.5 (shipped, product-arc ergonomics):** `ScoringAgent` deployment-mode flag — `model_source="zoo"|"placeholder"` separates the research-reproduction path (loads models, warnings on mismatch) from the product-demo path (skips model loading entirely, zero warnings). Backward-compatible default is `"zoo"`. Unblocks product-arc work (website, ClickUp, integrations) without waiting on Slice 5. 9 new tests in `tests/test_scoring_agent_modes.py`.
- **Slice 5 (deferred, product-arc, data-gated):** GTM-domain ensemble retrain — train the six-model zoo on actual B2B conversion labels (lead → MQL → SQL → closed-won). Closes the Taiwan-vs-GTM feature-space mismatch currently surfaced by `ScoringAgent`'s `RuntimeWarning` in `"zoo"` mode (no longer a concern in `"placeholder"` mode after Slice 3.5). The σ_SHAP governance harness is domain-agnostic and carries over unchanged. Requires acquiring a labelled B2B conversion corpus (public datasets, partner data, or defensible synthetic generator).

---

## 7. Running the pipeline

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

## 8. Citing

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
