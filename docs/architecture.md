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
| §III.B, scheduling bound | `T_fa = O(log n + k)` | Enforced by LangGraph priority scheduler; empirical validation in Slice 4 |
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

---

## 6. Slice roadmap

- **Slice 1 (shipped):** Q-MDP formal structure wired into LangGraph with deterministic replay smoke tests.
- **Slice 2 (shipped):** Imbalance-aware ML layer — six-model ensemble (§III.D), SMOTE (§III.C), TreeSHAP + σ_SHAP measurement (§V), full `scripts/reproduce_table2.py` reproducing Table I + Table II against UCI 572 with tolerance assertions (σ_SHAP ±0.005, recall ±0.03). `ScoringAgent` now loads real pickled models with a transparent fallback to the Slice 1 deterministic predictor when models are not yet trained.
- **Slice 3:** Durable checkpointing — swap `MemorySaver` for `SqliteSaver`, then PostgreSQL. Enables cross-process audit-trail persistence (per §III.B replay fidelity claim).
- **Slice 4:** Empirical validation — latency benchmarks against the `T_fa = O(log n + k)` scheduling claim (§III.B), replay-fidelity harness, audit-trace-completeness metric aligned with the ESG companion paper's measurement.

---

## 7. Running the pipeline

### Smoke-test the Q-MDP orchestration layer (fast, no ML training required)

```bash
pip install -r requirements.txt
python -m workflows.pipeline
```

Expected output: a demo lead ingested, scored, committed through governance, segmented into a tier, and routed to a next-best action, with the full execution trace printed. The ScoringAgent falls back to its deterministic placeholder predictor (a Slice 1 behaviour retained in Slice 2) when real trained models are not yet present on disk, and emits a `RuntimeWarning` noting this.

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
