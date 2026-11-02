"""
Empirical benchmarks against the paper's §III.B operational claims.

Slice 4 ships two measurement harnesses:

    - latency.py          — forward-scheduling latency vs graph size n;
                            empirical validation of the T_fa = O(log n + k)
                            scheduling bound (paper §III.B).

    - replay_fidelity.py  — generalised cross-process replay harness; takes
                            a checkpointed thread, replays from the
                            SqliteSaver, emits a completeness-and-divergence
                            report. Reusable across deployments (PinkMate,
                            ESG companion).

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Submission #4554.
"""

from __future__ import annotations

from pinkmate.benchmarks.latency import (
    LatencyMeasurement,
    LatencyBenchmarkResult,
    measure_pipeline_latency,
    run_latency_benchmark,
    fit_scheduling_overhead,
)
from pinkmate.benchmarks.replay_fidelity import (
    FieldDivergence,
    FidelityReport,
    run_replay_fidelity_check,
)

__all__ = [
    "LatencyMeasurement",
    "LatencyBenchmarkResult",
    "measure_pipeline_latency",
    "run_latency_benchmark",
    "fit_scheduling_overhead",
    "FieldDivergence",
    "FidelityReport",
    "run_replay_fidelity_check",
]
