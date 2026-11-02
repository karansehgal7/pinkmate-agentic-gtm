"""
Standalone runner for the Slice 4 empirical benchmarks.

Executes:
    1. The latency benchmark (`pinkmate.benchmarks.latency`) across the
       default n axis {5, 10, 25, 50, 100, 250} with 10 runs per n.
    2. The replay-fidelity harness (`pinkmate.benchmarks.replay_fidelity`)
       against a fresh end-to-end run of the real PinkMate pipeline on
       the SqliteSaver backend.

Produces:
    - `docs/figures/latency_total_vs_n.png`
    - `docs/figures/latency_per_step_overhead.png`
    - `docs/figures/replay_fidelity_summary.png`
    - `docs/benchmarks_report.md`

The PNGs and the markdown report are committed to the repo so RAEng /
academic reviewers can read the empirical validation in-browser without
cloning + installing.

CLI flags:
    --fast             Reduced n axis {5, 10, 25, 50} with 4 runs per n;
                       takes ~2 minutes instead of ~5-10 for the full run.
    --no-charts        Skip PNG chart generation (markdown report only).
    --output-dir PATH  Where to write artefacts; defaults to the repo
                       `docs/` directory.
    --runs-per-n N     Override runs per n (default 10, min 1).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Repo-root path bootstrap — this script is designed to be runnable as
# `python scripts/run_benchmarks.py` from the repo root, so we add the
# repo root to sys.path so `import pinkmate` works regardless of PYTHONPATH.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pinkmate.benchmarks.latency import (
    LatencyBenchmarkResult,
    run_latency_benchmark,
)
from pinkmate.benchmarks.replay_fidelity import (
    FidelityReport,
    run_replay_fidelity_check,
)
from pinkmate.persistence import make_checkpointer, thread_config
from pinkmate.state import LeadInput, PipelineState


DEFAULT_N_AXIS = (5, 10, 25, 50, 100, 250)
FAST_N_AXIS = (5, 10, 25, 50)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _mk_lead(lead_id: str) -> PipelineState:
    return PipelineState(
        lead_input=LeadInput(
            lead_id=lead_id,
            company="BenchCo",
            role="VP of Engineering",
            industry="SaaS",
            funding_stage="Series_A",
            team_size=120,
            source="benchmark-runner",
        )
    )


def run_full_benchmarks(
    n_values: Sequence[int],
    runs_per_n: int,
    sqlite_path: Path,
) -> tuple[LatencyBenchmarkResult, List[FidelityReport]]:
    """Execute both benchmarks and return structured results."""
    print(f"[1/2] Latency benchmark — n ∈ {sorted(n_values)}, {runs_per_n} runs each...")
    latency = run_latency_benchmark(n_values=n_values, runs_per_n=runs_per_n)
    print(f"      Done. Per-node baseline: {latency.per_node_baseline_ms:.3f} ms, "
          f"marginal: {latency.marginal_per_node_ms:.3f} ms")

    print(f"[2/2] Replay-fidelity harness — persisting a run then recovering...")
    # Use the real PinkMate pipeline (not the benchmark chain), so the
    # fidelity report reflects production state shapes.
    from workflows.pipeline import build_pipeline

    thread_ids = ["replay-fidelity-run-1", "replay-fidelity-run-2", "replay-fidelity-run-3"]
    reference_states: dict[str, PipelineState] = {}

    # Phase 1 — original "process": write checkpoints to disk.
    original_checkpointer = make_checkpointer(backend="sqlite", path=sqlite_path)
    try:
        original_app = build_pipeline(checkpointer=original_checkpointer)
        for tid in thread_ids:
            final = original_app.invoke(_mk_lead(tid), config=thread_config(tid))
            if isinstance(final, dict):
                final = PipelineState.model_validate(final)
            reference_states[tid] = final
    finally:
        # Explicitly close the SqliteSaver connection — on Windows an open
        # sqlite3.Connection locks the file against deletion ([WinError 32]),
        # which breaks the tempfile.TemporaryDirectory cleanup in main().
        # On POSIX this is a safe no-op; the OS handles unlinks on open
        # files gracefully.
        try:
            if hasattr(original_checkpointer, "conn"):
                original_checkpointer.conn.close()
        except Exception:
            pass
    del original_app
    del original_checkpointer

    # Phase 2 — fresh "process": fidelity checker opens a new checkpointer.
    # Each run_replay_fidelity_check() opens its own SqliteSaver and closes
    # it in a `finally` block, so no additional cleanup is needed here.
    reports: List[FidelityReport] = []
    for tid in thread_ids:
        report = run_replay_fidelity_check(
            thread_id=tid,
            sqlite_path=sqlite_path,
            reference_state=reference_states[tid],
        )
        reports.append(report)

    print(f"      Done. {sum(1 for r in reports if r.passed())}/{len(reports)} threads "
          f"passed strict replay-fidelity.")

    return latency, reports


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _save_chart_total_latency(result: LatencyBenchmarkResult, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    ns = [m.n for m in result.measurements]
    means = [m.mean_ms for m in result.measurements]
    errs = [m.stdev_ms for m in result.measurements]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.errorbar(ns, means, yerr=errs, fmt="o-", capsize=4,
                color="#d94f4f", label="Measured (mean ± stdev)",
                markersize=6, linewidth=1.5)

    # Linear fit curve.
    lin_a, lin_b = result.linear_fit.coefficients
    lin_curve = [lin_a * n + lin_b for n in ns]
    ax.plot(ns, lin_curve, "--", color="#4a7fbf", linewidth=1.2,
            label=f"Linear fit (R² = {result.linear_fit.r_squared:.3f})")

    # Log fit curve.
    log_a, log_b = result.log_linear_fit.coefficients
    log_curve = [log_a * math.log(n) + log_b for n in ns]
    ax.plot(ns, log_curve, ":", color="#5aa374", linewidth=1.2,
            label=f"Log-linear fit (R² = {result.log_linear_fit.r_squared:.3f})")

    ax.set_xlabel("Graph size n (number of nodes)")
    ax.set_ylabel("Total pipeline latency (ms)")
    ax.set_title("PinkMate Slice 4 — Total latency vs graph size (chain topology)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_chart_per_step_overhead(result: LatencyBenchmarkResult, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = result.per_step_overhead_series()
    ns = [n for n, _ in series]
    overheads = [o for _, o in series]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(ns, overheads, "o-", color="#5aa374", markersize=7, linewidth=1.5,
            label="Measured per-step overhead")
    ax.axhline(y=result.marginal_per_node_ms, linestyle="--", color="#4a7fbf",
               linewidth=1.2,
               label=f"Marginal per-node (large-n): {result.marginal_per_node_ms:.3f} ms")

    ax.set_xlabel("Graph size n (number of nodes)")
    ax.set_ylabel("Per-step scheduler overhead (ms/step)")
    ax.set_title("PinkMate Slice 4 — Per-step overhead stability vs graph size\n"
                 "(paper §III.B: T_fa = O(log n + k) implies bounded per-step cost)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Y-axis: start near 0 so the stability is visually meaningful.
    ax.set_ylim(bottom=0.0, top=max(overheads) * 1.5)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_chart_replay_fidelity(reports: Sequence[FidelityReport], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thread_ids = [r.thread_id for r in reports]
    completeness = [r.completeness * 100 for r in reports]
    passed = [r.passed() for r in reports]
    colors = ["#5aa374" if p else "#d94f4f" for p in passed]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    bars = ax.barh(thread_ids, completeness, color=colors, alpha=0.85)
    ax.axvline(x=100, linestyle="--", color="#777", linewidth=1,
               label="100% completeness threshold")

    for bar, p, c in zip(bars, passed, completeness):
        label = "PASS" if p else "FAIL"
        ax.text(c + 1, bar.get_y() + bar.get_height() / 2,
                f"{label} ({c:.1f}%)", va="center", fontsize=9)

    ax.set_xlim(0, 115)
    ax.set_xlabel("Replay-fidelity completeness (%)")
    ax.set_title("PinkMate Slice 4 — Cross-process replay fidelity per thread\n"
                 "(paper §III.B replay contract — fresh SqliteSaver on persisted trail)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _render_markdown_report(
    latency: LatencyBenchmarkResult,
    fidelity_reports: Sequence[FidelityReport],
    n_values: Sequence[int],
    runs_per_n: int,
    include_charts: bool,
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    lines: List[str] = []
    lines.append("# PinkMate Slice 4 — Empirical Validation Report")
    lines.append("")
    lines.append(f"- **Generated**: {now}")
    lines.append(f"- **Paper reference**: Sehgal & Bhatti (2026), *Q-MDP: Superposition-Aware "
                 "State Orchestration …*, IEEE Cyber-AI 2026 Submission #4554, §III.B")
    lines.append(f"- **Graph-size axis**: n ∈ {sorted(n_values)}")
    lines.append(f"- **Runs per n**: {runs_per_n}")
    lines.append(f"- **Topology**: linear chain, near-no-op pass-through nodes on real "
                 "`PipelineState` schema")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Latency section ──────────────────────────────────────────────────
    lines.append("## 1. Scheduling latency — paper §III.B bound")
    lines.append("")
    lines.append("**Paper claim**: `T_fa = O(log n + k)` — the scheduler's forward-")
    lines.append("scheduling cost per super-step is bounded by `log n + k`.")
    lines.append("")
    lines.append("**Operational test on chain topology**: a linear chain has sequential")
    lines.append("dependencies, so total latency necessarily scales at least linearly in n.")
    lines.append("The paper's bound is therefore validated by the **per-step** overhead")
    lines.append("staying bounded as n grows (not by total latency being logarithmic — that")
    lines.append("would be impossible on a chain).")
    lines.append("")

    if include_charts:
        lines.append("![Total latency vs graph size](figures/latency_total_vs_n.png)")
        lines.append("")
        lines.append("![Per-step overhead stability](figures/latency_per_step_overhead.png)")
        lines.append("")

    lines.append("### Measurements")
    lines.append("")
    lines.append("| n | mean (ms) | stdev (ms) | ms/step | runs |")
    lines.append("|---|-----------|------------|---------|------|")
    for m in latency.measurements:
        lines.append(f"| {m.n} | {m.mean_ms:.3f} | {m.stdev_ms:.3f} | "
                     f"{m.mean_ms/m.n:.3f} | {len(m.samples_ms)} |")
    lines.append("")

    lines.append("### Fits")
    lines.append("")
    lin = latency.linear_fit
    log = latency.log_linear_fit
    lines.append(f"- **Linear fit**: `T(n) = {lin.coefficients[0]:.4f}·n + {lin.coefficients[1]:.4f}`, "
                 f"R² = **{lin.r_squared:.4f}**")
    lines.append(f"- **Log-linear fit**: `T(n) = {log.coefficients[0]:.4f}·log(n) + {log.coefficients[1]:.4f}`, "
                 f"R² = **{log.r_squared:.4f}**")
    lines.append("")

    lines.append("### Per-step scheduler overhead")
    lines.append("")
    lines.append(f"- **Baseline (n=1)**: {latency.per_node_baseline_ms:.4f} ms "
                 f"(includes first-invocation warm-up)")
    lines.append(f"- **Marginal (large-n)**: {latency.marginal_per_node_ms:.4f} ms "
                 f"(asymptotic per-step cost, warm-up decayed)")
    lines.append("")
    lines.append("Per-step overhead series:")
    lines.append("")
    for n, overhead in latency.per_step_overhead_series():
        lines.append(f"- n = {n}: {overhead:.4f} ms/step")
    lines.append("")

    lines.append(f"### Verdict")
    lines.append("")
    verdict = latency.verdict
    lines.append(f"**{verdict}**")
    lines.append("")
    if verdict == "LINEAR CHAIN AS EXPECTED":
        lines.append("Total latency scales linearly in n with high R² — exactly as expected")
        lines.append("for chain topology with sequential dependencies. The paper's `O(log n + k)`")
        lines.append("per-step bound is operationally validated by the per-step overhead")
        lines.append("remaining bounded (approximately constant) across the n range: the scheduler")
        lines.append("does not pay a growing per-step cost as the graph grows.")
    elif verdict == "LOG-LINEAR OBSERVED":
        lines.append("Total latency scales sub-linearly in n — unusual for chain topology,")
        lines.append("suggesting the scheduler is exploiting opportunistic parallelism or")
        lines.append("per-step work is being batched. Worth deeper investigation.")
    else:
        lines.append("Neither fit meets the R² ≥ 0.9 threshold. Likely sources: measurement")
        lines.append("jitter from OS scheduler or GC pauses, or an insufficient n range to")
        lines.append("distinguish the two models. Re-run with higher runs_per_n or wider n.")
    lines.append("")
    lines.append("### Scope note")
    lines.append("")
    lines.append("This benchmark isolates the chain-topology case. The paper's `O(log n + k)`")
    lines.append("factor applies to parallel-dispatch topologies (fan-out / DAG) where the")
    lines.append("scheduler can exploit concurrent super-step dispatch. A fan-out benchmark")
    lines.append("against a reducer-equipped state model is a natural follow-up (scoped for a")
    lines.append("later slice, not required to validate the per-step bound shown here).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Replay-fidelity section ──────────────────────────────────────────
    lines.append("## 2. Cross-process replay fidelity — paper §III.B contract")
    lines.append("")
    lines.append("**Paper claim**: deterministic reproducibility of τ_exec across orchestration")
    lines.append("restarts.")
    lines.append("")
    lines.append("**Harness**: `pinkmate.benchmarks.replay_fidelity.run_replay_fidelity_check()` —")
    lines.append("a reusable consumer that takes `(thread_id, sqlite_path, reference_state)`,")
    lines.append("opens a FRESH `SqliteSaver` instance against the on-disk SQLite file")
    lines.append("(simulating a cold process restart), recovers the state history, and emits a")
    lines.append("structured completeness-and-divergence report.")
    lines.append("")

    if include_charts:
        lines.append("![Replay fidelity summary](figures/replay_fidelity_summary.png)")
        lines.append("")

    lines.append("### Per-thread reports")
    lines.append("")
    lines.append("| Thread ID | Completeness | Expected depth | Recovered depth | Trace OK | "
                 "Divergent fields | Passed strict |")
    lines.append("|-----------|--------------|----------------|-----------------|----------|"
                 "------------------|---------------|")
    for r in fidelity_reports:
        lines.append(
            f"| {r.thread_id} | {r.completeness*100:.1f}% | {r.expected_history_depth} | "
            f"{r.recovered_history_depth} | {'yes' if r.trace_integrity else 'no'} | "
            f"{len(r.divergences)} | {'PASS' if r.passed() else 'FAIL'} |"
        )
    lines.append("")

    n_pass = sum(1 for r in fidelity_reports if r.passed())
    lines.append(f"### Summary")
    lines.append("")
    lines.append(f"**{n_pass} / {len(fidelity_reports)} threads passed strict replay fidelity** "
                 f"(100% completeness, trace integrity preserved, zero field divergence).")
    lines.append("")

    if n_pass == len(fidelity_reports):
        lines.append("Paper §III.B replay-fidelity contract is operationally confirmed: a fresh")
        lines.append("`SqliteSaver` instance on the persisted on-disk audit trail reconstructs")
        lines.append("the terminal `PipelineState` and the full `τ_exec` execution trace with")
        lines.append("zero divergence from the live original.")
    else:
        lines.append("One or more threads failed the strict replay-fidelity check. Divergence")
        lines.append("details are in the per-thread report JSON (`docs/benchmarks_report.json`).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Footer ───────────────────────────────────────────────────────────
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/run_benchmarks.py                 # Full run (~5-10 min)")
    lines.append("python scripts/run_benchmarks.py --fast          # Quick check (~2 min)")
    lines.append("python scripts/run_benchmarks.py --no-charts     # Markdown only")
    lines.append("```")
    lines.append("")
    lines.append("Artefacts committed to the repo so reviewers see the empirical validation")
    lines.append("in-browser without cloning: `docs/benchmarks_report.md` + "
                 "`docs/figures/*.png`.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="Reduced n axis {5,10,25,50} with 4 runs per n.")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip PNG chart generation.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs",
                        help="Where to write the markdown report + figures.")
    parser.add_argument("--runs-per-n", type=int, default=None,
                        help="Override runs per n (default: 10 full / 4 fast).")
    args = parser.parse_args(argv)

    if args.fast:
        n_values = FAST_N_AXIS
        runs = args.runs_per_n or 4
    else:
        n_values = DEFAULT_N_AXIS
        runs = args.runs_per_n or 10

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"PinkMate Slice 4 — benchmarks")
    print(f"  n axis: {sorted(n_values)}")
    print(f"  runs per n: {runs}")
    print(f"  output: {output_dir}")
    print(f"  charts: {'off' if args.no_charts else 'on'}")
    print()

    with tempfile.TemporaryDirectory() as td:
        sqlite_path = Path(td) / "benchmarks.sqlite"
        latency, fidelity_reports = run_full_benchmarks(
            n_values=n_values,
            runs_per_n=runs,
            sqlite_path=sqlite_path,
        )

    include_charts = not args.no_charts
    if include_charts:
        print("[charts] Generating PNGs...")
        _save_chart_total_latency(latency, figures_dir / "latency_total_vs_n.png")
        _save_chart_per_step_overhead(latency, figures_dir / "latency_per_step_overhead.png")
        _save_chart_replay_fidelity(fidelity_reports, figures_dir / "replay_fidelity_summary.png")
        print(f"[charts] Saved to {figures_dir}/")

    print("[report] Rendering markdown report...")
    md = _render_markdown_report(
        latency=latency,
        fidelity_reports=fidelity_reports,
        n_values=n_values,
        runs_per_n=runs,
        include_charts=include_charts,
    )
    report_path = output_dir / "benchmarks_report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"[report] Wrote {report_path}")

    # Also persist a machine-readable JSON payload for downstream tooling.
    json_payload = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "latency": {
            "per_node_baseline_ms": latency.per_node_baseline_ms,
            "marginal_per_node_ms": latency.marginal_per_node_ms,
            "verdict": latency.verdict,
            "linear_fit_r_squared": latency.linear_fit.r_squared,
            "log_linear_fit_r_squared": latency.log_linear_fit.r_squared,
            "measurements": [m.as_row() for m in latency.measurements],
        },
        "replay_fidelity": [r.as_dict() for r in fidelity_reports],
    }
    json_path = output_dir / "benchmarks_report.json"
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")
    print(f"[report] Wrote {json_path}")

    print()
    print("Done.")
    print(f"  Latency verdict:   {latency.verdict}")
    print(f"  Replay fidelity:   "
          f"{sum(1 for r in fidelity_reports if r.passed())}/{len(fidelity_reports)} passed strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
