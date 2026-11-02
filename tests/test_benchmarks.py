"""
Slice 4 — Harness-correctness tests for the empirical benchmarks.

Scope and non-scope:
    These tests verify that the benchmark harnesses RUN end-to-end,
    produce well-formed structured output, and detect intentional
    divergence when it is injected. They DO NOT assert on empirical
    numerical outcomes (e.g., "linear fit R² must be ≥ 0.9") because
    empirical fits legitimately wobble ±5-15% run-to-run under OS
    scheduler jitter and GC pauses; a numerical assertion would make
    the test suite flaky without reflecting a real regression.

    The quantitative empirical assessment (fits, verdict, per-step
    overhead stability) is produced by `scripts/run_benchmarks.py` and
    committed to the repo at `docs/benchmarks_report.md` +
    `docs/figures/*.png` for reviewer inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinkmate.benchmarks.latency import (
    CurveFit,
    LatencyBenchmarkResult,
    LatencyMeasurement,
    build_chain_graph,
    fit_scheduling_overhead,
    measure_pipeline_latency,
    run_latency_benchmark,
)
from pinkmate.benchmarks.replay_fidelity import (
    FidelityReport,
    FieldDivergence,
    run_replay_fidelity_check,
)
from pinkmate.persistence import make_checkpointer, thread_config
from pinkmate.state import LeadInput, PipelineState
from workflows.pipeline import build_pipeline


# ---------------------------------------------------------------------------
# Latency harness — structure and correctness
# ---------------------------------------------------------------------------

class TestChainGraphBuilder:
    def test_builds_graph_of_specified_size(self):
        app = build_chain_graph(n=5)
        assert app is not None, "builder returned None"

    def test_rejects_zero_or_negative_n(self):
        with pytest.raises(ValueError, match="n >= 1"):
            build_chain_graph(n=0)
        with pytest.raises(ValueError, match="n >= 1"):
            build_chain_graph(n=-1)

    def test_chain_graph_executes_end_to_end(self):
        """A built chain graph should actually run on an invoke call."""
        app = build_chain_graph(n=3)
        state = PipelineState(
            lead_input=LeadInput(lead_id="TEST-CHAIN", source="test")
        )
        config = {"configurable": {"thread_id": "chain-smoke"}}
        final = app.invoke(state, config=config)
        if isinstance(final, dict):
            final = PipelineState.model_validate(final)
        # Each node appends one TraceEntry; expect n = 3 entries.
        assert len(final.trace) == 3
        assert [e.node for e in final.trace] == ["node_0", "node_1", "node_2"]


class TestLatencyMeasurement:
    def test_measurement_structure(self):
        m = measure_pipeline_latency(n=3, runs=3, warmup_runs=1)
        assert m.n == 3
        assert len(m.samples_ms) == 3
        assert all(s > 0 for s in m.samples_ms)
        assert m.mean_ms > 0
        assert m.stdev_ms >= 0

    def test_rejects_zero_runs(self):
        with pytest.raises(ValueError, match="runs must be >= 1"):
            measure_pipeline_latency(n=3, runs=0)

    def test_rejects_negative_warmup(self):
        with pytest.raises(ValueError, match="warmup_runs must be >= 0"):
            measure_pipeline_latency(n=3, runs=1, warmup_runs=-1)

    def test_as_row_returns_dict_with_expected_keys(self):
        m = LatencyMeasurement(n=5, samples_ms=(1.2, 1.5, 1.3))
        row = m.as_row()
        assert set(row.keys()) == {"n", "mean_ms", "stdev_ms", "runs"}
        assert row["n"] == 5
        assert row["runs"] == 3


class TestCurveFitting:
    def test_linear_fit_on_linear_data_has_perfect_r_squared(self):
        """A clean y = 2x + 1 series must fit perfectly in the linear model."""
        ms = tuple(
            LatencyMeasurement(n=n, samples_ms=(2.0 * n + 1.0,))
            for n in (5, 10, 25, 50, 100)
        )
        linear_fit, _log_fit = fit_scheduling_overhead(ms, per_node_baseline_ms=0.0)
        assert linear_fit.r_squared > 0.9999
        assert linear_fit.coefficients[0] == pytest.approx(2.0, rel=1e-6)
        assert linear_fit.coefficients[1] == pytest.approx(1.0, abs=1e-6)

    def test_fit_requires_at_least_two_measurements(self):
        single = (LatencyMeasurement(n=5, samples_ms=(1.0,)),)
        with pytest.raises(ValueError, match="at least 2"):
            fit_scheduling_overhead(single, per_node_baseline_ms=0.0)


class TestFullLatencyBenchmark:
    def test_runs_end_to_end_on_tiny_n_range(self):
        result = run_latency_benchmark(n_values=[5, 10, 25], runs_per_n=3)
        assert len(result.measurements) == 3
        assert result.per_node_baseline_ms > 0
        assert result.marginal_per_node_ms is not None

    def test_result_has_both_fits(self):
        result = run_latency_benchmark(n_values=[5, 10, 25], runs_per_n=3)
        assert isinstance(result.linear_fit, CurveFit)
        assert isinstance(result.log_linear_fit, CurveFit)
        assert result.linear_fit.model == "linear"
        assert result.log_linear_fit.model == "log-linear"

    def test_verdict_is_one_of_expected_literals(self):
        result = run_latency_benchmark(n_values=[5, 10, 25], runs_per_n=3)
        assert result.verdict in (
            "LINEAR CHAIN AS EXPECTED",
            "LOG-LINEAR OBSERVED",
            "AMBIGUOUS",
        )

    def test_per_step_overhead_series_matches_measurement_count(self):
        result = run_latency_benchmark(n_values=[5, 10, 25], runs_per_n=3)
        series = result.per_step_overhead_series()
        assert len(series) == len(result.measurements)
        for (n, overhead), m in zip(series, result.measurements):
            assert n == m.n
            assert overhead == pytest.approx(m.mean_ms / m.n)

    def test_rejects_empty_n_values(self):
        with pytest.raises(ValueError, match="n_values must be non-empty"):
            run_latency_benchmark(n_values=[], runs_per_n=1)

    def test_rejects_zero_or_negative_n(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            run_latency_benchmark(n_values=[0, 5], runs_per_n=1)


# ---------------------------------------------------------------------------
# Replay-fidelity harness — detection correctness
# ---------------------------------------------------------------------------

def _mk_lead(lead_id: str = "FID-001") -> PipelineState:
    return PipelineState(
        lead_input=LeadInput(
            lead_id=lead_id, company="FidCo", role="VP of Engineering",
            industry="SaaS", funding_stage="Series_A", team_size=120,
            source="fidelity-test",
        )
    )


def _run_once_and_persist(sqlite_path: Path, thread_id: str) -> PipelineState:
    """Run the real PinkMate pipeline once, persisting to the given sqlite file."""
    checkpointer = make_checkpointer(backend="sqlite", path=sqlite_path)
    app = build_pipeline(checkpointer=checkpointer)
    final = app.invoke(_mk_lead(thread_id), config=thread_config(thread_id))
    if isinstance(final, dict):
        final = PipelineState.model_validate(final)
    del app  # close the connection — simulate process end
    del checkpointer
    return final


class TestFidelityHarnessDetectsIdentical:
    def test_identical_replay_passes_strict(self, tmp_path: Path):
        sqlite_path = tmp_path / "fid.sqlite"
        thread_id = "identical-thread"
        reference = _run_once_and_persist(sqlite_path, thread_id)

        report = run_replay_fidelity_check(
            thread_id=thread_id,
            sqlite_path=sqlite_path,
            reference_state=reference,
        )
        assert report.completeness == 1.0
        assert report.trace_integrity is True
        assert len(report.divergences) == 0
        assert report.passed() is True

    def test_report_structure(self, tmp_path: Path):
        sqlite_path = tmp_path / "fid.sqlite"
        thread_id = "structure-thread"
        reference = _run_once_and_persist(sqlite_path, thread_id)

        report = run_replay_fidelity_check(
            thread_id=thread_id,
            sqlite_path=sqlite_path,
            reference_state=reference,
        )
        payload = report.as_dict()
        assert set(payload.keys()) >= {
            "thread_id", "completeness", "trace_integrity",
            "divergence_count", "divergences", "passed_strict",
        }
        assert payload["thread_id"] == thread_id


class TestFidelityHarnessDetectsDivergence:
    def test_divergence_detected_when_reference_mutated(self, tmp_path: Path):
        """
        Inject synthetic divergence by mutating the reference state's score
        after persistence. The harness must flag this as a field divergence.
        """
        sqlite_path = tmp_path / "fid-divergent.sqlite"
        thread_id = "divergent-thread"
        reference = _run_once_and_persist(sqlite_path, thread_id)

        # Deliberately corrupt the reference so recovered ≠ reference.
        original_score = reference.score.final_score
        reference.score.final_score = original_score + 42.0

        report = run_replay_fidelity_check(
            thread_id=thread_id,
            sqlite_path=sqlite_path,
            reference_state=reference,
        )
        assert report.completeness == 1.0  # structure still recovered fine
        assert report.passed() is False
        assert any(
            d.field_path == "score.final_score" for d in report.divergences
        ), f"score.final_score divergence not detected; got: " \
           f"{[d.field_path for d in report.divergences]}"

    def test_passed_with_tolerance_above_delta(self, tmp_path: Path):
        """A delta within the tolerance window should pass."""
        sqlite_path = tmp_path / "fid-tol.sqlite"
        thread_id = "tolerance-thread"
        reference = _run_once_and_persist(sqlite_path, thread_id)

        reference.score.final_score += 0.001  # tiny delta
        report = run_replay_fidelity_check(
            thread_id=thread_id,
            sqlite_path=sqlite_path,
            reference_state=reference,
        )
        assert report.passed(tolerance=0.01) is True
        assert report.passed(tolerance=0.0) is False


# ---------------------------------------------------------------------------
# Runner script — produces committed artefacts
# ---------------------------------------------------------------------------

class TestRunBenchmarksScript:
    def test_runner_produces_markdown_report_and_json(self, tmp_path: Path):
        """
        End-to-end test of the standalone runner in fast mode. Verifies
        the markdown and JSON artefacts land in the output directory and
        contain the expected top-level fields.
        """
        # Dynamic import — avoid import side-effects at module load.
        import importlib.util
        import sys as _sys

        repo_root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "run_benchmarks_mod",
            repo_root / "scripts" / "run_benchmarks.py",
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["run_benchmarks_mod"] = mod
        spec.loader.exec_module(mod)

        exit_code = mod.main([
            "--fast",
            "--no-charts",
            "--output-dir", str(tmp_path),
            "--runs-per-n", "2",
        ])
        assert exit_code == 0

        report_md = tmp_path / "benchmarks_report.md"
        report_json = tmp_path / "benchmarks_report.json"
        assert report_md.exists()
        assert report_json.exists()

        md_text = report_md.read_text(encoding="utf-8")
        assert "Scheduling latency" in md_text
        assert "Cross-process replay fidelity" in md_text
        assert "LINEAR CHAIN AS EXPECTED" in md_text or \
               "LOG-LINEAR OBSERVED" in md_text or \
               "AMBIGUOUS" in md_text

        payload = json.loads(report_json.read_text(encoding="utf-8"))
        assert "latency" in payload
        assert "replay_fidelity" in payload
        assert "verdict" in payload["latency"]
        assert payload["latency"]["verdict"] in (
            "LINEAR CHAIN AS EXPECTED",
            "LOG-LINEAR OBSERVED",
            "AMBIGUOUS",
        )
