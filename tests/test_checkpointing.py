"""
Slice 3 — Durable checkpointing + cross-process replay fidelity.

Verifies the operational mechanism underlying paper §III.B's deterministic
replay claim:

    "deterministic reproducibility of τ_exec across orchestration restarts"

In Slice 1 the trace τ_exec lived only in process memory (MemorySaver) and
was lost on process end — enough to prove the pipeline is deterministic
within a run, but not that execution can be replayed from an audit store.

Slice 3 swaps in SqliteSaver so every post-node PipelineState snapshot
is persisted to disk, keyed by `thread_id`. The tests here confirm:

    1. The SqliteSaver backend persists a run to disk and the compiled
       graph's `.get_state()` API returns the recorded snapshot.

    2. Cross-process replay fidelity: a fresh SqliteSaver instance
       pointed at the same on-disk file — simulating a process restart,
       without needing a subprocess — reads back the identical final
       state. The paper's §III.B contract operationalised.

    3. Thread isolation: distinct `thread_id` values record independent
       audit trails against the same backend, with no cross-contamination.

    4. State-history completeness: the full execution trace τ_exec
       (one checkpoint per orchestration node) is recoverable from the
       store, not just the terminal state.

The tests are hermetic — each uses a `tmp_path`-scoped SQLite file so
there is no shared state across test functions and no reliance on an
external database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver

from pinkmate.persistence import make_checkpointer, thread_config
from pinkmate.state import LeadInput, PipelineState
from workflows.pipeline import build_pipeline, run_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_lead(lead_id: str = "CHK-001", role: str = "VP of Engineering") -> PipelineState:
    """Build a high-quality lead that will commit under θ_governance=0.75."""
    return PipelineState(
        lead_input=LeadInput(
            lead_id=lead_id,
            company=f"CheckpointCo-{lead_id}",
            role=role,
            industry="SaaS",
            funding_stage="Series_A",
            team_size=120,
            source="checkpoint-test",
        )
    )


def _snapshot_values_as_state(snapshot) -> PipelineState:
    """
    Normalise a LangGraph StateSnapshot's .values to a PipelineState.

    Different LangGraph versions return either a dict or a Pydantic
    model from StateSnapshot.values; this helper handles both.
    """
    values = snapshot.values
    if isinstance(values, PipelineState):
        return values
    return PipelineState.model_validate(values)


# ---------------------------------------------------------------------------
# Test 1 — SqliteSaver backend persists state to disk
# ---------------------------------------------------------------------------

class TestSqlitePersistenceToDisk:
    """
    First the obvious: the SqliteSaver backend actually writes checkpoints
    to the on-disk SQLite file and the compiled graph's `.get_state()`
    API returns them.
    """

    def test_sqlite_file_is_created_on_run(self, tmp_path: Path):
        sqlite_path = tmp_path / "pinkmate-chk.sqlite"
        assert not sqlite_path.exists()

        run_pipeline(
            _mk_lead(),
            thread_id="file-created-thread",
            checkpoint_backend="sqlite",
            checkpoint_path=sqlite_path,
        )

        assert sqlite_path.exists(), "SqliteSaver did not create its on-disk file"
        assert sqlite_path.stat().st_size > 0

    def test_checkpoint_rows_persist_under_thread_id(self, tmp_path: Path):
        """
        Raw-SQL inspection — bypass the LangGraph API to prove the
        checkpoint rows exist in the DB keyed by thread_id. Confirms
        the backend is substrate-correct, not merely API-correct.
        """
        sqlite_path = tmp_path / "pinkmate-chk.sqlite"
        run_pipeline(
            _mk_lead(),
            thread_id="raw-sql-thread",
            checkpoint_backend="sqlite",
            checkpoint_path=sqlite_path,
        )

        conn = sqlite3.connect(sqlite_path)
        try:
            thread_count = conn.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
            ).fetchone()[0]
            checkpoint_count = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("raw-sql-thread",),
            ).fetchone()[0]
        finally:
            conn.close()

        assert thread_count == 1
        # The committed path traverses START → ingestion → enrichment → scoring
        # → segmentation → outreach → END. LangGraph writes a checkpoint
        # before each super-step plus a terminal one, so at least 5.
        assert checkpoint_count >= 5, (
            f"Expected ≥5 checkpoint rows for the committed path, got "
            f"{checkpoint_count}"
        )

    def test_get_state_returns_final_state(self, tmp_path: Path):
        """
        `.get_state(config)` on the compiled graph returns the final
        PipelineState after a run — the API contract the checkpointer
        exposes to downstream audit tooling.
        """
        sqlite_path = tmp_path / "pinkmate-chk.sqlite"
        checkpointer = make_checkpointer(backend="sqlite", path=sqlite_path)
        app = build_pipeline(checkpointer=checkpointer)
        config = thread_config("getstate-thread")

        live_final = app.invoke(_mk_lead(), config=config)
        if isinstance(live_final, dict):
            live_final = PipelineState.model_validate(live_final)

        snapshot = app.get_state(config)
        recorded = _snapshot_values_as_state(snapshot)

        assert recorded.committed_action == live_final.committed_action
        assert recorded.score.final_score == live_final.score.final_score
        assert recorded.belief.as_dict() == live_final.belief.as_dict()
        assert recorded.segment == live_final.segment
        assert recorded.next_best_action == live_final.next_best_action


# ---------------------------------------------------------------------------
# Test 2 — Cross-process replay fidelity (the paper's §III.B claim)
# ---------------------------------------------------------------------------

class TestCrossProcessReplayFidelity:
    """
    The paper §III.B replay-fidelity contract: identical input + persisted
    audit trail → identical reconstructed state across orchestration
    restarts. We simulate the "restart" by constructing a fresh
    SqliteSaver instance (with its own sqlite3.Connection) pointed at the
    same on-disk file — the same substrate a cold process-start sees.
    """

    def test_fresh_checkpointer_recovers_terminal_state(self, tmp_path: Path):
        sqlite_path = tmp_path / "replay.sqlite"
        thread_id = "replay-terminal-thread"

        # Phase 1 — original "process".
        checkpointer_a = make_checkpointer(backend="sqlite", path=sqlite_path)
        app_a = build_pipeline(checkpointer=checkpointer_a)
        config = thread_config(thread_id)
        original_final = app_a.invoke(_mk_lead(), config=config)
        if isinstance(original_final, dict):
            original_final = PipelineState.model_validate(original_final)
        # Close out the original backend's underlying connection — the
        # "process end".
        del app_a
        del checkpointer_a

        # Phase 2 — fresh "process", fresh checkpointer, same DB file.
        checkpointer_b = make_checkpointer(backend="sqlite", path=sqlite_path)
        app_b = build_pipeline(checkpointer=checkpointer_b)
        recovered_snapshot = app_b.get_state(config)
        recovered = _snapshot_values_as_state(recovered_snapshot)

        # Terminal-state equality — paper §III.B contract.
        assert recovered.committed_action == original_final.committed_action
        assert recovered.score.final_score == pytest.approx(
            original_final.score.final_score
        )
        assert recovered.belief.as_dict() == original_final.belief.as_dict()
        assert recovered.governance_decision.committed == (
            original_final.governance_decision.committed
        )
        assert recovered.governance_decision.state == (
            original_final.governance_decision.state
        )
        assert recovered.governance_decision.confidence == pytest.approx(
            original_final.governance_decision.confidence
        )
        assert recovered.segment == original_final.segment
        assert recovered.next_best_action == original_final.next_best_action
        assert recovered.deferred == original_final.deferred

    def test_fresh_checkpointer_recovers_full_execution_trace(self, tmp_path: Path):
        """
        τ_exec completeness: the per-node trace entries persist across
        the restart, not just the terminal state. This is the audit-replay
        guarantee — without a complete trace, 'why did execution land
        here?' cannot be answered after the fact.
        """
        sqlite_path = tmp_path / "replay-trace.sqlite"
        thread_id = "replay-trace-thread"

        checkpointer_a = make_checkpointer(backend="sqlite", path=sqlite_path)
        app_a = build_pipeline(checkpointer=checkpointer_a)
        config = thread_config(thread_id)
        original_final = app_a.invoke(_mk_lead(), config=config)
        if isinstance(original_final, dict):
            original_final = PipelineState.model_validate(original_final)
        original_trace_sig = [
            (e.step, e.node, e.action) for e in original_final.trace
        ]
        del app_a
        del checkpointer_a

        checkpointer_b = make_checkpointer(backend="sqlite", path=sqlite_path)
        app_b = build_pipeline(checkpointer=checkpointer_b)
        recovered = _snapshot_values_as_state(app_b.get_state(config))
        recovered_trace_sig = [
            (e.step, e.node, e.action) for e in recovered.trace
        ]

        assert recovered_trace_sig == original_trace_sig, (
            f"τ_exec did not survive the restart.\n"
            f"Original: {original_trace_sig}\n"
            f"Recovered: {recovered_trace_sig}"
        )
        # Every orchestration node that ran should appear in the trace.
        assert len(recovered.trace) == 5
        assert [e.node for e in recovered.trace] == [
            "ingestion", "enrichment", "scoring", "segmentation", "outreach",
        ]

    def test_state_history_is_complete_after_restart(self, tmp_path: Path):
        """
        LangGraph's `.get_state_history()` iterator must return the full
        chain of super-step snapshots after restart — the raw per-node
        audit trail distinct from the τ_exec stored inside PipelineState.
        """
        sqlite_path = tmp_path / "history.sqlite"
        thread_id = "history-thread"

        checkpointer_a = make_checkpointer(backend="sqlite", path=sqlite_path)
        app_a = build_pipeline(checkpointer=checkpointer_a)
        config = thread_config(thread_id)
        app_a.invoke(_mk_lead(), config=config)
        del app_a
        del checkpointer_a

        checkpointer_b = make_checkpointer(backend="sqlite", path=sqlite_path)
        app_b = build_pipeline(checkpointer=checkpointer_b)
        history = list(app_b.get_state_history(config))

        # Each orchestration super-step produces one history entry;
        # a 5-node committed-path pipeline produces at least that many.
        assert len(history) >= 5, (
            f"State history truncated after restart: got {len(history)} "
            f"entries, expected ≥5."
        )


# ---------------------------------------------------------------------------
# Test 3 — Thread isolation
# ---------------------------------------------------------------------------

class TestThreadIsolation:
    """
    Distinct thread_ids must record independent audit trails against the
    same backend. Mixing threads would break both the audit-replay
    contract (which thread am I replaying?) and the operational contract
    (one lead's partial execution bleeding into another's).
    """

    def test_two_threads_do_not_cross_contaminate(self, tmp_path: Path):
        sqlite_path = tmp_path / "isolation.sqlite"
        checkpointer = make_checkpointer(backend="sqlite", path=sqlite_path)
        app = build_pipeline(checkpointer=checkpointer)

        config_a = thread_config("thread-A")
        config_b = thread_config("thread-B")

        lead_a = _mk_lead(lead_id="LEAD-A", role="CTO")
        lead_b = _mk_lead(lead_id="LEAD-B", role="VP of Engineering")

        app.invoke(lead_a, config=config_a)
        app.invoke(lead_b, config=config_b)

        snap_a = _snapshot_values_as_state(app.get_state(config_a))
        snap_b = _snapshot_values_as_state(app.get_state(config_b))

        assert snap_a.lead_input.lead_id == "LEAD-A"
        assert snap_b.lead_input.lead_id == "LEAD-B"
        assert snap_a.lead_input.role == "CTO"
        assert snap_b.lead_input.role == "VP of Engineering"

    def test_distinct_threads_persisted_separately_in_sqlite(self, tmp_path: Path):
        sqlite_path = tmp_path / "isolation-raw.sqlite"
        checkpointer = make_checkpointer(backend="sqlite", path=sqlite_path)
        app = build_pipeline(checkpointer=checkpointer)

        app.invoke(_mk_lead("LEAD-A"), config=thread_config("thread-A"))
        app.invoke(_mk_lead("LEAD-B"), config=thread_config("thread-B"))
        app.invoke(_mk_lead("LEAD-C"), config=thread_config("thread-C"))

        conn = sqlite3.connect(sqlite_path)
        try:
            threads = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        finally:
            conn.close()

        assert [t[0] for t in threads] == ["thread-A", "thread-B", "thread-C"]


# ---------------------------------------------------------------------------
# Test 4 — Checkpointer factory contract
# ---------------------------------------------------------------------------

class TestCheckpointerFactory:
    """
    `make_checkpointer()` is the one call site that selects the backend.
    These tests pin its contract so swapping in a future PostgresSaver
    (planned later slice) does not silently break assumptions.
    """

    def test_memory_backend_returns_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        cp = make_checkpointer(backend="memory")
        assert isinstance(cp, MemorySaver)

    def test_sqlite_backend_returns_sqlite_saver(self, tmp_path: Path):
        cp = make_checkpointer(backend="sqlite", path=tmp_path / "f.sqlite")
        assert isinstance(cp, SqliteSaver)

    def test_sqlite_backend_creates_parent_directory(self, tmp_path: Path):
        nested = tmp_path / "deeply" / "nested" / "checkpoints" / "f.sqlite"
        assert not nested.parent.exists()
        make_checkpointer(backend="sqlite", path=nested)
        assert nested.parent.exists()
        assert nested.exists()

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported checkpoint backend"):
            make_checkpointer(backend="redis")  # type: ignore[arg-type]
