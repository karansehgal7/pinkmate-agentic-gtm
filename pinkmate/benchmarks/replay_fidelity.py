"""
Generalised cross-process replay-fidelity harness for the paper §III.B
replay-fidelity contract.

Paper claim (§III.B):
    Deterministic reproducibility of τ_exec across orchestration restarts.
    Operationally: identical input + persisted audit trail → identical
    reconstructed state across process boundaries.

Slice 3 proved this at the unit level (`tests/test_checkpointing.py::
TestCrossProcessReplayFidelity`). Slice 4 generalises the proof into a
reusable measurement harness that any Q-MDP deployment — PinkMate, the
ESG companion deployment (arXiv:2606.02604), future instantiations — can
run against its own checkpointed audit trails and emit a structured
completeness-and-divergence report.

Design:
    The harness is a pure consumer. It takes a (thread_id, sqlite_path)
    pair plus a reference terminal PipelineState, opens a FRESH
    SqliteSaver + compiled graph against the sqlite file, reconstructs
    the state history, and emits:

        - completeness   : was the full state history recovered?
        - divergence     : per-field delta between the recovered terminal
                           state and the reference terminal state.
        - trace_integrity: does the recovered τ_exec match the original?

    This harness is deliberately *not* opinionated about how the
    reference was produced. A deployment calling it may pass:
        a. The in-memory PipelineState from the original invocation
           (strongest test — compares against live truth).
        b. A previously-persisted reference snapshot (weaker but still
           useful for drift detection over time).

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Submission #4554.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pinkmate.persistence import make_checkpointer, thread_config
from pinkmate.state import PipelineState


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldDivergence:
    """A single divergent field between reference and recovered state."""

    field_path: str
    """Dotted field path, e.g. 'score.final_score' or 'belief.QUALIFIED'."""

    reference_value: Any
    """Value in the reference (original) state."""

    recovered_value: Any
    """Value in the recovered (post-restart) state."""

    delta: Optional[float] = None
    """Absolute numeric delta for scalar fields; None for non-numeric."""

    def as_dict(self) -> dict:
        return {
            "field_path": self.field_path,
            "reference_value": _jsonable(self.reference_value),
            "recovered_value": _jsonable(self.recovered_value),
            "delta": self.delta,
        }


@dataclass(frozen=True)
class FidelityReport:
    """
    Structured output of a single replay-fidelity check.

    Interpretation:
        A deployment passes the paper §III.B contract iff:
            - completeness == 1.0
            - trace_integrity is True
            - divergences is empty (or all deltas are below the
              deployment's chosen tolerance)
    """

    thread_id: str
    """Thread whose checkpoint was replayed."""

    completeness: float
    """Fraction of expected checkpoints recovered: 1.0 = full history,
    0.0 = nothing recovered."""

    expected_history_depth: int
    """Reference count of super-step snapshots in the history (from the
    original run or from the thread's prior history)."""

    recovered_history_depth: int
    """Observed count of snapshots in the recovered history."""

    trace_integrity: bool
    """True iff the recovered PipelineState.trace matches the reference
    trace at the (step, node, action) tuple level."""

    divergences: Tuple[FieldDivergence, ...] = field(default_factory=tuple)
    """List of fields that differed between reference and recovered."""

    def passed(self, tolerance: float = 0.0) -> bool:
        """
        True iff the deployment passes the §III.B replay-fidelity
        contract under the given tolerance.

        Args:
            tolerance: Max absolute delta allowed per numeric field.
                A tolerance of 0.0 (default) requires exact equality.
        """
        if self.completeness < 1.0:
            return False
        if not self.trace_integrity:
            return False
        for d in self.divergences:
            if d.delta is None:
                # Non-numeric divergence is always a hard fail.
                return False
            if d.delta > tolerance:
                return False
        return True

    def as_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "completeness": self.completeness,
            "expected_history_depth": self.expected_history_depth,
            "recovered_history_depth": self.recovered_history_depth,
            "trace_integrity": self.trace_integrity,
            "divergence_count": len(self.divergences),
            "divergences": [d.as_dict() for d in self.divergences],
            "passed_strict": self.passed(tolerance=0.0),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Coerce a value to a JSON-friendly shape for the report payload."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return str(value)
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _snapshot_values_as_state(snapshot: Any) -> PipelineState:
    """Normalise a LangGraph StateSnapshot's `.values` to a PipelineState."""
    values = snapshot.values
    if isinstance(values, PipelineState):
        return values
    return PipelineState.model_validate(values)


# ---------------------------------------------------------------------------
# Divergence computation
# ---------------------------------------------------------------------------

_COMPARED_SCALAR_FIELDS: Tuple[str, ...] = (
    "committed_action",
    "segment",
    "next_best_action",
    "deferred",
    "theta_governance",
    "seed",
)
"""Top-level scalar fields on PipelineState compared for divergence."""


def _compute_divergences(
    reference: PipelineState, recovered: PipelineState
) -> List[FieldDivergence]:
    """
    Compare two PipelineState objects field by field and return a list
    of divergent fields with per-field deltas where applicable.
    """
    divergences: List[FieldDivergence] = []

    # Top-level scalar fields.
    for fname in _COMPARED_SCALAR_FIELDS:
        ref_val = getattr(reference, fname)
        rec_val = getattr(recovered, fname)
        if ref_val != rec_val:
            delta: Optional[float] = None
            if isinstance(ref_val, (int, float)) and isinstance(rec_val, (int, float)):
                delta = abs(float(ref_val) - float(rec_val))
            divergences.append(
                FieldDivergence(
                    field_path=fname,
                    reference_value=ref_val,
                    recovered_value=rec_val,
                    delta=delta,
                )
            )

    # Score — numeric per-model breakdown.
    if reference.score is None and recovered.score is not None:
        divergences.append(FieldDivergence("score", None, "<present>", None))
    elif reference.score is not None and recovered.score is None:
        divergences.append(FieldDivergence("score", "<present>", None, None))
    elif reference.score is not None and recovered.score is not None:
        if reference.score.final_score != recovered.score.final_score:
            divergences.append(
                FieldDivergence(
                    field_path="score.final_score",
                    reference_value=reference.score.final_score,
                    recovered_value=recovered.score.final_score,
                    delta=abs(reference.score.final_score - recovered.score.final_score),
                )
            )

    # Belief — per-state probability dict.
    if reference.belief is not None and recovered.belief is not None:
        ref_belief = reference.belief.as_dict()
        rec_belief = recovered.belief.as_dict()
        for state_label in sorted(set(ref_belief) | set(rec_belief)):
            ref_p = ref_belief.get(state_label)
            rec_p = rec_belief.get(state_label)
            if ref_p != rec_p:
                delta = None
                if isinstance(ref_p, (int, float)) and isinstance(rec_p, (int, float)):
                    delta = abs(float(ref_p) - float(rec_p))
                divergences.append(
                    FieldDivergence(
                        field_path=f"belief.{state_label}",
                        reference_value=ref_p,
                        recovered_value=rec_p,
                        delta=delta,
                    )
                )

    # Governance decision — committed, state, confidence.
    if reference.governance_decision is not None and recovered.governance_decision is not None:
        ref_gov = reference.governance_decision
        rec_gov = recovered.governance_decision
        if ref_gov.committed != rec_gov.committed:
            divergences.append(
                FieldDivergence(
                    "governance_decision.committed",
                    ref_gov.committed, rec_gov.committed, None,
                )
            )
        if ref_gov.state != rec_gov.state:
            divergences.append(
                FieldDivergence(
                    "governance_decision.state",
                    ref_gov.state, rec_gov.state, None,
                )
            )
        if ref_gov.confidence != rec_gov.confidence:
            divergences.append(
                FieldDivergence(
                    "governance_decision.confidence",
                    ref_gov.confidence, rec_gov.confidence,
                    abs(ref_gov.confidence - rec_gov.confidence),
                )
            )

    return divergences


def _compare_trace_integrity(
    reference: PipelineState, recovered: PipelineState
) -> bool:
    """Does the recovered trace match the reference at (step, node, action)?"""
    ref_sig = [(e.step, e.node, e.action) for e in reference.trace]
    rec_sig = [(e.step, e.node, e.action) for e in recovered.trace]
    return ref_sig == rec_sig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_replay_fidelity_check(
    thread_id: str,
    sqlite_path: Union[str, Path],
    reference_state: PipelineState,
    *,
    graph_builder: Optional[Callable[[object], object]] = None,
) -> FidelityReport:
    """
    Replay a persisted thread from its on-disk SqliteSaver file and emit
    a structured fidelity report against the reference terminal state.

    This is a *consumer* harness — it does not re-run the original
    pipeline. It only reads the persisted checkpoints through a fresh
    checkpointer instance, simulating a cold process start.

    Args:
        thread_id: The audit-trail key of the thread to replay.
        sqlite_path: On-disk SQLite file holding the persisted
            checkpoints (the Slice 3 SqliteSaver backend).
        reference_state: The source-of-truth terminal `PipelineState`
            against which divergence is computed. Usually the live
            final state of the original run.
        graph_builder: Factory taking a checkpointer and returning a
            compiled LangGraph app. Defaults to the real PinkMate
            pipeline (`workflows.pipeline.build_pipeline`). Supplying
            a custom builder lets other Q-MDP deployments reuse the
            harness without a hard dependency on the PinkMate graph.

    Returns:
        A `FidelityReport` with completeness, divergences, and
        trace-integrity flags.
    """
    # Default to the real PinkMate pipeline builder — imported lazily so
    # this module is importable in contexts where `workflows/` is not
    # available (e.g., when the harness is reused inside the ESG
    # companion deployment).
    if graph_builder is None:
        from workflows.pipeline import build_pipeline as default_builder
        graph_builder = lambda cp: default_builder(checkpointer=cp)

    checkpointer = make_checkpointer(backend="sqlite", path=sqlite_path)
    app = graph_builder(checkpointer)
    config = thread_config(thread_id)

    # Recover the full state history (iterator → list for depth count).
    history = list(app.get_state_history(config))
    recovered_depth = len(history)

    # Reference history depth — the reference's trace length +1 for the
    # initial pre-execution snapshot is a reasonable proxy. In practice,
    # LangGraph records a snapshot per super-step plus the terminal one;
    # we assert recovered_depth >= len(reference.trace) rather than
    # equality, since LangGraph's internal super-step grouping may bundle
    # multiple node transitions into one checkpoint.
    expected_depth = max(len(reference_state.trace), 1)
    if recovered_depth == 0:
        completeness = 0.0
    else:
        completeness = min(1.0, recovered_depth / expected_depth)

    # Recover the current state and compare against the reference.
    if recovered_depth == 0:
        return FidelityReport(
            thread_id=thread_id,
            completeness=0.0,
            expected_history_depth=expected_depth,
            recovered_history_depth=0,
            trace_integrity=False,
            divergences=tuple(),
        )

    recovered_state = _snapshot_values_as_state(app.get_state(config))
    divergences = _compute_divergences(reference_state, recovered_state)
    trace_ok = _compare_trace_integrity(reference_state, recovered_state)

    return FidelityReport(
        thread_id=thread_id,
        completeness=completeness,
        expected_history_depth=expected_depth,
        recovered_history_depth=recovered_depth,
        trace_integrity=trace_ok,
        divergences=tuple(divergences),
    )
