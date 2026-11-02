"""
Checkpointer factory — the durable-persistence backend for the Q-MDP
pipeline's execution trace (τ_exec) and PipelineState snapshots.

Reference:
    Sehgal, K., & Bhatti, K. N. (2026). "Q-MDP: Superposition-Aware State
    Orchestration for Deterministic Governance of Agentic Systems in
    Regulated Financial and ESG Infrastructure." IEEE Cyber-AI 2026,
    Submission #4554.

Paper correspondence:
    §III.B claims deterministic replay of τ_exec across orchestration
    restarts. The checkpointer is the mechanism by which that claim is
    operationalised — every post-node PipelineState is persisted so that:

        1. An interrupted run can be resumed from its last committed node.
        2. An audit replay can reconstruct every intermediate state
           (s_0, a_0, r_0, s_1), ..., (s_{t-1}, a_{t-1}, r_{t-1}, s_t)
           without re-executing stochastic steps.
        3. The same (thread_id, state) pair produces identical outputs
           across process boundaries — the paper's cross-process
           replay-fidelity contract.

Slice roadmap:
    Slice 1 → MemorySaver (in-process only, deterministic replay within
              a single Python process).
    Slice 3 → SqliteSaver (on-disk, cross-process, single-node audit
              trail). THIS SLICE.
    Later  → PostgresSaver (multi-node team / production deployments
              with shared audit infrastructure).

This factory is the one call site that selects the backend, so swapping
in PostgresSaver in a future slice is a single-file change.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Literal, Optional, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

CheckpointBackend = Literal["memory", "sqlite"]
"""Supported checkpointer backends.

- "memory": langgraph.checkpoint.memory.MemorySaver (in-process only).
- "sqlite": langgraph.checkpoint.sqlite.SqliteSaver (on-disk, durable).
"""

DEFAULT_SQLITE_PATH = "checkpoints/pinkmate.sqlite"
"""Default on-disk location for the SQLite checkpoint store.

Chosen inside the repo root so a user running the demo gets a predictable,
discoverable artefact file; `.gitignore` excludes `*.sqlite` and
`checkpoints/` so no checkpoint data is ever committed.
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_checkpointer(
    backend: CheckpointBackend = "memory",
    path: Optional[Union[str, Path]] = None,
) -> object:
    """
    Build a LangGraph checkpointer per the requested durability backend.

    Args:
        backend: "memory" (Slice 1 behaviour, non-durable, in-process) or
            "sqlite" (Slice 3 behaviour, on-disk, cross-process durable).
        path: Only consulted when `backend="sqlite"`. If `None`, defaults
            to ``DEFAULT_SQLITE_PATH`` relative to the current working
            directory. The parent directory is created if it does not
            exist, so callers do not need to pre-create ``checkpoints/``.

    Returns:
        A LangGraph checkpointer object suitable for passing to
        ``StateGraph.compile(checkpointer=...)``. The returned SqliteSaver
        has already had ``setup()`` called so the required tables exist.

    Design note:
        Returning the raw checkpointer (rather than a context manager
        wrapper) is deliberate — the compiled LangGraph app is itself the
        long-lived object, and the checkpointer's sqlite3.Connection stays
        open for its lifetime. Tests that need isolation construct a fresh
        checkpointer per test with a tmp_path-scoped SQLite file.

    Raises:
        ValueError: If `backend` is not one of the supported literals.
    """
    if backend == "memory":
        return MemorySaver()

    if backend == "sqlite":
        resolved_path = Path(path) if path is not None else Path(DEFAULT_SQLITE_PATH)

        # Only create the parent directory for on-disk paths; the special
        # ":memory:" DSN has no parent to create.
        if str(resolved_path) != ":memory:":
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False — LangGraph may interact with the
        # connection from a thread other than the one that created it
        # (notably in async paths and in some test fixtures). This is
        # safe because the checkpointer serialises writes internally.
        conn = sqlite3.connect(str(resolved_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver

    raise ValueError(
        f"Unsupported checkpoint backend: {backend!r}. "
        f"Supported backends: 'memory', 'sqlite'."
    )


# ---------------------------------------------------------------------------
# Convenience helpers for the demo CLI and tests
# ---------------------------------------------------------------------------

def thread_config(thread_id: str) -> dict:
    """
    Build the LangGraph `config` dict that keys a run to a specific
    thread_id in the checkpointer's audit trail.

    Each distinct thread_id is an independent conversation / lead run.
    Re-invoking the compiled graph with the same thread_id resumes from
    (or appends to) that thread's persisted history.
    """
    return {"configurable": {"thread_id": thread_id}}
