# Slice 7.5 architecture diagram

ASCII + Mermaid pair for the production hardening slice. Drop into
`docs/architecture.md` under §8 if you want an inline diagram, or
publish separately as `docs/figures/slice_7_5.png` after rendering
the Mermaid.

---

## ASCII — scoring path with hardening pipeline

```
                     ┌────────────────────────────────┐
                     │  LeadInput                     │
                     │  + protected_attributes: dict  │   ← Slice 7.5 schema
                     │  + 30 LeadForge features       │
                     └────────────────┬───────────────┘
                                      │
                                      ▼
            ┌──────────────────────────────────────────────┐
            │     ScoringAgent  (agents/scoring_agent.py)  │
            └──────────────────────┬───────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
   ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
   │  pre-check   │    │  conformal score │    │  (batch handoff)  │
   │  drift       │    │                  │    │  fairness audit   │
   │              │    │  (point, lo, up) │    │                   │
   │  PSI+KL+KS   │    │  + conservative  │    │  Fairlearn        │
   │  Slice 7.5b  │    │  commit flag     │    │  Slice 7.5c       │
   │              │    │  Slice 7.5a      │    │                   │
   └──────┬───────┘    └────────┬─────────┘    └─────────┬─────────┘
          │                     │                        │
          ▼                     ▼                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │       ProductionHardeningPipeline  (pinkmate/ml/)           │
   │                                                             │
   │    conformal.py   +  drift.py  +  fairness.py               │
   │         │                │              │                   │
   │         └────> MAPIE ────┘              └────> Fairlearn     │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────────┐
                     │  ScoringOutput                 │
                     │  + conformal_lower             │   ← Slice 7.5 schema
                     │  + conformal_upper             │
                     │  + drift_flags: list[str]      │
                     │  + is_committed_at_theta       │
                     │  (conservative, not point)     │
                     └────────────────────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────────┐
                     │  Agent trace  (SQLite)         │
                     │  + fairness_reports            │
                     │  + drift_results (batch)       │
                     └────────────────────────────────┘
```

---

## Mermaid — three-way sibling diagram

Positions the production-hardening trio alongside the
measurement-variance trio (σ_SHAP, σ_calibration, σ_SHAP_hazard)
already in the architecture.

```mermaid
flowchart TD
    subgraph Measurement["Measurement variance (Slices 2 / 5.3 / 6)"]
        sshap[σ_SHAP<br/>attribution stability]
        scal[σ_calibration<br/>ECE/Brier variance]
        shazard[σ_SHAP_hazard<br/>survival attribution]
    end

    subgraph Hardening["Production hardening (Slice 7.5)"]
        conformal[Conformal intervals<br/>coverage guarantee]
        drift[PSI + KL + KS<br/>drift sentinels]
        fairness[Fairlearn<br/>fairness diagnostics]
    end

    subgraph Governance["Governance primitive (Slice 1 → refined)"]
        commit_old[commit iff<br/>point_score >= θ]
        commit_new[commit iff<br/>lower_bound >= θ<br/><br/>PRA SS1/23 P3]
    end

    sshap -.supports.-> commit_old
    scal -.supports.-> commit_old
    shazard -.supports.-> commit_old
    conformal ==refines==> commit_new
    drift -.guards.-> commit_new
    fairness -.audits.-> commit_new

    commit_old -.-> commit_new

    classDef shipped fill:#e1f5e1,stroke:#2e7d32,color:#1b5e20
    classDef new fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef refined fill:#fff8e1,stroke:#f9a825,color:#e65100

    class sshap,scal,shazard shipped
    class conformal,drift,fairness new
    class commit_old,commit_new refined
```

---

## Reading guide

The two diagrams tell complementary stories:

* **ASCII** shows the **control flow** at inference time: where each
  check fires, what data it consumes, what it attaches to the trace.
* **Mermaid** shows the **architectural positioning**: Slice 7.5's
  three modules are siblings of the existing σ_* measurement trio,
  and they refine (not replace) the Slice 1 governance primitive.

The ASCII diagram is version-control-friendly (text diff). The
Mermaid rendering is better for the paper figures and the README
hero. Both are kept in sync by hand when the architecture evolves.
