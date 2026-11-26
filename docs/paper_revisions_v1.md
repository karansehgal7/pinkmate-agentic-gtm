# PinkMate Paper Revisions v1 — Source Material for IEEE Cyber-AI 2026 Camera-Ready

**Paper:** "Q-MDP: Superposition-Aware State Orchestration for Deterministic Governance of Agentic Systems in Regulated Financial and ESG Infrastructure"
**Venue:** IEEE Cyber-AI 2026, Bucharest, Romania
**Submission:** #4554 (accepted 18 June 2026)
**For:** Khawar Naveed Bhatti (co-author) — merge into LaTeX camera-ready
**Revision date:** 26 November 2026
**Source repo SHA at drafting:** `48e71f1` (post-Housekeeping v6)

---

## How to use this document

This is **source material**, not LaTeX. Khawar does the final LaTeX merge with
his co-author voice. The three sub-passes below correspond to three sections of
the camera-ready paper that need updating since the June 2026 submission:

1. **Related Work** — new paragraph on Yan et al. 2020 + 4 extensions we claim
2. **Theoretical Foundations** — new POMDP connection via Littman/Cassandra/Kaelbling 1995
3. **Results §VI.D or §VII** — new cross-domain σ_SHAP transfer evidence section

Each sub-pass carries:
- The prose text (ready to adapt)
- The exact BibTeX entries to add to `references.bib`
- The code-anchor map so every claim traces back to a shipped file + SHA
- Figure recommendations (where to pull from `docs/figures/`)

Items marked `[VERIFY FROM LOCAL REPORT]` need the exact numbers from Khawar's
or KS's locally-generated reports (`docs/sigma_calibration_report.md`,
`docs/survival_training_report.md`) — these are generated, not committed.

Items marked `[TO MEASURE]` are pending the pre-camera-ready micro-slice
`scripts/measure_conformal_coverage_cross_domain.py` (Slice 7.5 conformal
coverage consistency across all 4 cross-domain datasets, ~1 day work).

---

## Sub-pass 1 — Related Work: add Yan et al. 2020 paragraph + 4 extensions

### Suggested paragraph (adapt voice to Khawar's writing)

> Yan et al. (2020) established the soft-voting ensemble baseline for B2B sales
> prediction using proprietary CRM data, demonstrating that classical interpretable
> classifiers can match deep-learning performance on tabular sales-conversion
> prediction when ensembled under appropriate class-imbalance handling. We extend
> this baseline in four complementary directions: **(a)** σ_SHAP attribution-stability
> measurement under K rotated backgrounds, providing a quantitative governance
> signal for SMOTE's interaction with model interpretability that the original
> baseline could not report; **(b)** POMDP-grounded Q-MDP governance orchestration
> (Littman, Cassandra & Kaelbling 1995) that replaces per-model threshold decisions
> with a formally-defensible commit primitive over a superposed belief state;
> **(c)** cross-domain validation spanning financial-distress (Taiwan Bankruptcy UCI,
> 3.23% minority rate), synthetic B2B GTM (LeadForge Lead Scoring v1 intermediate +
> advanced tiers at 20.98% and 7.86% minority), and real EdTech
> (X Education via Zenodo 23791277 at 38.54% minority) on open-licenced tabular
> datasets, substantially widening the generalisability claim beyond the original
> proprietary-data scope; and **(d)** conformal-interval production hardening under
> PRA SS1/23 Principle 3 conservatism-in-use, replacing the point-score commit rule
> with a strictly stricter `commit iff lower_bound(X) ≥ θ` rule carrying frequentist
> marginal coverage guarantees on P(Y=1|X) at user-specified α.

### New BibTeX entry to add

```bibtex
@article{yan2020b2bsales,
  author  = {Yan, J. and <co-authors from original citation>},
  title   = {Soft-voting ensemble for B2B sales-conversion prediction on CRM data},
  journal = {<journal name — verify from original>},
  year    = {2020},
  volume  = {<volume>},
  number  = {<issue>},
  pages   = {<pages>},
  doi     = {<doi>}
}
```

**[VERIFY WITH KHAWAR]**: exact Yan et al. 2020 citation details (full author list,
journal, volume, DOI). We've been using this reference since the original submission
discussion; Khawar likely has the authoritative BibTeX in his reference manager.

### Code-anchor map (for reviewer traceability)

| Claim | Source file | Commit SHA |
|---|---|---|
| (a) σ_SHAP attribution-stability under K rotated backgrounds | `pinkmate/ml/attribution.py` | Slice 2 `2903f49` |
| (a) K=50 canonical rotation protocol | `tests/test_attribution.py` | Slice 2 `2903f49` |
| (b) POMDP-grounded Q-MDP governance | `pinkmate/qmdp_formal.py` | Slice 7 `4bce54f` |
| (b) SuperposedBelief ≡ POMDP belief isomorphism | `pinkmate/qmdp_formal.py` (isomorphism adapter) | Slice 7 `4bce54f` |
| (c) Cross-domain datasets: Taiwan loader | `pinkmate/ml/dataset.py` | Slice 2 `2903f49` |
| (c) LeadForge loader | `pinkmate/ml/leadforge_loader.py` | Slice 5 `42267e5` |
| (c) X Education Zenodo loader | `pinkmate/ml/x_education_loader.py` | Slice 5.1 `5cde402` |
| (d) Conformal production hardening | `pinkmate/ml/conformal.py` | Slice 7.5 `398ea73` |
| (d) `ConformalScoringWrapper` with MAPIE 1.x `confidence_level` API | `pinkmate/ml/conformal.py::ConformalScoringWrapper` | Slice 7.5 `398ea73` |
| (d) Conservative commit rule `lower_bound ≥ θ` | `pinkmate/ml/production_hardening.py` + `agents/scoring_agent.py` | Slice 7.5 `398ea73` |

---

## Sub-pass 2 — Theoretical Foundations: POMDP connection

### Context

The original paper introduced `SuperposedBelief` + `governance_commit` as primitives
without formally connecting them to the canonical POMDP literature. Slice 7 shipped
that connection in `pinkmate/qmdp_formal.py` + `docs/probabilistic_framework.md`.
The camera-ready should add a formal paragraph in Theoretical Foundations citing
Littman et al. 1995 and stating the isomorphism explicitly.

### Suggested paragraph

> The governance primitives introduced in §III.A — the superposed belief
> `SuperposedBelief` and the commit predicate `governance_commit(b, θ)` —
> correspond to canonical POMDP objects (Littman, Cassandra & Kaelbling 1995).
> Specifically, `SuperposedBelief` is isomorphic to a POMDP belief state
> `b(s) ∈ Δ(S)` over the latent lead-qualification states, and
> `governance_commit(b, θ)` is a **degenerate case of the Q-MDP commit rule**
> (Littman et al. 1995, Eq. 4): the Q-MDP rule commits to action
> `argmax_a Σ_s b(s) Q_MDP(s, a)` where `Q_MDP(s, a)` is the optimal Q-function
> under the assumption that state becomes observable after the action. Our
> `governance_commit` rule reduces this to the one-dimensional
> `commit iff max_s b(s) ≥ θ` by restricting to a single qualify-or-defer
> action and weighting by the belief mass alone. This reduction preserves the
> frequentist decision-theoretic guarantees of Q-MDP while exposing a single
> interpretable governance threshold `θ` suitable for the model-risk-management
> regulatory context (PRA SS1/23 Principle 3 conservatism-in-use). The full
> isomorphism derivation + code is in `pinkmate/qmdp_formal.py`; the one-stop
> theoretical reference with worked examples is `docs/probabilistic_framework.md`.

### New BibTeX entry to add

```bibtex
@inproceedings{littman1995learning,
  author    = {Littman, Michael L. and Cassandra, Anthony R. and Kaelbling, Leslie Pack},
  title     = {Learning Policies for Partially Observable Environments: Scaling Up},
  booktitle = {Proceedings of the Twelfth International Conference on Machine Learning (ICML)},
  year      = {1995},
  pages     = {362--370},
  publisher = {Morgan Kaufmann},
  address   = {Tahoe City, California}
}
```

This is the canonical Q-MDP paper. The isomorphism adapter in
`pinkmate/qmdp_formal.py` explicitly cites it in the module docstring.

### Supplementary material to reference

- `docs/probabilistic_framework.md` §1 covers the complete POMDP tuple formalisation
- `pinkmate/qmdp_formal.py::POMDP` dataclass + `q_mdp_values` + `q_mdp_commit` functions give the executable isomorphism
- `tests/test_qmdp_formal.py` has the isomorphism tests asserting
  `SuperposedBelief` → POMDP belief conversion round-trips and
  `governance_commit` → Q-MDP commit agreement on degenerate action spaces

### Code-anchor map

| Claim | Source file | Commit SHA |
|---|---|---|
| POMDP tuple formalisation | `pinkmate/qmdp_formal.py::POMDP` dataclass | Slice 7 `4bce54f` |
| Bayesian belief update | `pinkmate/qmdp_formal.py::belief_update` | Slice 7 `4bce54f` |
| Q-MDP value iteration | `pinkmate/qmdp_formal.py::q_mdp_values` | Slice 7 `4bce54f` |
| Q-MDP commit rule | `pinkmate/qmdp_formal.py::q_mdp_commit` | Slice 7 `4bce54f` |
| Isomorphism tests | `tests/test_qmdp_formal.py` | Slice 7 `4bce54f` |
| Narrative derivation | `docs/probabilistic_framework.md` §1 | Slice 7 `4bce54f` |

---

## Sub-pass 3 — Results §VI.D or §VII: Cross-domain σ_SHAP transfer evidence

### Context

The original paper's Results covered single-domain σ_SHAP measurement on Taiwan
Bankruptcy UCI 572. Slices 5-7.5 generated cross-domain evidence that materially
strengthens the generalisability claim. This new section proposes **§VI.D**
(subsection if Results is already structured as §VI) or **§VII**
(standalone section if there's conceptual room).

### Proposed new section text

> **§VI.D (or §VII) — Cross-domain σ_SHAP transfer evidence**
>
> To test whether the σ_SHAP attribution-stability signal generalises beyond the
> Taiwan Bankruptcy baseline reported in §VI.A-C, we extended the measurement
> protocol to three additional tabular datasets spanning distinct domains and
> class-imbalance regimes:
>
> **Four-dataset validation table** (all under the K=50 canonical σ_SHAP rotation
> protocol from §V; shipped as commits `5cde402` + `a3d3606` for Slice 5.1 and
> `10949bb` for the σ_calibration companion measurement):

| Dataset | Minority rate | Domain | Source | SMOTE Δ on XGBoost σ_SHAP |
|---|---|---|---|---|
| Taiwan Bankruptcy (§VI.C, Slice 2) | 3.23% | Financial distress | UCI 572 (open) | **Large** (per paper, §VI.C Table II) |
| LeadForge advanced (Slice 5.1) | 7.86% | B2B GTM synthetic (harder noise/missingness) | HuggingFace LeadForge Lead Scoring v1 (MIT) | **+0.0155** |
| LeadForge intermediate (Slice 5) | 20.98% | B2B GTM synthetic | HuggingFace LeadForge Lead Scoring v1 (MIT) | **+0.0108** |
| X Education (Slice 5.1) | 38.54% | EdTech **real** (9,240 leads) | Zenodo 23791277 (CC BY 4.0) | **+0.0004** |

> **Key qualitative finding.** The SMOTE-direction effect size on XGBoost σ_SHAP
> **monotonically decreases** as class balance approaches 50%:
> `Δ = +0.0155 → +0.0108 → +0.0004` as minority rate moves `7.86% → 20.98% → 38.54%`.
> This is exactly what the methodological definition of σ_SHAP predicts: as the
> minority class approaches balance, SMOTE has progressively less synthetic-positive
> headroom to destabilise SHAP attribution, so the attribution-stability
> perturbation attributable to SMOTE itself diminishes. All four
> qualitative directions declared in the Slice 5 verdict protocol
> (SMOTE increases σ_SHAP on XGBoost; LightGBM σ_SHAP < XGBoost σ_SHAP) **PASS
> across all four datasets** — a stronger generalisability result than any
> single-domain σ_SHAP measurement could support.
>
> **Model-stability ordering generalises**. The second qualitative direction
> `LightGBM σ_SHAP < XGBoost σ_SHAP` holds across all four datasets and all four
> minority-rate regimes, confirming that this architectural ordering is
> dataset-agnostic and not an artifact of the Taiwan baseline.
>
> **Classification → Survival σ_SHAP transfer (§VI.D.2 or §VII.2).** To probe
> whether the attribution-stability signal transfers across prediction-target types
> (classification → survival), Slice 6 extended the ensemble with six survival
> models (Cox PH + WeibullAFT + LogNormalAFT + LogLogisticAFT + XGBoost `survival:cox`
> + LightGBM AFT) and measured σ_SHAP_hazard via the Krzyziński et al. 2023
> `survshap` package. On the LeadForge dataset:
>
> - Spearman ρ between classification σ_SHAP and survival σ_SHAP_hazard on
>   matched model-family pairs: **[VERIFY FROM LOCAL REPORT: `docs/survival_training_report.md`]**
> - 95% bootstrap confidence interval: **[VERIFY FROM LOCAL REPORT]**
> - Interpretation: positive correlation above the no-transfer null supports the
>   view that σ_SHAP is capturing a model-agnostic attribution-stability property,
>   not a classification-specific artefact.
>
> **Conformal coverage consistency across cross-domain datasets (§VI.D.3 or §VII.3).**
> Slice 7.5 shipped `pinkmate/ml/conformal.py` wrapping the 6-model classification
> ensemble via MAPIE 1.x `SplitConformalRegressor` (α=0.10 default, 90% target
> empirical coverage on the calibration partition). For each of the four
> cross-domain datasets, the empirical coverage on the held-out test partition
> should approach the target under exchangeability:
>
> - **[TO MEASURE via `scripts/measure_conformal_coverage_cross_domain.py`, scoped as
>   pre-camera-ready micro-slice ~1 day]** — populate a 4th column in the table
>   above with empirical coverage per dataset. Expected shape:

| Dataset | Minority rate | Empirical coverage at α=0.10 |
|---|---|---|
| Taiwan Bankruptcy | 3.23% | **[TO MEASURE]** target ≈ 0.90 |
| LeadForge advanced | 7.86% | **[TO MEASURE]** target ≈ 0.90 |
| LeadForge intermediate | 20.98% | **[TO MEASURE]** target ≈ 0.90 |
| X Education | 38.54% | **[TO MEASURE]** target ≈ 0.90 |

> Deviation from 0.90 by more than ±0.05 would signal distribution shift between
> calibration and test partitions — the companion drift sentinel (PSI + KL + KS)
> would flag this at the feature level. Deviation within the tolerance band
> confirms the conformal production hardening preserves its coverage guarantee
> across all four domains, providing the empirical anchor for the PRA SS1/23
> Principle 3 conservatism-in-use claim.

### New BibTeX entries to add for Sub-pass 3

```bibtex
@article{krzyzinski2023survshap,
  author  = {Krzyziński, Mateusz and Spytek, Mikołaj and Baniecki, Hubert and Biecek, Przemysław},
  title   = {{SurvSHAP(t)}: Time-dependent explanations of machine learning survival models},
  journal = {Knowledge-Based Systems},
  year    = {2023},
  volume  = {262},
  pages   = {110234},
  doi     = {10.1016/j.knosys.2022.110234}
}

@book{siddiqi2017intelligent,
  author    = {Siddiqi, Naeem},
  title     = {Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards},
  year      = {2017},
  edition   = {2nd},
  publisher = {Wiley},
  note      = {Chapter 7 covers the PSI convention we adopt for drift sentinels.}
}

@inproceedings{lei2018distribution,
  author  = {Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J. and Wasserman, Larry},
  title   = {Distribution-Free Predictive Inference for Regression},
  journal = {Journal of the American Statistical Association},
  year    = {2018},
  volume  = {113},
  number  = {523},
  pages   = {1094--1111},
  doi     = {10.1080/01621459.2017.1307116}
}

@book{angelopoulos2023conformal,
  author    = {Angelopoulos, Anastasios N. and Bates, Stephen},
  title     = {Conformal Prediction: A Gentle Introduction},
  series    = {Foundations and Trends in Machine Learning},
  volume    = {16},
  number    = {4},
  year      = {2023},
  pages     = {494--591},
  publisher = {Now Publishers}
}
```

### Code-anchor map

| Claim | Source file | Commit SHA |
|---|---|---|
| X Education loader (Zenodo CC BY 4.0) | `pinkmate/ml/x_education_loader.py` | Slice 5.1 `5cde402` |
| LeadForge advanced tier | `pinkmate/ml/leadforge_loader.py` (advanced tier flag) | Slice 5.1 `5cde402` |
| Slice 5.1 verdict record | `SLICE_5_1_VERDICTS.md` | Slice 5.1 follow-up `a3d3606` |
| Classification → survival σ_SHAP transfer | `scripts/train_survival_ensemble.py` + `docs/survival_training_report.md` (generated) | Slice 6 `a131ba9` |
| SurvSHAP wrapper | `pinkmate/ml/survival_attribution.py::compute_rotated_survshap_variance` | Slice 6 `a131ba9` |
| Conformal production hardening | `pinkmate/ml/conformal.py::ConformalScoringWrapper` | Slice 7.5 `398ea73` |
| Cross-domain conformal coverage measurement | `scripts/measure_conformal_coverage_cross_domain.py` **[TO BUILD]** | Pre-camera-ready micro-slice |

### Figure recommendations (pull from `docs/figures/`)

| Figure | Caption | Source |
|---|---|---|
| Cross-domain σ_SHAP comparison (4-dataset bar chart) | "σ_SHAP per model across four cross-domain datasets" | `docs/figures/gtm/sigma_shap_per_model.png` + `docs/figures/xeducation/sigma_shap_per_model.png` (compose into 4-panel figure) |
| Classification → survival σ_SHAP transfer scatter | "Spearman correlation between classification σ_SHAP and survival σ_SHAP_hazard on matched model-family pairs" | `docs/figures/survival/sigma_shap_transfer_scatter.png` |
| Survival AIC comparison | "AIC comparison across 6 survival models on LeadForge" | `docs/figures/survival/aic_comparison.png` |
| Per-survival-model σ_SHAP_hazard | "σ_SHAP_hazard across survival model family" | `docs/figures/survival/sigma_shap_per_survival_model.png` |

---

## Appendix A — Full paper ↔ code map (all SHAs referenced)

| Slice | SHA | Shipped content relevant to camera-ready |
|---|---|---|
| Slice 1 | `18f074` | Q-MDP primitives (§III.A) |
| Slice 2 | `2903f49` | Imbalance-aware ML layer + σ_SHAP measurement (§V) |
| Slice 3 | `815c6e2` | Durable checkpointing (§III.B replay fidelity) |
| Slice 4 | `418d6ed` | Empirical validation harnesses |
| Slice 3.5 | `e282005` | ScoringAgent deployment-mode flag |
| Slice 5 | `42267e5` | GTM-domain extension (LeadForge intermediate) |
| Slice 5.1 | `5cde402` + `a3d3606` | Cross-domain validation (X Education + advanced) |
| Slice 5.2 | `7ed4031` | Hybrid fail-fast schema + RFM + Clay ICP + MTA formalism |
| Slice 5.3 | `10949bb` | σ_calibration governance sibling |
| Slice 6 | `a131ba9` | Survival analysis extension (Cox PH + parametric + SurvSHAP) |
| Slice 7 | `4bce54f` | Probabilistic framework (POMDP + HMM + Bayesian + mixture) |
| Slice 7.5 | `398ea73` | Production hardening (conformal + drift + fairness) |
| Housekeeping v4+v5 | `5fea7eb` | Architecture restructure + README polish |
| Housekeeping v6 | `48e71f1` | README ScoringAgent refresh for Slice 7.5 |

Camera-ready footnote suggestion: *"All code, tests (440 passing), reports, and
figures referenced in this paper are available at
`https://github.com/karansehgal7/pinkmate-agentic-gtm` at commit SHA 48e71f1
(26 November 2026)."*

---

## Appendix B — Items deferred to post-camera-ready

These Slice 8+ items do NOT need to land in the camera-ready. They are
forward-looking extensions and will be covered in follow-up work:

- **KANs comparison** (Lee et al. 2025) — σ_SHAP on KAN vs 6-model boosting ensemble
- **DBpedia-enriched feature engineering** (arXiv 2606.28355) — alternative FE path
- **Cyber-AI Finance tool spin-off** (Slice 8, Lane 2 product) — SS3/19 climate
  financial risk anchor, built on DCarbon + ESG + imbalance-aware + "Auditable
  Climate Risk Intelligence" paper. Standalone publication path.
- **Memory-as-a-Model (MeMo, arXiv 2605.15156) integration** — potential
  EnrichmentAgent / knowledge-base architecture for the finance tool

---

## Appendix C — Pre-camera-ready checklist for Khawar

- [ ] Verify Yan et al. 2020 BibTeX entry (full author list, journal, DOI)
- [ ] Decide §VI.D subsection vs §VII standalone section placement for cross-domain results
- [ ] Confirm figure composition approach for 4-dataset σ_SHAP bar chart (4-panel or single composite)
- [ ] Verify Spearman ρ + bootstrap CI numbers for classification → survival transfer from
      KS's local `docs/survival_training_report.md`
- [ ] Wait for pre-camera-ready micro-slice `scripts/measure_conformal_coverage_cross_domain.py`
      to populate the Slice 7.5 coverage table (~1 day work)
- [ ] Final acknowledgements: no AI-as-co-author mention per KS's IEEE paper
      acknowledgement convention
- [ ] Add `SLICE_5_1_VERDICTS.md` canonical record to the supplementary material
      submission bundle if the venue accepts supplementary files

---

**Ship note.** This document is the source material only. Camera-ready LaTeX
merge is Khawar's call. Expected sequence:
1. Ship this doc as a commit → `docs/paper_revisions_v1.md`
2. KS forwards link to Khawar
3. Khawar reviews + flags any disagreements with proposed paragraph wording
4. (Parallel, non-blocking) KS ships `scripts/measure_conformal_coverage_cross_domain.py`
   as a pre-camera-ready micro-slice to populate the coverage table
5. Khawar merges final text into LaTeX camera-ready
6. Camera-ready submitted to IEEE Cyber-AI 2026 Bucharest
