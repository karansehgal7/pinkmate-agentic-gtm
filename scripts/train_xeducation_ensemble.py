"""
Slice 5.1 — train the six-model ensemble on X Education Lead Scoring v1.

What this script does:
    1. Loads X Education Lead Scoring from the Zenodo 23791277 CC BY 4.0
       mirror. Falls back to a synthetic X Education-shaped fixture if
       Zenodo is unreachable (CI / sandbox).
    2. Preprocesses per the paper's §VI.A methodology reused from Slice 2
       (median impute → standard scale → stratified 80/20 split, seed=42).
    3. Trains the six-model zoo (LR / RF / AdaBoost / XGBoost / LightGBM /
       CatBoost) on the SMOTE-oversampled training partition.
    4. Trains an XGBoost baseline WITHOUT SMOTE as the direction-test control.
    5. Measures classification metrics on the held-out test partition.
    6. Measures σ_SHAP (K rotated backgrounds × background size × N reference
       inputs, paper defaults 50/100/200) for XGBoost-with-SMOTE,
       XGBoost-without-SMOTE baseline, and LightGBM-with-SMOTE.
    7. Persists trained models + metrics + attribution to models/xeducation/.
    8. Writes a Markdown report (docs/xeducation_training_report.md) with
       committed PNG charts (docs/figures/xeducation/*.png).

**Real-data validation on X Education EdTech lead scoring.**
Slice 5.1 adds X Education as the first REAL-DATA validation point in the
PinkMate build programme. Combined with Taiwan Bankruptcy (3.23% minority,
financial-distress) and LeadForge intermediate (20.98% minority, B2B GTM
synthetic), the σ_SHAP attribution-stability directions are now tested
across the full 3-38% imbalance range and across three domain contexts.

**Verdict framing (locked BEFORE training):**
The paper's §VI.C SMOTE-direction observation is methodologically EXPECTED
to weaken at near-balanced classes — SMOTE has less synthetic positives to
generate as class balance approaches 50%. A DRIFT verdict on X Education
at 38.5% minority is a nuanced methodological-boundary finding, not a
failure. The model-stability ordering (LightGBM σ_SHAP < XGBoost σ_SHAP)
is architectural, dataset-agnostic, and expected PASS regardless of
imbalance.

**Methodology-integrity test:**
X Education's Tags column is a documented leakage trap that drives 0.98+
AUC in prior published work. We exclude Tags + Lead Quality + Lead Profile
+ Last Notable Activity as defensive leakage-trap exclusions. All ensemble
AUCs are expected in the 0.70-0.85 range once these are removed; any
model exceeding 0.95 fails the methodology-integrity test (enforced in
tests/test_xeducation_training.py).

Usage:
    python scripts/train_xeducation_ensemble.py              # Full K=50
    python scripts/train_xeducation_ensemble.py --fast       # K=10 quick
    python scripts/train_xeducation_ensemble.py --synthetic  # Hermetic
    python scripts/train_xeducation_ensemble.py --no-attribution

Reference:
    - Paper §VI.A: dataset + preprocessing methodology
    - Paper §VI.B: Table I metrics structure
    - Paper §VI.C, §V: σ_SHAP attribution stability
    - Aqil & Halim (2026): Zenodo 23791277 mirror of X Education
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pinkmate.ml.attribution import (
    DEFAULT_K_ROTATED_BACKGROUNDS,
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_N_REFERENCE_INPUTS,
    AttributionResult,
    compute_rotated_shap_variance,
)
from pinkmate.ml.dataset import preprocess
from pinkmate.ml.imbalance import apply_smote
from pinkmate.ml.model_zoo import MODEL_NAMES, TrainedZoo, build_model_zoo, train_zoo
from pinkmate.ml.pipeline import ModelMetrics, classification_metrics
from pinkmate.ml.x_education_loader import (
    XEDU_APPROX_MINORITY_RATE,
    XEDU_N_LEADS,
    XEDU_RANDOM_SEED,
    XEDU_TRAIN_SIZE,
    XEducationDataset,
    load_x_education_lead_scoring,
    make_synthetic_xeducation_like,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--fast", action="store_true",
                   help="K=10 quick-check instead of K=50 full run.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic X Education-shaped data (CI / no-Zenodo envs).")
    p.add_argument("--no-attribution", action="store_true",
                   help="Train models + metrics only, skip σ_SHAP measurement.")
    p.add_argument("--seed", type=int, default=XEDU_RANDOM_SEED,
                   help="Random seed. Default 42 (matches Slice 2 methodology).")
    p.add_argument("--models-dir", type=Path, default=Path("models/xeducation"),
                   help="Directory to persist trained models + summaries. "
                        "Default models/xeducation/.")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"),
                   help="Directory for the markdown report + figures/ subdirectory.")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip PNG chart generation (markdown only).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Artefact persistence
# ---------------------------------------------------------------------------

@dataclass
class SliceFiveOneArtefacts:
    """Everything produced by one X Education training run."""

    source: str
    seed: int
    minority_rate: float
    zoo: TrainedZoo
    metrics_with_smote: List[ModelMetrics]
    attributions_with_smote: Dict[str, AttributionResult]
    xgb_no_smote_attribution: Optional[AttributionResult]
    sha256: str

    def save(self, models_dir: Path) -> None:
        """Persist models + metrics + attribution to models/xeducation/."""
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.zoo.models.items():
            with open(models_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)

        with open(models_dir / "metrics.json", "w") as f:
            json.dump([m.as_dict() for m in self.metrics_with_smote], f, indent=2)

        attribution_payload = {
            "dataset": "x_education",
            "source": self.source,
            "seed": self.seed,
            "minority_rate": self.minority_rate,
            "sha256": self.sha256,
            "with_smote": {
                name: {
                    "sigma_shap": r.sigma_shap,
                    "mean_absolute_phi": r.mean_absolute_phi,
                    "k_backgrounds": r.k_backgrounds,
                    "background_size": r.background_size,
                    "n_reference": r.n_reference,
                }
                for name, r in self.attributions_with_smote.items()
            },
        }
        if self.xgb_no_smote_attribution is not None:
            r = self.xgb_no_smote_attribution
            attribution_payload["xgboost_no_smote_baseline"] = {
                "sigma_shap": r.sigma_shap,
                "mean_absolute_phi": r.mean_absolute_phi,
                "k_backgrounds": r.k_backgrounds,
                "background_size": r.background_size,
                "n_reference": r.n_reference,
            }
        with open(models_dir / "attribution.json", "w") as f:
            json.dump(attribution_payload, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline runner — parallels run_gtm_pipeline() for X Education domain
# ---------------------------------------------------------------------------

def run_xeducation_pipeline(
    source: str = "zenodo",
    seed: int = XEDU_RANDOM_SEED,
    k_backgrounds: int = DEFAULT_K_ROTATED_BACKGROUNDS,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    n_reference: int = DEFAULT_N_REFERENCE_INPUTS,
    include_xgb_no_smote_baseline: bool = True,
    skip_attribution: bool = False,
) -> SliceFiveOneArtefacts:
    """
    End-to-end X Education training pipeline:
        1. Load X Education Lead Scoring
        2. Preprocess (median impute → standard scale → stratified 80/20)
        3. SMOTE on training partition only
        4. Train six-model zoo
        5. Measure classification metrics
        6. Measure σ_SHAP for XGBoost+LightGBM with SMOTE
        7. (Optional) measure σ_SHAP for XGBoost without SMOTE baseline
    """
    if source == "synthetic":
        LOGGER.info("Loading synthetic X Education-shaped fixture")
        ds: XEducationDataset = make_synthetic_xeducation_like(seed=seed)
    else:
        ds = load_x_education_lead_scoring(fallback_to_synthetic=True)

    minority_rate = float(ds.y.mean())
    LOGGER.info(
        "X Education loaded from %s — %d rows, %d features, minority rate %.4f",
        ds.source, len(ds.X), ds.X.shape[1], minority_rate,
    )

    split = preprocess(ds.X, ds.y, seed=seed, train_size=XEDU_TRAIN_SIZE)

    X_tr_smote, y_tr_smote = apply_smote(
        split.X_train, split.y_train, seed=seed,
    )
    LOGGER.info(
        "SMOTE applied — %d → %d training rows (%.4f → 0.5 minority rate)",
        len(split.y_train), len(y_tr_smote),
        float(np.mean(split.y_train)),
    )

    zoo = train_zoo(X_tr_smote, y_tr_smote, seed=seed)
    metrics = classification_metrics(zoo, split.X_test, split.y_test)

    attributions_with_smote: Dict[str, AttributionResult] = {}
    xgb_no_smote_attribution: Optional[AttributionResult] = None

    if not skip_attribution:
        for model_name in ("xgboost", "lightgbm"):
            model = zoo.models.get(model_name)
            if model is None:
                LOGGER.warning("Model '%s' not trained; skipping σ_SHAP", model_name)
                continue
            LOGGER.info("Computing σ_SHAP for %s (with SMOTE)...", model_name)
            result = compute_rotated_shap_variance(
                model=model,
                model_name=model_name,
                X_train=X_tr_smote,
                X_reference=split.X_test[:n_reference],
                feature_names=split.feature_names,
                k_backgrounds=k_backgrounds,
                background_size=background_size,
                seed=seed,
            )
            attributions_with_smote[model_name] = result

        if include_xgb_no_smote_baseline:
            LOGGER.info("Training XGBoost baseline WITHOUT SMOTE for direction test...")
            baseline_specs = build_model_zoo(seed=seed)
            xgb_baseline = baseline_specs["xgboost"]
            xgb_baseline.fit(split.X_train, split.y_train)
            LOGGER.info("Computing σ_SHAP for XGBoost (without SMOTE baseline)...")
            xgb_no_smote_attribution = compute_rotated_shap_variance(
                model=xgb_baseline,
                model_name="xgboost_no_smote",
                X_train=split.X_train,
                X_reference=split.X_test[:n_reference],
                feature_names=split.feature_names,
                k_backgrounds=k_backgrounds,
                background_size=background_size,
                seed=seed,
            )

    return SliceFiveOneArtefacts(
        source=ds.source,
        seed=seed,
        minority_rate=minority_rate,
        zoo=zoo,
        metrics_with_smote=metrics,
        attributions_with_smote=attributions_with_smote,
        xgb_no_smote_attribution=xgb_no_smote_attribution,
        sha256=ds.sha256,
    )


# ---------------------------------------------------------------------------
# Verdict + reporting
# ---------------------------------------------------------------------------

def compute_verdict(art: SliceFiveOneArtefacts) -> Dict[str, str]:
    """
    Qualitative-direction verdict for X Education (same structure as Slice 5):
    - Does SMOTE INCREASE σ_SHAP on XGBoost (direction from Taiwan Slice 2)?
    - Is LightGBM σ_SHAP < XGBoost σ_SHAP with SMOTE?

    Reports PASS / DRIFT / NOT_RUN for each.
    """
    verdict = {}

    if art.xgb_no_smote_attribution is None or "xgboost" not in art.attributions_with_smote:
        verdict["smote_direction"] = "NOT_RUN"
    else:
        xgb_no = art.xgb_no_smote_attribution.sigma_shap
        xgb_yes = art.attributions_with_smote["xgboost"].sigma_shap
        if xgb_yes > xgb_no:
            verdict["smote_direction"] = (
                f"PASS — SMOTE increases σ_SHAP on XGBoost "
                f"({xgb_no:.4f} → {xgb_yes:.4f}, Δ=+{xgb_yes - xgb_no:.4f})"
            )
        else:
            verdict["smote_direction"] = (
                f"DRIFT — SMOTE did NOT increase σ_SHAP on XGBoost "
                f"({xgb_no:.4f} → {xgb_yes:.4f}, Δ={xgb_yes - xgb_no:.4f}). "
                "Methodologically EXPECTED at near-balanced classes — SMOTE "
                "has less headroom to destabilize attribution as class balance "
                "approaches 50%. Consistent with the paper's definition, NOT a "
                "failure."
            )

    xgb_s = art.attributions_with_smote.get("xgboost")
    lgb_s = art.attributions_with_smote.get("lightgbm")
    if xgb_s is None or lgb_s is None:
        verdict["model_stability_ordering"] = "NOT_RUN"
    else:
        if lgb_s.sigma_shap < xgb_s.sigma_shap:
            verdict["model_stability_ordering"] = (
                f"PASS — LightGBM σ_SHAP < XGBoost σ_SHAP "
                f"({lgb_s.sigma_shap:.4f} < {xgb_s.sigma_shap:.4f})"
            )
        else:
            verdict["model_stability_ordering"] = (
                f"DRIFT — LightGBM σ_SHAP >= XGBoost σ_SHAP "
                f"({lgb_s.sigma_shap:.4f} vs {xgb_s.sigma_shap:.4f}). "
                "Unexpected — ordering is architectural and dataset-agnostic; "
                "investigate hyperparameters or feature engineering."
            )

    return verdict


def render_markdown_report(
    art: SliceFiveOneArtefacts,
    verdict: Dict[str, str],
    include_charts: bool,
) -> str:
    """Render the Slice 5.1 X Education training report as Markdown."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")

    lines: List[str] = []
    lines.append("# PinkMate Slice 5.1 — X Education Training Report")
    lines.append("")
    lines.append(f"- **Generated**: {now}")
    lines.append(f"- **Dataset**: X Education Lead Scoring (Zenodo 23791277, CC BY 4.0)")
    lines.append(f"- **Source**: `{art.source}`")
    lines.append(f"- **Minority rate (Converted=1)**: {art.minority_rate:.4f}")
    lines.append(f"- **Seed**: {art.seed}")
    lines.append(f"- **Dataset SHA-256**: `{art.sha256}`")
    lines.append(f"- **Leakage columns excluded**: Tags, Lead Quality, Lead Profile, "
                 f"Last Notable Activity")
    lines.append(f"- **Near-zero-variance columns excluded**: Magazine, "
                 f"Receive More Updates About Our Courses, "
                 f"Update me on Supply Chain Content, Get updates on DM Content, "
                 f"I agree to pay the amount through cheque")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Methodology validation on real EdTech lead-scoring data")
    lines.append("")
    lines.append("X Education is real labelled EdTech B2B lead-scoring data")
    lines.append("(~38.54% conversion), mirrored from Kaggle to Zenodo under")
    lines.append("CC BY 4.0 (Aqil & Halim 2026). The Tags column — known to")
    lines.append("drive 0.98+ AUC inflation via sales-team post-hoc labels —")
    lines.append("is excluded along with Lead Quality, Lead Profile, and Last")
    lines.append("Notable Activity as defensive leakage-trap exclusions.")
    lines.append("")
    lines.append("Expected AUC range after leakage exclusion: **0.70-0.85**.")
    lines.append("Any model exceeding 0.95 fails the methodology-integrity test.")
    lines.append("")
    lines.append("**The σ_SHAP numbers below DO NOT constitute performance**")
    lines.append("**claims about PinkMate on real target-industry B2B GTM leads.**")
    lines.append("X Education is an EdTech lead-scoring dataset; real-data")
    lines.append("validation on a target-industry corpus remains deferred to")
    lines.append("later data-gated work.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Classification metrics (held-out test partition)")
    lines.append("")
    lines.append("| Model | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|-------|-----------|--------|-----|---------|")
    for m in art.metrics_with_smote:
        d = m.as_dict()
        lines.append(f"| {d['model']} | {d['precision']:.4f} | {d['recall']:.4f} | "
                     f"{d['f1']:.4f} | {d['roc_auc']:.4f} |")
    lines.append("")

    if include_charts:
        lines.append("![Classification metrics per model](figures/xeducation/classification_metrics.png)")
        lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 2. Attribution stability — σ_SHAP (paper §V)")
    lines.append("")
    if not art.attributions_with_smote:
        lines.append("*σ_SHAP measurement was skipped (--no-attribution).*")
    else:
        lines.append("| Model | Training regime | σ_SHAP | mean &#124;φ&#124; | K backgrounds | Background size | N reference |")
        lines.append("|-------|-----------------|--------|-------------------|---------------|-----------------|-------------|")
        for name, r in art.attributions_with_smote.items():
            lines.append(
                f"| {name} | with SMOTE | {r.sigma_shap:.4f} | {r.mean_absolute_phi:.4f} | "
                f"{r.k_backgrounds} | {r.background_size} | {r.n_reference} |"
            )
        if art.xgb_no_smote_attribution is not None:
            r = art.xgb_no_smote_attribution
            lines.append(
                f"| xgboost | **without SMOTE (baseline)** | {r.sigma_shap:.4f} | "
                f"{r.mean_absolute_phi:.4f} | {r.k_backgrounds} | {r.background_size} | "
                f"{r.n_reference} |"
            )
        lines.append("")

        if include_charts:
            lines.append("![σ_SHAP per model](figures/xeducation/sigma_shap_per_model.png)")
            lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## 3. Qualitative-direction verdict (vs Taiwan + LeadForge)")
    lines.append("")
    lines.append(f"**SMOTE direction on XGBoost (Taiwan expected: INCREASE)**")
    lines.append(f"→ {verdict.get('smote_direction', 'NOT_RUN')}")
    lines.append("")
    lines.append(f"**Model stability ordering (Taiwan expected: LightGBM < XGBoost)**")
    lines.append(f"→ {verdict.get('model_stability_ordering', 'NOT_RUN')}")
    lines.append("")
    lines.append("**Methodological context:**")
    lines.append("X Education at 38.54% minority sits at the near-balanced end of")
    lines.append("the imbalance spectrum. The paper's §VI.C SMOTE-direction")
    lines.append("observation is methodologically expected to WEAKEN at near-")
    lines.append("balanced classes because SMOTE has less synthetic-positive")
    lines.append("headroom to destabilize attribution. A DRIFT here is a nuanced")
    lines.append("methodological-boundary finding, consistent with the paper's")
    lines.append("definition. The model-stability ordering is architectural and")
    lines.append("expected PASS regardless of imbalance.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 4. Cross-dataset context (new in Slice 5.1)")
    lines.append("")
    lines.append("| Dataset | Minority | Domain | Source | smote_direction | model_stability_ordering |")
    lines.append("|---------|----------|--------|--------|-----------------|--------------------------|")
    lines.append("| Taiwan Bankruptcy | 3.23% | Financial distress | Real (UCI) | PASS (paper §VI.C) | PASS (paper §VI.C) |")
    lines.append("| LeadForge intermediate | 20.98% | B2B GTM | Synthetic (HF) | PASS (Slice 5, 42267e5) | PASS (Slice 5, 42267e5) |")
    smote_short = (verdict.get("smote_direction", "NOT_RUN").split("—")[0]).strip()
    ordering_short = (verdict.get("model_stability_ordering", "NOT_RUN").split("—")[0]).strip()
    lines.append(f"| X Education | {art.minority_rate:.2%} | EdTech lead scoring | Real (Zenodo) | {smote_short} (Slice 5.1) | {ordering_short} (Slice 5.1) |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 5. Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/train_xeducation_ensemble.py             # Full K=50 run (~5 min)")
    lines.append("python scripts/train_xeducation_ensemble.py --fast      # K=10 quick check (~2 min)")
    lines.append("python scripts/train_xeducation_ensemble.py --synthetic # Hermetic synthetic run")
    lines.append("```")
    lines.append("")
    lines.append("Trained pickles land under `models/xeducation/`. The ScoringAgent")
    lines.append("does NOT pick these up by default (ScoringAgent still defaults to")
    lines.append("`models/gtm/`); X Education models are a research artefact, not a")
    lines.append("deployment target.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PNG chart generation
# ---------------------------------------------------------------------------

def _save_chart_classification_metrics(
    art: SliceFiveOneArtefacts, path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m.model_name for m in art.metrics_with_smote]
    precisions = [m.precision for m in art.metrics_with_smote]
    recalls = [m.recall for m in art.metrics_with_smote]
    f1s = [m.f1 for m in art.metrics_with_smote]
    aucs = [m.roc_auc for m in art.metrics_with_smote]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    x = np.arange(len(names))
    width = 0.2
    ax.bar(x - 1.5 * width, precisions, width, label="Precision", color="#4a7fbf")
    ax.bar(x - 0.5 * width, recalls, width, label="Recall", color="#d94f4f")
    ax.bar(x + 0.5 * width, f1s, width, label="F1", color="#5aa374")
    ax.bar(x + 1.5 * width, aucs, width, label="ROC-AUC", color="#c08c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("PinkMate Slice 5.1 — X Education classification metrics")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_chart_sigma_shap_per_model(
    art: SliceFiveOneArtefacts, path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    if art.xgb_no_smote_attribution is not None:
        labels.append("xgboost\n(NO SMOTE)")
        values.append(art.xgb_no_smote_attribution.sigma_shap)
        colors.append("#c08c3c")
    for name, r in art.attributions_with_smote.items():
        labels.append(f"{name}\n(with SMOTE)")
        values.append(r.sigma_shap)
        colors.append("#4a7fbf" if name == "xgboost" else "#5aa374")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("σ_SHAP")
    k_bg = (art.attributions_with_smote.get("xgboost").k_backgrounds
            if "xgboost" in art.attributions_with_smote else "?")
    ax.set_title("PinkMate Slice 5.1 — σ_SHAP attribution stability per model\n"
                 f"(X Education, K={k_bg} backgrounds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if argv is not None:
        sys.argv = ["train_xeducation_ensemble.py"] + list(argv)
    args = parse_args()

    k = 10 if args.fast else DEFAULT_K_ROTATED_BACKGROUNDS
    source = "synthetic" if args.synthetic else "zenodo"

    print(f"PinkMate Slice 5.1 — X Education training")
    print(f"  Source: {source}")
    print(f"  K backgrounds: {k}")
    print(f"  Seed: {args.seed}")
    print(f"  Models dir: {args.models_dir}")
    print(f"  Docs dir: {args.docs_dir}")
    print(f"  Skip attribution: {args.no_attribution}")
    print()

    art = run_xeducation_pipeline(
        source=source,
        seed=args.seed,
        k_backgrounds=k,
        include_xgb_no_smote_baseline=True,
        skip_attribution=args.no_attribution,
    )

    print(f"\n--- Trained {len(art.zoo.models)} models, ")
    print(f"    Minority rate: {art.minority_rate:.4f}, "
          f"σ_SHAP measurements: {len(art.attributions_with_smote)}")

    art.save(args.models_dir)
    print(f"[artefacts] Saved to {args.models_dir.resolve()}")

    docs_dir: Path = args.docs_dir
    figures_dir = docs_dir / "figures" / "xeducation"
    docs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    include_charts = not args.no_charts and not args.no_attribution
    if include_charts:
        print("[charts] Generating PNGs...")
        _save_chart_classification_metrics(
            art, figures_dir / "classification_metrics.png"
        )
        _save_chart_sigma_shap_per_model(
            art, figures_dir / "sigma_shap_per_model.png"
        )
        print(f"[charts] Saved to {figures_dir}/")

    verdict = compute_verdict(art)
    report = render_markdown_report(art, verdict, include_charts)
    report_path = docs_dir / "xeducation_training_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[report] Wrote {report_path}")

    print()
    print(f"Verdict (qualitative direction):")
    for k_v, v in verdict.items():
        print(f"  {k_v}: {v}")

    # Methodology-integrity AUC guard (soft warning here; hard assert in tests)
    high_auc = [m for m in art.metrics_with_smote if m.roc_auc >= 0.95]
    if high_auc:
        print()
        print("[WARNING] Methodology-integrity concern:")
        for m in high_auc:
            print(f"  {m.model_name} ROC-AUC = {m.roc_auc:.4f} (>= 0.95) — "
                  f"may indicate a leakage column slipped through. Review "
                  f"pinkmate/ml/x_education_loader.py LEAKAGE_COLUMNS.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
