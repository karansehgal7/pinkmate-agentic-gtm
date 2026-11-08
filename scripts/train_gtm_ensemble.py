"""
Slice 5 — train the six-model GTM-domain ensemble on LeadForge Lead Scoring v1.

What this script does:
    1. Loads LeadForge Lead Scoring v1 (intermediate tier by default) from
       HuggingFace via `datasets.load_dataset()`. Falls back to a synthetic
       LeadForge-shaped fixture if HuggingFace is unreachable (CI / sandbox).
    2. Preprocesses per the paper's §VI.A methodology reused from Slice 2
       (median impute → standard scale → stratified 80/20 split, seed=42).
    3. Trains the six-model zoo (LR / RF / AdaBoost / XGBoost / LightGBM /
       CatBoost) on the SMOTE-oversampled training partition.
    4. Trains an XGBoost baseline WITHOUT SMOTE as the direction-test control
       for σ_SHAP's SMOTE-effect transferability (per Critique 1 from the
       Phase 1 close-out — Taiwan SMOTE increased σ_SHAP; we test whether
       the direction transfers to LeadForge's ~20% minority class).
    5. Measures classification metrics on the held-out test partition.
    6. Measures σ_SHAP (K rotated backgrounds × background size × N reference
       inputs, paper defaults 50/100/200) for XGBoost-with-SMOTE,
       XGBoost-without-SMOTE baseline, and LightGBM-with-SMOTE.
    7. Persists trained models + metrics + attribution to `models/gtm/`.
    8. Writes a markdown report (`docs/gtm_training_report.md`) with
       committed PNG charts (`docs/figures/gtm/*.png`) for in-browser review.

**Methodology validation on synthetic B2B-shaped data.**
The σ_SHAP numbers reported by this script and the PNG charts committed to
docs/figures/gtm/ are the output of applying the Q-MDP σ_SHAP governance
harness to B2B-shaped tabular data for the first time. **They DO NOT
constitute performance claims about PinkMate on real B2B GTM leads.**
Real-data validation is deferred to Slice 5.1+ once a labelled real
corpus is available (candidates: pink-mate.co.uk landing-page capture
over 6-12 months, academic partnership via Kent Business School, specific
industry engagement).

Usage:
    python scripts/train_gtm_ensemble.py                     # Full K=50 run (~10 min)
    python scripts/train_gtm_ensemble.py --fast              # K=10 quick-check (~2 min)
    python scripts/train_gtm_ensemble.py --synthetic         # Synthetic data (no HF access)
    python scripts/train_gtm_ensemble.py --no-attribution    # Skip σ_SHAP (train only)
    python scripts/train_gtm_ensemble.py --tier advanced     # Advanced difficulty (Slice 5.1 preview)

Reference:
    - Paper §VI.A: dataset + preprocessing methodology
    - Paper §VI.B: Table I metrics structure
    - Paper §VI.C, §V: σ_SHAP attribution stability
    - LeadForge dataset: https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1
    - LeadForge feature dictionary: https://leadforge-dev.github.io/leadforge/docs/dataset/features
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

# Allow invocation via "python scripts/train_gtm_ensemble.py" from repo root.
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
from pinkmate.ml.leadforge_loader import (
    LEADFORGE_RANDOM_SEED,
    LEADFORGE_TRAIN_SIZE,
    LEADFORGE_TIERS,
    LeadForgeDataset,
    load_leadforge_lead_scoring,
    make_synthetic_leadforge_like,
)
from pinkmate.ml.model_zoo import MODEL_NAMES, TrainedZoo, build_model_zoo, train_zoo
from pinkmate.ml.pipeline import ModelMetrics, classification_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--fast", action="store_true",
                   help="K=10 quick-check instead of K=50 full run.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic LeadForge-shaped data (for CI / no-HF envs).")
    p.add_argument("--no-attribution", action="store_true",
                   help="Train models + metrics only, skip σ_SHAP measurement.")
    p.add_argument("--tier", type=str, default="intermediate",
                   choices=list(LEADFORGE_TIERS),
                   help="LeadForge difficulty tier. Default 'intermediate'. "
                        "'advanced' is a Slice 5.1 preview.")
    p.add_argument("--seed", type=int, default=LEADFORGE_RANDOM_SEED,
                   help="Random seed. Default 42 (matches Slice 2 methodology).")
    p.add_argument("--models-dir", type=Path, default=Path("models/gtm"),
                   help="Directory to persist trained models + summaries. "
                        "Default models/gtm/ (separate from models/taiwan/).")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"),
                   help="Directory for the markdown report + figures/ subdirectory.")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip PNG chart generation (markdown only).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Artefact persistence (parallels SliceTwoArtefacts.save())
# ---------------------------------------------------------------------------

@dataclass
class SliceFiveArtefacts:
    """
    Everything produced by one training run — models, metrics, attributions.
    """

    tier: str
    source: str
    seed: int
    minority_rate: float
    zoo: TrainedZoo
    metrics_with_smote: List[ModelMetrics]
    attributions_with_smote: Dict[str, AttributionResult]
    xgb_no_smote_attribution: Optional[AttributionResult]
    sha256: str

    def save(self, models_dir: Path) -> None:
        """Persist models + metrics + attribution to models/gtm/."""
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.zoo.models.items():
            with open(models_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)

        with open(models_dir / "metrics.json", "w") as f:
            json.dump([m.as_dict() for m in self.metrics_with_smote], f, indent=2)

        attribution_payload = {
            "tier": self.tier,
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
# Pipeline runner — parallels run_slice_two_pipeline() for GTM domain
# ---------------------------------------------------------------------------

def run_gtm_pipeline(
    tier: str = "intermediate",
    source: str = "huggingface",
    seed: int = LEADFORGE_RANDOM_SEED,
    k_backgrounds: int = DEFAULT_K_ROTATED_BACKGROUNDS,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    n_reference: int = DEFAULT_N_REFERENCE_INPUTS,
    include_xgb_no_smote_baseline: bool = True,
    skip_attribution: bool = False,
) -> SliceFiveArtefacts:
    """
    End-to-end GTM training pipeline:
        1. Load LeadForge Lead Scoring v1
        2. Preprocess (median impute → standard scale → stratified 80/20)
        3. SMOTE on training partition only
        4. Train six-model zoo
        5. Measure classification metrics
        6. Measure σ_SHAP for XGBoost+LightGBM with SMOTE
        7. (Optional) measure σ_SHAP for XGBoost without SMOTE baseline
    """
    # Step 1 — load
    if source == "synthetic":
        LOGGER.info("Loading synthetic LeadForge-shaped fixture (no HF access)")
        ds = make_synthetic_leadforge_like(seed=seed, tier=tier)
    else:
        ds = load_leadforge_lead_scoring(
            tier=tier,
            fallback_to_synthetic=True,
        )

    minority_rate = float(ds.y.mean())
    LOGGER.info(
        "LeadForge '%s' loaded from %s — %d rows, %d features, "
        "minority rate %.4f",
        ds.tier, ds.source, len(ds.X), ds.X.shape[1], minority_rate,
    )

    # Step 2 — preprocess (reuse Slice 2's preprocess())
    split = preprocess(ds.X, ds.y, seed=seed, train_size=LEADFORGE_TRAIN_SIZE)

    # Step 3 — SMOTE on training partition only
    X_tr_smote, y_tr_smote = apply_smote(
        split.X_train, split.y_train, seed=seed,
    )
    LOGGER.info(
        "SMOTE applied — %d → %d training rows (%.4f → 0.5 minority rate)",
        len(split.y_train), len(y_tr_smote),
        float(np.mean(split.y_train)),
    )

    # Step 4 — train six-model zoo (with SMOTE)
    zoo = train_zoo(X_tr_smote, y_tr_smote, seed=seed)

    # Step 5 — classification metrics
    metrics = classification_metrics(zoo, split.X_test, split.y_test)

    attributions_with_smote: Dict[str, AttributionResult] = {}
    xgb_no_smote_attribution: Optional[AttributionResult] = None

    if not skip_attribution:
        # Step 6 — σ_SHAP for XGBoost + LightGBM with SMOTE
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

        # Step 7 — XGBoost without SMOTE baseline (direction-test control).
        # Reuses build_model_zoo() to construct an XGBoost estimator with
        # the same hyper-parameters, then fits it on the raw (un-SMOTE'd)
        # training partition.
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

    return SliceFiveArtefacts(
        tier=ds.tier,
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
# Verdict + reporting (qualitative-direction, not magnitude-tolerance)
# ---------------------------------------------------------------------------

def compute_verdict(art: SliceFiveArtefacts) -> Dict[str, str]:
    """
    Qualitative-direction verdict (per Phase 1 Critique 3):
    - Does SMOTE INCREASE σ_SHAP on XGBoost (direction from Taiwan Slice 2)?
    - Is LightGBM σ_SHAP < XGBoost σ_SHAP with SMOTE (direction from Taiwan)?

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
                "Expected on Taiwan; may not transfer to LeadForge's milder imbalance."
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
                "Expected ordering from Taiwan; may not transfer."
            )

    return verdict


def render_markdown_report(
    art: SliceFiveArtefacts,
    verdict: Dict[str, str],
    include_charts: bool,
) -> str:
    """Render the Slice 5 training report as a Markdown document."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")

    lines: List[str] = []
    lines.append("# PinkMate Slice 5 — GTM-Domain Training Report")
    lines.append("")
    lines.append(f"- **Generated**: {now}")
    lines.append(f"- **Dataset**: LeadForge Lead Scoring v1 — tier `{art.tier}`")
    lines.append(f"- **Source**: `{art.source}`")
    lines.append(f"- **Rows loaded**: {len(art.zoo.models.get(MODEL_NAMES[0]).classes_) if hasattr(art.zoo.models.get(MODEL_NAMES[0]), 'classes_') else 'N/A'} classes; test-partition size fixed by seed=42 split")
    lines.append(f"- **Minority rate (converted_within_90_days)**: {art.minority_rate:.4f}")
    lines.append(f"- **Seed**: {art.seed}")
    lines.append(f"- **Dataset SHA-256**: `{art.sha256}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Methodology validation on synthetic B2B-shaped data")
    lines.append("")
    lines.append("The σ_SHAP numbers below and the PNG charts under")
    lines.append("`docs/figures/gtm/` are the output of applying the Q-MDP")
    lines.append("σ_SHAP governance harness to B2B-shaped tabular data for")
    lines.append("the first time. **They DO NOT constitute performance claims**")
    lines.append("**about PinkMate on real B2B GTM leads.** Real-data validation")
    lines.append("is deferred to Slice 5.1+ once a labelled real corpus is")
    lines.append("available.")
    lines.append("")
    lines.append("This honesty bar mirrors Slice 2's Taiwan Bankruptcy framing —")
    lines.append("the paper validates the framework on a declared benchmark, it")
    lines.append("does not claim the trained models should be used for other")
    lines.append("domains without retraining.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table I-style classification metrics
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
        lines.append("![Classification metrics per model](figures/gtm/classification_metrics.png)")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Attribution stability (σ_SHAP)
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
            lines.append("![σ_SHAP per model](figures/gtm/sigma_shap_per_model.png)")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Qualitative-direction verdict
    lines.append("## 3. Qualitative-direction verdict (vs Taiwan Slice 2)")
    lines.append("")
    lines.append("Per Phase 1 Critique 3 (close-out of Slice 5 scope planning), the")
    lines.append("verdict on LeadForge is **qualitative-direction**, not")
    lines.append("magnitude-tolerance. Taiwan Slice 2's two §VI.C qualitative")
    lines.append("observations are tested for transferability:")
    lines.append("")
    lines.append(f"**SMOTE direction on XGBoost (Taiwan expected: INCREASE)**")
    lines.append(f"→ {verdict.get('smote_direction', 'NOT_RUN')}")
    lines.append("")
    lines.append(f"**Model stability ordering (Taiwan expected: LightGBM < XGBoost)**")
    lines.append(f"→ {verdict.get('model_stability_ordering', 'NOT_RUN')}")
    lines.append("")
    lines.append("DRIFT verdicts on LeadForge are **expected possible outcomes**,")
    lines.append("not failures — LeadForge's ~20% minority class is less imbalanced")
    lines.append("than Taiwan's 3.23%, so SMOTE's effect on σ_SHAP has less")
    lines.append("headroom to move in the same direction with the same magnitude.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Reproducing
    lines.append("## 4. Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/train_gtm_ensemble.py             # Full K=50 run (~10 min)")
    lines.append("python scripts/train_gtm_ensemble.py --fast      # K=10 quick check (~2 min)")
    lines.append("python scripts/train_gtm_ensemble.py --synthetic # Hermetic synthetic run")
    lines.append("python scripts/train_gtm_ensemble.py --tier advanced  # Slice 5.1 preview")
    lines.append("```")
    lines.append("")
    lines.append("Trained pickles land under `models/gtm/`. The ScoringAgent default")
    lines.append("(`model_source=\"zoo\"`, `models_dir=\"models/gtm\"`) picks them up")
    lines.append("automatically on next construction.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PNG chart generation
# ---------------------------------------------------------------------------

def _save_chart_classification_metrics(
    art: SliceFiveArtefacts, path: Path,
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
    ax.bar(x - 1.5*width, precisions, width, label="Precision", color="#4a7fbf")
    ax.bar(x - 0.5*width, recalls, width, label="Recall", color="#d94f4f")
    ax.bar(x + 0.5*width, f1s, width, label="F1", color="#5aa374")
    ax.bar(x + 1.5*width, aucs, width, label="ROC-AUC", color="#c08c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"PinkMate Slice 5 — GTM-domain classification metrics "
                 f"(LeadForge {art.tier})")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_chart_sigma_shap_per_model(
    art: SliceFiveArtefacts, path: Path,
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
    ax.set_title(f"PinkMate Slice 5 — σ_SHAP attribution stability per model\n"
                 f"(LeadForge {art.tier}, K={art.attributions_with_smote.get('xgboost').k_backgrounds if 'xgboost' in art.attributions_with_smote else '?'} "
                 f"backgrounds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if argv is not None:
        # For test invocation with explicit argv
        sys.argv = ["train_gtm_ensemble.py"] + list(argv)
    args = parse_args()

    k = 10 if args.fast else DEFAULT_K_ROTATED_BACKGROUNDS
    source = "synthetic" if args.synthetic else "huggingface"

    print(f"PinkMate Slice 5 — GTM-domain training")
    print(f"  Tier: {args.tier}")
    print(f"  Source: {source}")
    print(f"  K backgrounds: {k}")
    print(f"  Seed: {args.seed}")
    print(f"  Models dir: {args.models_dir}")
    print(f"  Docs dir: {args.docs_dir}")
    print(f"  Skip attribution: {args.no_attribution}")
    print()

    art = run_gtm_pipeline(
        tier=args.tier,
        source=source,
        seed=args.seed,
        k_backgrounds=k,
        include_xgb_no_smote_baseline=True,
        skip_attribution=args.no_attribution,
    )

    print(f"\n--- Trained {len(art.zoo.models)} models, ")
    print(f"    Minority rate: {art.minority_rate:.4f}, "
          f"σ_SHAP measurements: {len(art.attributions_with_smote)}")

    # Save pickled models + json summaries
    art.save(args.models_dir)
    print(f"[artefacts] Saved to {args.models_dir.resolve()}")

    # Generate markdown report + optional PNG charts
    docs_dir: Path = args.docs_dir
    figures_dir = docs_dir / "figures" / "gtm"
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
    report_path = docs_dir / "gtm_training_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[report] Wrote {report_path}")

    print()
    print(f"Verdict (qualitative direction):")
    for k_v, v in verdict.items():
        print(f"  {k_v}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
