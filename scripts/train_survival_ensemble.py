#!/usr/bin/env python3
"""
scripts/train_survival_ensemble.py — Slice 6 training harness.

Trains all 6 survival models on LeadForge (transformed to survival schema
via Decision B=3 synthetic time construction), computes σ_SHAP on hazard
outputs via rotated backgrounds, correlates with Slice 5.1 classification
σ_SHAP, writes docs/survival_training_report.md + figures.

Answers the three Slice 6 research questions:
    1. Does σ_SHAP transfer from classification to survival?
       -> Spearman correlation between classification σ_SHAP (from Slice 5.1
          attribution.json files) and survival σ_SHAP (computed here) across
          matched model pairs.
    2. Is Cox PH σ_SHAP lower than Weibull σ_SHAP?
       -> Direct comparison in the per-model σ_SHAP table.
    3. Which parametric distribution fits LeadForge best per AIC?
       -> AIC ranking of {weibull_aft, lognormal_aft, loglogistic_aft}.

Usage (from repo root):
    python scripts/train_survival_ensemble.py --k 10         # quick preview
    python scripts/train_survival_ensemble.py --k 50         # canonical
    python scripts/train_survival_ensemble.py --models cox_ph weibull_aft
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pinkmate.ml.survival import (  # noqa: E402
    MODEL_KEYS,
    build_survival_model,
    fit_model,
)
from pinkmate.ml.survival_attribution import (  # noqa: E402
    DEFAULT_TIME_GRID,
    compute_rotated_survshap_variance,
    spearman_transfer_correlation,
)
from pinkmate.ml.survival_data import (  # noqa: E402
    DEFAULT_MAX_FOLLOWUP_DAYS,
    DEFAULT_TOUCH_INTERVAL_DAYS,
    sweep_touch_interval_sensitivity,
    transform_leadforge_to_survival,
)


# Classification sigma_SHAP sources from Slice 5.1
CLASSIFICATION_SIGMA_SHAP_SOURCES: Dict[str, str] = {
    "leadforge_intermediate": "models/gtm/attribution.json",
    "leadforge_advanced": "models/gtm_advanced/attribution.json",
    "xeducation": "models/xeducation/attribution.json",
}

# Classification model key -> survival model key for transfer correlation
MODEL_FAMILY_PAIRS = [
    ("xgboost", "xgb_survival"),
    ("lightgbm", "lgb_aft"),
    ("random_forest", "cox_ph"),
    ("gradient_boosting", "weibull_aft"),
    ("logistic_regression", "lognormal_aft"),
    ("adaboost", "loglogistic_aft"),
]


def load_leadforge_dataframe(repo_root: Path) -> Optional[pd.DataFrame]:
    """Load LeadForge intermediate tier from Slice 5 cached data."""
    candidates = [
        repo_root / "models" / "gtm" / "train_dataframe.parquet",
        repo_root / "models" / "gtm" / "train_dataframe.csv",
        repo_root / "data" / "leadforge_intermediate.parquet",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)
    return None


def load_classification_sigma_shap(repo_root: Path) -> Dict[str, Dict[str, float]]:
    """Load σ_SHAP values from Slice 5.1 attribution.json files."""
    out: Dict[str, Dict[str, float]] = {}
    for dataset_key, rel_path in CLASSIFICATION_SIGMA_SHAP_SOURCES.items():
        p = repo_root / rel_path
        if not p.exists():
            continue
        with open(p) as f:
            attribution = json.load(f)
        per_model: Dict[str, float] = {}
        for container_key in ("with_smote", "without_smote", "models"):
            container = attribution.get(container_key, {})
            if isinstance(container, dict):
                for model_name, entry in container.items():
                    if isinstance(entry, dict):
                        sig = (
                            entry.get("sigma_shap")
                            or entry.get("sigma_shap_mean")
                            or entry.get("sigma")
                        )
                        if sig is not None:
                            per_model[model_name] = float(sig)
        if per_model:
            out[dataset_key] = per_model
    return out


def run_slice6(args) -> int:
    print(f"[Slice 6] Survival ensemble training, K={args.k}")

    # ---------- Load LeadForge ----------
    df = load_leadforge_dataframe(REPO_ROOT)
    if df is None:
        print("[Slice 6] LeadForge cached data not found under models/gtm/. "
              "Run scripts/train_gtm_ensemble.py first to populate it.")
        return 1
    print(f"[Slice 6] LeadForge loaded: n={len(df)}, cols={len(df.columns)}")

    # ---------- Transform to survival schema ----------
    sf = transform_leadforge_to_survival(
        df,
        max_followup_days=args.max_followup,
        touch_interval_days=args.touch_interval,
    )
    print(f"[Slice 6] {sf.summary()}")

    # ---------- Sensitivity sweep ----------
    sens = sweep_touch_interval_sensitivity(df, max_followup_days=args.max_followup)
    sens_summary = {ti: fr.censoring_rate for ti, fr in sens.items()}
    print(f"[Slice 6] touch_interval sensitivity (censoring rate): {sens_summary}")

    # ---------- Train selected models ----------
    model_keys = args.models if args.models else list(MODEL_KEYS)
    fit_results: Dict[str, object] = {}
    fitted_models: Dict[str, object] = {}
    models_dir = REPO_ROOT / "models" / "survival"
    models_dir.mkdir(parents=True, exist_ok=True)

    for key in model_keys:
        if key not in MODEL_KEYS:
            print(f"  [skip] unknown model '{key}'")
            continue
        print(f"  [{key}] fitting ...")
        try:
            m = build_survival_model(key)
            res = fit_model(m, sf.X, sf.time, sf.event)
            fit_results[key] = res
            fitted_models[key] = m
            print(f"  [{key}] c-index={res.concordance_index:.4f} "
                  f"logL={res.log_likelihood:.2f} AIC={res.aic:.2f} "
                  f"({res.fit_time_seconds:.1f}s)")
            with open(models_dir / f"{key}.pkl", "wb") as f:
                pickle.dump(m, f)
        except Exception as e:
            print(f"  [{key}] fit failed: {e}")

    if not fit_results:
        print("[Slice 6] No models successfully fit; aborting.")
        return 1

    # ---------- σ_SHAP on survival (rotated backgrounds) ----------
    reference_n = min(args.n_reference, len(sf.X))
    X_reference = sf.X.iloc[:reference_n]
    survival_sigma_shap: Dict[str, float] = {}
    attribution_results = {}
    for key, m in fitted_models.items():
        print(f"  [{key}] computing σ_SHAP with K={args.k} ...")
        try:
            attr = compute_rotated_survshap_variance(
                model=m,
                model_key=key,
                X_train=sf.X,
                X_reference=X_reference,
                feature_names=sf.feature_names,
                k_backgrounds=args.k,
                background_size=args.background_size,
                seed=args.seed,
            )
            survival_sigma_shap[key] = attr.sigma_shap_hazard
            attribution_results[key] = attr
            print(f"  [{key}] σ_SHAP_hazard = {attr.sigma_shap_hazard:.4f} "
                  f"(used_survshap={attr.extra.get('used_survshap_package')})")
        except Exception as e:
            print(f"  [{key}] attribution failed: {e}")

    # ---------- σ_SHAP transfer correlation ----------
    class_sigma_shap = load_classification_sigma_shap(REPO_ROOT)
    best_dataset = max(class_sigma_shap, key=lambda k: len(class_sigma_shap[k])) \
        if class_sigma_shap else None
    transfer = None
    if best_dataset:
        print(f"[Slice 6] Transfer correlation against classification σ_SHAP "
              f"from '{best_dataset}' ({len(class_sigma_shap[best_dataset])} models)")
        transfer = spearman_transfer_correlation(
            sigma_shap_classification=class_sigma_shap[best_dataset],
            sigma_shap_survival=survival_sigma_shap,
            model_pairs=MODEL_FAMILY_PAIRS,
            seed=args.seed,
        )
        rho, lo, hi, n = transfer
        print(f"[Slice 6] Spearman ρ = {rho:.4f} [95% CI: {lo:.4f}, {hi:.4f}], n={n}")

    # ---------- Figures ----------
    figs_dir = REPO_ROOT / "docs" / "figures" / "survival"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1: AIC comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    keys_sorted = sorted(fit_results, key=lambda k: fit_results[k].aic)
    aics = [fit_results[k].aic for k in keys_sorted]
    ax.barh(keys_sorted, aics, color="steelblue", edgecolor="black")
    ax.set_xlabel("AIC (lower = better fit)")
    ax.set_title("Slice 6 — Survival model AIC comparison on LeadForge")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(figs_dir / "aic_comparison.png", dpi=150)
    plt.close(fig)

    # Fig 2: σ_SHAP per model
    if survival_sigma_shap:
        fig, ax = plt.subplots(figsize=(8, 5))
        keys_sorted = sorted(survival_sigma_shap, key=survival_sigma_shap.get)
        sigs = [survival_sigma_shap[k] for k in keys_sorted]
        ax.barh(keys_sorted, sigs, color="darkorange", edgecolor="black")
        ax.set_xlabel(r"$\sigma_{\mathrm{SHAP\ hazard}}$ (lower = more stable)")
        ax.set_title("Slice 6 — σ_SHAP on hazard outputs per survival model")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(figs_dir / "sigma_shap_per_survival_model.png", dpi=150)
        plt.close(fig)

    # Fig 3: transfer scatter
    if transfer is not None and best_dataset is not None:
        xs, ys, labels = [], [], []
        for class_key, surv_key in MODEL_FAMILY_PAIRS:
            x = class_sigma_shap[best_dataset].get(class_key)
            y = survival_sigma_shap.get(surv_key)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
                labels.append(f"{class_key}\n→ {surv_key}")
        if xs:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(xs, ys, s=70, alpha=0.75, edgecolor="black")
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (xs[i], ys[i]), fontsize=7, alpha=0.7)
            ax.set_xlabel(r"Classification $\sigma_{\mathrm{SHAP}}$ (Slice 5.1)")
            ax.set_ylabel(r"Survival $\sigma_{\mathrm{SHAP\ hazard}}$ (Slice 6)")
            ax.set_title("Slice 6 — σ_SHAP transfer: classification vs survival")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(figs_dir / "sigma_shap_transfer_scatter.png", dpi=150)
            plt.close(fig)

    # ---------- Report ----------
    report_path = REPO_ROOT / "docs" / "survival_training_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Slice 6 — Survival analysis training report",
        "",
        "## Data transformation (Decision B=3)",
        "",
        f"- Max follow-up (right-censoring): {sf.max_followup_days} days",
        f"- Touch interval assumption: {sf.touch_interval_days} days per touch",
        f"- N samples: {sf.n_samples}, events: {sf.n_events}, censoring rate: {sf.censoring_rate:.2%}",
        "",
        "### Sensitivity of censoring rate to touch_interval_days",
        "",
        "| touch_interval_days | censoring rate |",
        "|---|---|",
    ]
    for ti, fr in sens.items():
        lines.append(f"| {ti} | {fr.censoring_rate:.2%} |")
    lines.extend([
        "",
        "> Honesty bar: touch_interval_days is a modelling assumption. ",
        "> Results should be read as 'σ_SHAP transfer under plausible synthetic ",
        "> time construction', not as time-to-conversion from directly ",
        "> observed first-touch timestamps.",
        "",
        "## Per-model fit statistics",
        "",
        "| Model | c-index | log-likelihood | AIC | n_events | fit (s) |",
        "|---|---|---|---|---|---|",
    ])
    for key in sorted(fit_results, key=lambda k: fit_results[k].aic):
        r = fit_results[key]
        lines.append(
            f"| `{key}` | {r.concordance_index:.4f} | {r.log_likelihood:.2f} "
            f"| {r.aic:.2f} | {r.n_events} | {r.fit_time_seconds:.1f} |"
        )
    lines.extend([
        "",
        "## σ_SHAP on hazard outputs (Slice 6 headline)",
        "",
        "| Model | σ_SHAP_hazard | K | background_size | time_grid |",
        "|---|---|---|---|---|",
    ])
    for key, attr in attribution_results.items():
        lines.append(
            f"| `{key}` | {attr.sigma_shap_hazard:.4f} | {attr.k_rotations} "
            f"| {attr.background_size} | {list(attr.time_grid)} |"
        )

    # Research-question verdicts
    lines.extend([
        "",
        "## Research-question verdicts",
        "",
        "### Q1 — Does σ_SHAP transfer from classification to survival?",
        "",
    ])
    if transfer is not None and not np.isnan(transfer[0]):
        rho, lo, hi, n = transfer
        lines.append(f"Spearman ρ(classification σ_SHAP, survival σ_SHAP_hazard) = **{rho:.4f}** (n={n}).")
        lines.append(f"95% bootstrap CI: [{lo:.4f}, {hi:.4f}].")
        lines.append("")
        if lo > 0:
            lines.append("**Verdict:** POSITIVE correlation (CI excludes 0) — σ_SHAP transfers across prediction-target regimes. Attribution-stability is a model-level property, not target-specific.")
        elif hi < 0:
            lines.append("**Verdict:** NEGATIVE correlation (CI excludes 0) — σ_SHAP does NOT transfer; classification and survival measure opposing attribution-stability behaviours.")
        else:
            lines.append("**Verdict:** No significant correlation (CI includes 0) — σ_SHAP does NOT transfer cleanly between classification and survival. Report both separately.")
    else:
        lines.append("Transfer correlation UNAVAILABLE — insufficient matched classification σ_SHAP pairs. Fix by training Slice 5.1 ensemble first.")

    # Q2 — Cox PH vs Weibull σ_SHAP
    lines.extend([
        "",
        "### Q2 — Is Cox PH σ_SHAP lower than Weibull σ_SHAP?",
        "",
    ])
    cox = survival_sigma_shap.get("cox_ph")
    weib = survival_sigma_shap.get("weibull_aft")
    if cox is not None and weib is not None:
        verdict = "YES" if cox < weib else "NO"
        lines.append(f"σ_SHAP(cox_ph) = {cox:.4f}; σ_SHAP(weibull_aft) = {weib:.4f}. **{verdict}**.")
        if cox < weib:
            lines.append("Semi-parametric Cox PH shows more stable attribution than parametric Weibull AFT on LeadForge — consistent with the principle that distributional assumption adds instability.")
        else:
            lines.append("Parametric Weibull AFT shows more stable attribution than semi-parametric Cox PH on this data — unexpected, worth investigating (may be a LeadForge-specific effect of the Decision B=3 synthetic time construction).")
    else:
        lines.append("INSUFFICIENT DATA — both cox_ph and weibull_aft σ_SHAP measurements required.")

    # Q3 — AIC ranking
    lines.extend([
        "",
        "### Q3 — Which parametric distribution fits LeadForge best?",
        "",
        "AIC ranking (lower is better):",
        "",
    ])
    parametric_aics = {
        k: fit_results[k].aic
        for k in ("weibull_aft", "lognormal_aft", "loglogistic_aft")
        if k in fit_results
    }
    for i, (k, aic) in enumerate(sorted(parametric_aics.items(), key=lambda kv: kv[1]), 1):
        lines.append(f"{i}. `{k}` — AIC = {aic:.2f}")
    if parametric_aics:
        best = min(parametric_aics, key=parametric_aics.get)
        lines.append("")
        lines.append(f"**Best parametric family on LeadForge: `{best}`** (lowest AIC).")

    lines.extend([
        "",
        "## Honesty bar",
        "",
        "- Synthetic days_since_first_touch construction (Decision B=3) is a modelling assumption. σ_SHAP transfer verdict is reported with that caveat.",
        "- Right-censoring at 90 days matches the LeadForge-intermediate 90-day sales cycle convention. Longer-horizon validation deferred to Slice 8+ when real time-to-conversion data becomes available.",
        "- XGBoost survival:cox and LightGBM AFT wrappers use library-default settings. Hyperparameter tuning deferred.",
        "- `exponential_aft` not included because lifelines >=0.30 does not ship an ExponentialAFTFitter; WeibullAFT's shape parameter approaches 1 under exponential-like data and is the AFT equivalent.",
        "",
        "## Figures",
        "",
        "- `docs/figures/survival/aic_comparison.png` — AIC bar chart across all 6 models.",
        "- `docs/figures/survival/sigma_shap_per_survival_model.png` — σ_SHAP_hazard per model.",
        "- `docs/figures/survival/sigma_shap_transfer_scatter.png` — classification vs survival σ_SHAP scatter.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Slice 6] report written to {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n-reference", type=int, default=200)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--max-followup", type=float, default=DEFAULT_MAX_FOLLOWUP_DAYS)
    parser.add_argument("--touch-interval", type=float, default=DEFAULT_TOUCH_INTERVAL_DAYS)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys to train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return run_slice6(args)


if __name__ == "__main__":
    sys.exit(main())
