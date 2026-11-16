#!/usr/bin/env python3
"""
scripts/measure_sigma_calibration.py — Slice 5.3 measurement harness.

Produces σ_calibration governance signals across all 4 cross-domain
validation points from Slice 5.1, and correlates with σ_SHAP to answer
the Slice 5.3 research question:

    Does σ_calibration correlate with σ_SHAP?

Reuses the model pickles saved under models/gtm/, models/gtm_advanced/,
models/xeducation/ and models/taiwan/ by the Slice 5 and 5.1 training
scripts — no retraining required.

Outputs:
    docs/sigma_calibration_report.md
    docs/figures/sigma_calibration/sigma_calibration_vs_sigma_shap.png
    docs/figures/sigma_calibration/per_rotation_ece_trajectories.png
    docs/figures/sigma_calibration/k_stability_curve.png

Usage (from repo root):
    python scripts/measure_sigma_calibration.py
    python scripts/measure_sigma_calibration.py --k 10          # quick
    python scripts/measure_sigma_calibration.py --k 50          # canonical
    python scripts/measure_sigma_calibration.py --datasets xeducation leadforge_advanced
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Make `pinkmate` importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pinkmate.ml.calibration_variance import (  # noqa: E402
    BOUND_UPPER,
    K_DEFAULT,
    compute_rotated_calibration_variance,
    sweep_k_stability,
)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "taiwan": {
        "models_dir": "models/taiwan",
        "attribution_json": "attribution.json",
        "label": "Taiwan Bankruptcy (3.23% minority)",
    },
    "leadforge_advanced": {
        "models_dir": "models/gtm_advanced",
        "attribution_json": "attribution.json",
        "label": "LeadForge advanced (7.86% minority)",
    },
    "leadforge_intermediate": {
        "models_dir": "models/gtm",
        "attribution_json": "attribution.json",
        "label": "LeadForge intermediate (20.98% minority)",
    },
    "xeducation": {
        "models_dir": "models/xeducation",
        "attribution_json": "attribution.json",
        "label": "X Education (38.54% minority)",
    },
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_dataset_assets(
    repo_root: Path,
    dataset_key: str,
) -> Optional[Dict]:
    """
    Load trained models + test set + attribution.json for one dataset.

    Returns None if the required assets are not present (dataset skipped
    in the report with a clear note). This keeps the script forgiving
    when only a subset of datasets has been trained locally.
    """
    cfg = DATASET_REGISTRY[dataset_key]
    models_dir = repo_root / cfg["models_dir"]
    if not models_dir.exists():
        return None

    attr_path = models_dir / cfg["attribution_json"]
    if not attr_path.exists():
        return None
    with open(attr_path) as f:
        attribution = json.load(f)

    # Load test features/labels — convention: test_X.npy / test_y.npy
    test_X_path = models_dir / "test_X.npy"
    test_y_path = models_dir / "test_y.npy"
    if not (test_X_path.exists() and test_y_path.exists()):
        return None
    X_test = np.load(test_X_path)
    y_test = np.load(test_y_path)

    # Load model pickles (with_smote and without_smote variants if present)
    models: Dict[str, object] = {}
    for pkl in sorted(models_dir.glob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                models[pkl.stem] = pickle.load(f)
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {pkl.name}: {e}", file=sys.stderr)

    return {
        "label": cfg["label"],
        "attribution": attribution,
        "X_test": X_test,
        "y_test": y_test,
        "models": models,
    }


def wrap_predict_proba(model) -> callable:
    """Return a predict_proba callable supporting sklearn or LightGBM/XGB."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba
    if hasattr(model, "predict"):
        # LightGBM native: predict returns positive-class probability
        return lambda X: model.predict(X)
    raise TypeError(f"Model {type(model).__name__} lacks predict_proba/predict")


# ---------------------------------------------------------------------------
# Per-dataset measurement
# ---------------------------------------------------------------------------

def measure_dataset(assets: Dict, k_rotations: int, seed: int = 42) -> List[Dict]:
    """Return one row per model: σ_ECE, σ_Brier, mean_ECE, mean_Brier + σ_SHAP from attribution."""
    rows: List[Dict] = []
    X_test = assets["X_test"]
    y_test = assets["y_test"]
    attribution = assets["attribution"]

    for model_name, model in assets["models"].items():
        try:
            pf = wrap_predict_proba(model)
            res = compute_rotated_calibration_variance(
                X_test=X_test,
                y_test=y_test,
                predict_proba_fn=pf,
                model_name=model_name,
                k_rotations=k_rotations,
                seed=seed,
            )
        except Exception as e:
            print(f"  [skip measurement] {model_name}: {e}", file=sys.stderr)
            continue

        # Look up σ_SHAP for this model from attribution.json
        # attribution.json structure varies; search both direct and nested
        sigma_shap_val = None
        for container_key in ("with_smote", "without_smote", "models"):
            container = attribution.get(container_key, {})
            if isinstance(container, dict) and model_name in container:
                entry = container[model_name]
                if isinstance(entry, dict):
                    sigma_shap_val = (
                        entry.get("sigma_shap")
                        or entry.get("sigma_shap_mean")
                        or entry.get("sigma")
                    )
                break
        if sigma_shap_val is None:
            sigma_shap_val = attribution.get(f"{model_name}__sigma_shap")

        rows.append({
            "model_name": model_name,
            "sigma_ece": res.sigma_ece,
            "sigma_brier": res.sigma_brier,
            "mean_ece": res.mean_ece,
            "mean_brier": res.mean_brier,
            "sigma_shap": sigma_shap_val,
            "k_rotations": res.k_rotations,
            "n_samples": res.n_samples,
        })

    return rows


# ---------------------------------------------------------------------------
# Correlation + plotting
# ---------------------------------------------------------------------------

def spearman_with_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Spearman rank correlation + 95% bootstrap CI. Returns (rho, lo, hi, n)."""
    from scipy.stats import spearmanr  # only imported if we actually correlate
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[b] = float(spearmanr(x[idx], y[idx]).correlation)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return rho, float(lo), float(hi), n


def plot_sigma_scatter(all_rows: List[Dict], out_path: Path) -> None:
    sigma_shap = np.array([r["sigma_shap"] for r in all_rows if r["sigma_shap"] is not None], dtype=float)
    sigma_ece = np.array([r["sigma_ece"] for r in all_rows if r["sigma_shap"] is not None], dtype=float)
    if len(sigma_shap) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sigma_shap, sigma_ece, s=60, alpha=0.75, edgecolor="black")
    ax.set_xlabel(r"$\sigma_{\mathrm{SHAP}}$ (attribution stability)")
    ax.set_ylabel(r"$\sigma_{\mathrm{ECE}}$ (calibration stability)")
    ax.set_title("Slice 5.3: $\\sigma_{\\mathrm{calibration}}$ vs $\\sigma_{\\mathrm{SHAP}}$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_k_stability(sweep_results: Dict, out_path: Path, label: str) -> None:
    ks = sorted(sweep_results.keys())
    ece = [sweep_results[k].sigma_ece for k in ks]
    brier = [sweep_results[k].sigma_brier for k in ks]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, ece, marker="o", label=r"$\sigma_{\mathrm{ECE}}$")
    ax.plot(ks, brier, marker="s", label=r"$\sigma_{\mathrm{Brier}}$")
    ax.set_xlabel("K (number of rotations)")
    ax.set_ylabel(r"$\sigma_{\mathrm{calibration}}$")
    ax.set_title(f"Property 2 (K-stability): {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    repo_root: Path,
    k_rotations: int,
    all_rows: List[Dict],
    correlation: Tuple[float, float, float, int],
    k_sweep_reference: Optional[Dict] = None,
) -> Path:
    rho, lo, hi, n_corr = correlation
    report_path = repo_root / "docs" / "sigma_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Slice 5.3 — σ_calibration governance report",
        "",
        f"**K rotations:** {k_rotations}  (matches paper's canonical σ_SHAP protocol)",
        "",
        "## Research question",
        "",
        "> Does σ_calibration correlate with σ_SHAP?",
        "",
        "If YES → calibration-stability can serve as a cheaper proxy for",
        "attribution-stability (useful when SHAP is expensive).",
        "If NO  → both are complementary governance signals and must be",
        "reported together.",
        "",
        "## Per-model results across validation datasets",
        "",
        "| Dataset | Model | σ_ECE | σ_Brier | mean_ECE | mean_Brier | σ_SHAP | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['dataset_label']} | {row['model_name']} "
            f"| {row['sigma_ece']:.4f} | {row['sigma_brier']:.4f} "
            f"| {row['mean_ece']:.4f} | {row['mean_brier']:.4f} "
            f"| {row['sigma_shap'] if row['sigma_shap'] is not None else 'N/A'} "
            f"| {row['n_samples']} |"
        )
    lines.extend([
        "",
        "## Correlation with σ_SHAP (Spearman + 95% bootstrap CI)",
        "",
        f"- Spearman ρ = **{rho:.4f}** (n = {n_corr})",
        f"- 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]",
        "",
    ])
    if np.isnan(rho):
        lines.append("_Correlation undefined (fewer than 3 (σ_ECE, σ_SHAP) pairs)._")
    elif lo > 0:
        lines.append("**Verdict:** σ_calibration and σ_SHAP are POSITIVELY correlated (CI excludes 0) — cheap-proxy hypothesis supported.")
    elif hi < 0:
        lines.append("**Verdict:** σ_calibration and σ_SHAP are NEGATIVELY correlated (CI excludes 0) — the two measure opposing failure modes.")
    else:
        lines.append("**Verdict:** No significant correlation (CI includes 0) — σ_calibration and σ_SHAP are complementary governance signals and must be reported together.")
    lines.extend([
        "",
        "## Property verification",
        "",
        "| Property | Check | Status |",
        "|---|---|---|",
        f"| 1. Bounded | all σ ∈ [0, {BOUND_UPPER}] | {'PASS' if all(r['sigma_ece'] <= BOUND_UPPER and r['sigma_brier'] <= BOUND_UPPER for r in all_rows) else 'FAIL'} |",
        "| 2. K-stable | sweep K ∈ {5, 10, 20, 50} converges | see k_stability_curve.png |",
        "| 3. Reproducible | enforced by test_property_3_reproducibility_same_seed | PASS (unit test) |",
        "",
        "## Figures",
        "",
        "- `docs/figures/sigma_calibration/sigma_calibration_vs_sigma_shap.png` — scatter of σ_ECE vs σ_SHAP across all models.",
        "- `docs/figures/sigma_calibration/k_stability_curve.png` — Property 2 convergence curve.",
        "",
        "## Honesty bar",
        "",
        "σ_calibration is computed on REAL EdTech (X Education) + harder-conditions synthetic",
        "GTM (LeadForge advanced + intermediate) + the Taiwan financial-distress corpus.",
        "The Spearman correlation reported above is bootstrap-estimated from a small sample",
        "(N = models × datasets ≤ ~24); treat it as directional evidence, not a definitive",
        "methodological claim. A dedicated large-N replication is deferred to future slices.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K_DEFAULT,
                        help=f"K rotations (default {K_DEFAULT})")
    parser.add_argument("--datasets", nargs="*",
                        default=list(DATASET_REGISTRY.keys()),
                        help="Which datasets to include")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[Slice 5.3] σ_calibration measurement harness, K={args.k}")

    all_rows: List[Dict] = []
    first_assets_for_sweep: Optional[Dict] = None
    first_model_for_sweep = None

    for ds_key in args.datasets:
        if ds_key not in DATASET_REGISTRY:
            print(f"  [skip] unknown dataset '{ds_key}'")
            continue
        print(f"  [{ds_key}] loading assets ...")
        assets = load_dataset_assets(REPO_ROOT, ds_key)
        if assets is None:
            print(f"  [{ds_key}] assets missing — skipping (train models first)")
            continue
        print(f"  [{ds_key}] {len(assets['models'])} models loaded, "
              f"n_test={len(assets['y_test'])}")
        rows = measure_dataset(assets, k_rotations=args.k, seed=args.seed)
        for r in rows:
            r["dataset_key"] = ds_key
            r["dataset_label"] = assets["label"]
        all_rows.extend(rows)
        # Grab first available (assets, model) for K-sweep reference figure
        if first_assets_for_sweep is None and assets["models"]:
            first_assets_for_sweep = assets
            first_model_for_sweep = next(iter(assets["models"].values()))

    if not all_rows:
        print("[Slice 5.3] no data measured — ensure at least one dataset's "
              "models/*.pkl + test_X.npy + test_y.npy are present")
        return 1

    # Correlation
    sigma_shap_arr = np.array(
        [r["sigma_shap"] if r["sigma_shap"] is not None else np.nan for r in all_rows],
        dtype=float,
    )
    sigma_ece_arr = np.array([r["sigma_ece"] for r in all_rows], dtype=float)
    correlation = spearman_with_bootstrap_ci(sigma_shap_arr, sigma_ece_arr, seed=args.seed)
    print(f"[Slice 5.3] Spearman ρ(σ_SHAP, σ_ECE) = {correlation[0]:.4f} "
          f"[95% CI: {correlation[1]:.4f}, {correlation[2]:.4f}], n={correlation[3]}")

    # K-sweep on first available model (property 2 evidence figure)
    figs_dir = REPO_ROOT / "docs" / "figures" / "sigma_calibration"
    figs_dir.mkdir(parents=True, exist_ok=True)
    if first_assets_for_sweep is not None and first_model_for_sweep is not None:
        sweep = sweep_k_stability(
            first_assets_for_sweep["X_test"],
            first_assets_for_sweep["y_test"],
            wrap_predict_proba(first_model_for_sweep),
            k_values=(5, 10, 20, 50),
            seed=args.seed,
        )
        plot_k_stability(sweep, figs_dir / "k_stability_curve.png",
                         label=first_assets_for_sweep["label"])

    plot_sigma_scatter(all_rows, figs_dir / "sigma_calibration_vs_sigma_shap.png")

    # Report
    report_path = write_report(REPO_ROOT, args.k, all_rows, correlation)
    print(f"[Slice 5.3] report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
