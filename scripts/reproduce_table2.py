"""
Standalone reproduction of Table I + Table II from the Q-MDP paper
(Sehgal & Bhatti, IEEE Cyber-AI 2026, Submission #4554).

What this script does:
    1. Fetches the Taiwan Bankruptcy Prediction dataset (UCI 572) via
       `ucimlrepo` and caches it under `data/taiwan/`.
    2. Preprocesses per paper §VI.A (median impute → standard scale →
       stratified 80/20 split at seed 42).
    3. Trains the six-model zoo on the SMOTE-oversampled training partition.
    4. Measures Table I classification metrics on the held-out test partition.
    5. Measures σ_SHAP per §V (K=50 rotated backgrounds × size 100,
       on 200 reference inputs from the test partition) for XGBoost-with-SMOTE
       and LightGBM-with-SMOTE.
    6. Also trains an XGBoost baseline WITHOUT SMOTE and measures its σ_SHAP
       for the Table II row 1 baseline.
    7. Persists trained models + metrics + attribution summary to `models/`.
    8. Prints a side-by-side comparison of the reproduced numbers against
       the paper's declared Table II values, with tolerance thresholds.

Usage:
    python scripts/reproduce_table2.py                  # Full K=50 run (~10 min)
    python scripts/reproduce_table2.py --fast           # K=10 quick-check (~2 min)
    python scripts/reproduce_table2.py --synthetic      # Synthetic data (no UCI access)
    python scripts/reproduce_table2.py --no-attribution # Skip Table II, train only

Reference:
    - Paper §VI.A: dataset + preprocessing
    - Paper §VI.B: Table I values
    - Paper §VI.C: Table II values
    - Paper §VIII: reproducibility statement
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

# Allow invocation via "python scripts/reproduce_table2.py" from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pinkmate.ml.pipeline import SliceTwoArtefacts, run_slice_two_pipeline


# ---------------------------------------------------------------------------
# Paper-declared Table II values (per §VI.C)
# ---------------------------------------------------------------------------

PAPER_TABLE2: Dict[str, Dict[str, float]] = {
    "xgboost_no_smote": {
        "recall": None,          # Not reported (near zero on untreated imbalance)
        "sigma_shap": 0.0063,
        "mean_abs_phi": 0.092,
    },
    "xgboost_with_smote": {
        "recall": 0.52,
        "sigma_shap": 0.0214,
        "mean_abs_phi": 0.159,
    },
    "lightgbm_with_smote": {
        "recall": 0.52,
        "sigma_shap": 0.0101,
        "mean_abs_phi": 0.083,
    },
}


# ---------------------------------------------------------------------------
# Tolerance (per Slice 2 design decision 1, agreed with KS 2026-10-27)
# ---------------------------------------------------------------------------

SIGMA_SHAP_TOLERANCE: float = 0.005
RECALL_TOLERANCE: float = 0.03


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--fast", action="store_true",
                   help="K=10 quick-check instead of K=50 full run.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic Taiwan-shaped data (for CI / no-UCI envs).")
    p.add_argument("--no-attribution", action="store_true",
                   help="Train models + metrics only, skip σ_SHAP measurement.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Paper default 42.")
    p.add_argument("--models-dir", type=Path, default=Path("models"),
                   help="Directory to persist trained models + summaries.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_metrics(artefacts: SliceTwoArtefacts) -> None:
    print_banner("Table I — Classification metrics (SMOTE applied, 0.5 threshold)")
    print(f"{'Model':<22s}{'Precision':>12s}{'Recall':>10s}{'F1-Score':>12s}{'ROC-AUC':>10s}")
    print("-" * 66)
    for m in artefacts.metrics_with_smote:
        d = m.as_dict()
        print(f"{d['model']:<22s}{d['precision']:>12.3f}{d['recall']:>10.3f}"
              f"{d['f1']:>12.3f}{d['roc_auc']:>10.3f}")


def print_table2_comparison(artefacts: SliceTwoArtefacts) -> int:
    """Print the Table II reproduction side-by-side. Returns 0 if within tolerance, 1 otherwise."""
    print_banner("Table II — σ_SHAP attribution-stability trade-off")

    header = (
        f"{'Configuration':<28s}{'Recall':>8s}  "
        f"{'σ_SHAP (ours)':>14s}  {'σ_SHAP (paper)':>15s}  "
        f"{'mean|φ| (ours)':>15s}  {'mean|φ| (paper)':>16s}  Status"
    )
    print(header)
    print("-" * len(header))

    status_code = 0
    rows = []

    if artefacts.xgb_no_smote_attribution is not None:
        r = artefacts.xgb_no_smote_attribution
        paper = PAPER_TABLE2["xgboost_no_smote"]
        rows.append(("XGBoost, no SMOTE", None, r.sigma_shap, paper["sigma_shap"],
                     r.mean_absolute_phi, paper["mean_abs_phi"], None, "xgboost_no_smote"))

    xgb_r = artefacts.attributions_with_smote.get("xgboost")
    if xgb_r is not None:
        xgb_recall = next(m.recall for m in artefacts.metrics_with_smote if m.model_name == "xgboost")
        paper = PAPER_TABLE2["xgboost_with_smote"]
        rows.append(("XGBoost, with SMOTE", xgb_recall, xgb_r.sigma_shap, paper["sigma_shap"],
                     xgb_r.mean_absolute_phi, paper["mean_abs_phi"], paper["recall"], "xgboost_with_smote"))

    lgb_r = artefacts.attributions_with_smote.get("lightgbm")
    if lgb_r is not None:
        lgb_recall = next(m.recall for m in artefacts.metrics_with_smote if m.model_name == "lightgbm")
        paper = PAPER_TABLE2["lightgbm_with_smote"]
        rows.append(("LightGBM, with SMOTE", lgb_recall, lgb_r.sigma_shap, paper["sigma_shap"],
                     lgb_r.mean_absolute_phi, paper["mean_abs_phi"], paper["recall"], "lightgbm_with_smote"))

    for label, our_recall, our_sigma, paper_sigma, our_phi, paper_phi, paper_recall, key in rows:
        sigma_ok = abs(our_sigma - paper_sigma) <= SIGMA_SHAP_TOLERANCE
        if our_recall is not None and paper_recall is not None:
            recall_ok = abs(our_recall - paper_recall) <= RECALL_TOLERANCE
            status = "PASS" if (sigma_ok and recall_ok) else "DRIFT"
            recall_str = f"{our_recall:>8.3f}"
        else:
            status = "PASS" if sigma_ok else "DRIFT"
            recall_str = f"{'—':>8s}"
        if status == "DRIFT":
            status_code = 1

        print(f"{label:<28s}{recall_str}  "
              f"{our_sigma:>14.4f}  {paper_sigma:>15.4f}  "
              f"{our_phi:>15.4f}  {paper_phi:>16.4f}  {status}")

    print()
    print(f"Tolerances: σ_SHAP ±{SIGMA_SHAP_TOLERANCE}, recall ±{RECALL_TOLERANCE}")
    if status_code == 0:
        print("All measured values within tolerance of the paper's declared Table II.")
    else:
        print("One or more values drifted outside tolerance. Investigate before publishing.")
    return status_code


def print_findings(artefacts: SliceTwoArtefacts) -> None:
    """Confirm the two §VI.C qualitative findings direction-wise."""
    print_banner("Qualitative findings (§VI.C)")

    xgb_no_smote = artefacts.xgb_no_smote_attribution
    xgb_smote = artefacts.attributions_with_smote.get("xgboost")
    lgb_smote = artefacts.attributions_with_smote.get("lightgbm")

    if xgb_no_smote and xgb_smote:
        delta = xgb_smote.sigma_shap - xgb_no_smote.sigma_shap
        ratio = (xgb_smote.sigma_shap / xgb_no_smote.sigma_shap) if xgb_no_smote.sigma_shap > 0 else float("inf")
        direction = "INCREASES" if delta > 0 else "DECREASES"
        print(f"Observation 1 (imbalance-aware training trade-off):")
        print(f"   XGBoost no-SMOTE  → σ_SHAP = {xgb_no_smote.sigma_shap:.4f}")
        print(f"   XGBoost w/ SMOTE  → σ_SHAP = {xgb_smote.sigma_shap:.4f}")
        print(f"   SMOTE {direction} σ_SHAP by {ratio:.2f}× (paper reports ~3.4× increase)")

    if xgb_smote and lgb_smote:
        print()
        print(f"Observation 2 (architecture as secondary lever):")
        print(f"   XGBoost w/ SMOTE  → σ_SHAP = {xgb_smote.sigma_shap:.4f}")
        print(f"   LightGBM w/ SMOTE → σ_SHAP = {lgb_smote.sigma_shap:.4f}")
        if lgb_smote.sigma_shap < xgb_smote.sigma_shap:
            print(f"   LightGBM MORE stable than XGBoost at comparable recall ✓ (direction matches §VI.C)")
        else:
            print(f"   LightGBM LESS stable than XGBoost — DIRECTIONAL DRIFT from §VI.C.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = "synthetic" if args.synthetic else "uci"
    k = 10 if args.fast else 50
    bg = 50 if args.fast else 100
    nref = 100 if args.fast else 200

    print_banner(
        f"PinkMate — Slice 2 Reproduction (source={source}, K={k}, bg={bg}, n_ref={nref})"
    )

    artefacts = run_slice_two_pipeline(
        source=source,
        seed=args.seed,
        k_backgrounds=k,
        background_size=bg,
        n_reference=nref,
        include_xgb_no_smote_baseline=not args.no_attribution,
    )

    print_metrics(artefacts)

    status_code = 0
    if not args.no_attribution:
        status_code = print_table2_comparison(artefacts)
        print_findings(artefacts)

    artefacts.save(args.models_dir)
    print()
    print(f"Trained models + summaries saved to {args.models_dir.resolve()}")
    print()

    if args.synthetic:
        print("NOTE: This was a synthetic-data run. Numerical σ_SHAP values will NOT")
        print("match the paper's Table II. Re-run without --synthetic on a machine")
        print("with UCI network access to reproduce the published values.")

    return status_code


if __name__ == "__main__":
    sys.exit(main())
