"""
Measure Slice 7.5 conformal coverage across all four cross-domain datasets.

Pre-camera-ready micro-slice for the IEEE Cyber-AI 2026 paper revisions pass.
Populates the §3.3 conformal-coverage table in `docs/paper_revisions_v1.md`.

Research questions
------------------
Primary:
    Does the α=0.10 conformal coverage guarantee hold empirically on all
    four cross-domain datasets (Taiwan Bankruptcy, LeadForge intermediate,
    LeadForge advanced, X Education)?

Secondary:
    Does the conformal interval width correlate with minority rate?
    Hypothesis: wider intervals where minority class is rarer.

Tertiary:
    Does the conservative commit rule (``lower_bound ≥ θ``) fire less
    often than the point-score commit rule, as expected by construction?

Methodology (locked 26 Nov 2026)
--------------------------------
Decision A: Script lives under ``scripts/`` matching the pattern of
    ``reproduce_table2.py`` + ``measure_sigma_calibration.py``.
Decision B: Reuse existing ``models/<dataset>/calibration.npz`` artefacts
    written by Slice 7.5 training scripts. Within each calibration file,
    split 80% for conformal calibration and 20% for final-eval coverage
    measurement. No retraining.
Decision C: Report coverage + 95% bootstrap CI over 200 bootstrap samples
    of the final-eval partition.
Decision D: Tolerance band for "coverage holds" is ±0.05 (loose) —
    matches paper §VI.C σ_SHAP ±0.005 / recall ±0.03 convention.
Decision E: Script + tests ship as Commit A. Paper source-material
    update happens as Commit B after this script's local run.

Expected inputs
---------------
For each dataset in ``DATASETS`` the script expects:
    ``models/<dataset>/<model_name>.pkl`` (6 classification model pickles)
    ``models/<dataset>/calibration.npz`` (X_calibration + y_calibration)

If any of these are missing for a given dataset, that dataset is logged
with ``status="skipped"`` in the report and the script continues with the
remaining datasets. This matches the ``ScoringAgent`` graceful-degradation
convention from Slice 7.5.

Outputs
-------
``docs/conformal_coverage_report.md`` — generated locally, NOT committed
(gitignored like ``docs/survival_training_report.md`` and
``docs/sigma_calibration_report.md``). Camera-ready paper revisions pull
the key numbers from this file.

Usage
-----
::

    python scripts/measure_conformal_coverage_cross_domain.py
    python scripts/measure_conformal_coverage_cross_domain.py --alpha 0.05
    python scripts/measure_conformal_coverage_cross_domain.py --n-bootstrap 500

Dependencies
------------
No new dependencies. Uses MAPIE 1.x + numpy + scipy (already shipped in
Slice 7.5 via ``requirements.txt``).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pinkmate.ml.conformal import ConformalScoringWrapper

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("conformal_coverage")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Dataset registry — matches Slice 5.1 four-dataset validation base
# ---------------------------------------------------------------------------

DATASETS: List[Dict[str, Any]] = [
    {
        "key": "taiwan",
        "label": "Taiwan Bankruptcy",
        "models_dir": "models/taiwan",
        "minority_rate_pct": 3.23,
        "domain": "Financial distress",
        "source": "UCI 572 (open)",
        "slice_shipped": "Slice 2 (2903f49)",
    },
    {
        "key": "gtm_advanced",
        "label": "LeadForge advanced",
        "models_dir": "models/gtm_advanced",
        "minority_rate_pct": 7.86,
        "domain": "B2B GTM synthetic (harder)",
        "source": "HuggingFace LeadForge v1 (MIT)",
        "slice_shipped": "Slice 5.1 (5cde402)",
    },
    {
        "key": "gtm",
        "label": "LeadForge intermediate",
        "models_dir": "models/gtm",
        "minority_rate_pct": 20.98,
        "domain": "B2B GTM synthetic",
        "source": "HuggingFace LeadForge v1 (MIT)",
        "slice_shipped": "Slice 5 (42267e5)",
    },
    {
        "key": "xeducation",
        "label": "X Education",
        "models_dir": "models/xeducation",
        "minority_rate_pct": 38.54,
        "domain": "EdTech real",
        "source": "Zenodo 23791277 (CC BY 4.0)",
        "slice_shipped": "Slice 5.1 (5cde402)",
    },
]

MODEL_NAMES: Tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "adaboost",
    "xgboost",
    "lightgbm",
    "catboost",
)

# Defaults matching the paper §VI.C tolerance convention
DEFAULT_ALPHA: float = 0.10
DEFAULT_N_BOOTSTRAP: int = 200
DEFAULT_TOLERANCE: float = 0.05
DEFAULT_THETA: float = 0.75
DEFAULT_SEED: int = 42
CALIBRATION_SPLIT_FRACTION: float = 0.80  # 80% calib, 20% final-eval


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DatasetCoverageResult:
    """Per-dataset conformal-coverage measurement result."""

    key: str
    label: str
    minority_rate_pct: float
    domain: str
    source: str
    slice_shipped: str
    status: str  # "ok" | "skipped" | "error"
    note: str = ""
    n_calibration: int = 0
    n_final_eval: int = 0
    empirical_coverage: Optional[float] = None
    coverage_ci_low: Optional[float] = None
    coverage_ci_high: Optional[float] = None
    mean_interval_width: Optional[float] = None
    median_interval_width: Optional[float] = None
    point_commit_rate: Optional[float] = None
    conservative_commit_rate: Optional[float] = None
    within_tolerance: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-dataset measurement
# ---------------------------------------------------------------------------


def load_ensemble(models_dir: Path) -> Dict[str, Any]:
    """Load the 6-model classification ensemble from ``models_dir``.

    Missing pickles are silently skipped; the returned dict holds only the
    models that loaded cleanly. Callers decide whether to proceed or skip.
    """
    ensemble: Dict[str, Any] = {}
    for name in MODEL_NAMES:
        path = models_dir / f"{name}.pkl"
        if not path.exists():
            continue
        try:
            with open(path, "rb") as fh:
                ensemble[name] = pickle.load(fh)
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", path, exc)
    return ensemble


def load_calibration(cal_path: Path) -> Optional[Tuple[NDArray, NDArray]]:
    """Load calibration.npz; return None on failure."""
    if not cal_path.exists():
        return None
    try:
        data = np.load(cal_path, allow_pickle=True)
        return np.asarray(data["X_calibration"]), np.asarray(data["y_calibration"])
    except Exception as exc:
        LOGGER.warning("Failed to load %s: %s", cal_path, exc)
        return None


def split_calibration_final_eval(
    X: NDArray,
    y: NDArray,
    frac_calibration: float = CALIBRATION_SPLIT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Deterministically split (X, y) into calibration + final-eval partitions."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_cal = int(np.floor(n * frac_calibration))
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]
    return X[cal_idx], y[cal_idx], X[eval_idx], y[eval_idx]


def bootstrap_coverage_ci(
    in_interval: NDArray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """Return (lo, hi) of a ``confidence`` bootstrap CI on the coverage.

    ``in_interval`` is a boolean array of length n_eval where True means the
    true label fell inside the predicted conformal interval. The bootstrap
    draws n_bootstrap resamples with replacement and reports the empirical
    percentile interval.
    """
    rng = np.random.default_rng(seed)
    n = len(in_interval)
    if n == 0:
        return 0.0, 0.0
    boot_coverages = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        boot_coverages[b] = float(np.mean(in_interval[resample_idx]))
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boot_coverages, alpha))
    hi = float(np.quantile(boot_coverages, 1.0 - alpha))
    return lo, hi


def measure_dataset_coverage(
    dataset_cfg: Dict[str, Any],
    repo_root: Path,
    alpha: float,
    n_bootstrap: int,
    tolerance: float,
    theta: float,
    seed: int,
) -> DatasetCoverageResult:
    """Produce a :class:`DatasetCoverageResult` for one dataset."""
    key = dataset_cfg["key"]
    base = DatasetCoverageResult(
        key=key,
        label=dataset_cfg["label"],
        minority_rate_pct=dataset_cfg["minority_rate_pct"],
        domain=dataset_cfg["domain"],
        source=dataset_cfg["source"],
        slice_shipped=dataset_cfg["slice_shipped"],
        status="ok",
    )
    models_dir = repo_root / dataset_cfg["models_dir"]
    cal_path = models_dir / "calibration.npz"

    ensemble = load_ensemble(models_dir)
    if not ensemble:
        base.status = "skipped"
        base.note = (
            f"no model pickles in {models_dir} — run the matching training "
            "script locally and re-run this measurement."
        )
        LOGGER.warning("%s: %s", key, base.note)
        return base

    loaded = load_calibration(cal_path)
    if loaded is None:
        base.status = "skipped"
        base.note = (
            f"no calibration.npz in {models_dir} — this file is written by "
            "the Slice 7.5-updated training scripts. Re-run the training "
            "script to produce it, then re-run this measurement."
        )
        LOGGER.warning("%s: %s", key, base.note)
        return base
    X_full, y_full = loaded

    X_cal, y_cal, X_eval, y_eval = split_calibration_final_eval(
        X_full, y_full, frac_calibration=CALIBRATION_SPLIT_FRACTION, seed=seed
    )
    base.n_calibration = int(X_cal.shape[0])
    base.n_final_eval = int(X_eval.shape[0])

    # Guard: need enough calibration data for a stable conformal wrapper
    if base.n_calibration < 20 or base.n_final_eval < 20:
        base.status = "skipped"
        base.note = (
            f"insufficient calibration/eval rows (n_cal={base.n_calibration}, "
            f"n_eval={base.n_final_eval}). Need >= 20 of each."
        )
        LOGGER.warning("%s: %s", key, base.note)
        return base

    try:
        wrapper = ConformalScoringWrapper(ensemble, alpha=alpha)
        wrapper.conformalize(X_cal, y_cal)
        points, lowers, uppers = wrapper.predict_interval_batch(X_eval)
    except Exception as exc:
        base.status = "error"
        base.note = f"ConformalScoringWrapper failed: {type(exc).__name__}: {exc}"
        LOGGER.error("%s: %s", key, base.note)
        return base

    # Primary measurement — empirical coverage
    y_eval_f = y_eval.astype(float)
    in_interval = (y_eval_f >= lowers) & (y_eval_f <= uppers)
    empirical_coverage = float(np.mean(in_interval))
    ci_low, ci_high = bootstrap_coverage_ci(
        in_interval, n_bootstrap=n_bootstrap, confidence=0.95, seed=seed
    )

    # Secondary — interval widths
    widths = uppers - lowers

    # Tertiary — commit rates
    conservative_commits = int(np.sum(lowers >= theta))
    point_commits = int(np.sum(points >= theta))
    n = base.n_final_eval

    base.empirical_coverage = empirical_coverage
    base.coverage_ci_low = ci_low
    base.coverage_ci_high = ci_high
    base.mean_interval_width = float(np.mean(widths))
    base.median_interval_width = float(np.median(widths))
    base.point_commit_rate = float(point_commits) / n
    base.conservative_commit_rate = float(conservative_commits) / n
    base.within_tolerance = bool(
        abs(empirical_coverage - (1.0 - alpha)) <= tolerance
    )
    return base


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: Optional[float], ndigits: int = 4) -> str:
    return "—" if x is None else f"{x:.{ndigits}f}"


def _status_marker(r: DatasetCoverageResult) -> str:
    if r.status == "skipped":
        return "⚠️ skipped"
    if r.status == "error":
        return "❌ error"
    return "✅ ok" if r.within_tolerance else "⚠️ out of tolerance"


def render_markdown_report(
    results: List[DatasetCoverageResult],
    alpha: float,
    n_bootstrap: int,
    tolerance: float,
    theta: float,
    seed: int,
) -> str:
    target_coverage = 1.0 - alpha
    lines: List[str] = []
    lines.append("# Conformal coverage across four cross-domain datasets\n")
    lines.append(
        "Generated by `scripts/measure_conformal_coverage_cross_domain.py`.\n"
    )
    lines.append("")
    lines.append("## Protocol\n")
    lines.append(
        f"- α = `{alpha}` ({target_coverage:.0%} target empirical coverage)"
    )
    lines.append(f"- Bootstrap samples for 95% CI: `{n_bootstrap}`")
    lines.append(f"- Tolerance band: `±{tolerance:.2f}` around target")
    lines.append(f"- Governance threshold θ = `{theta}`")
    lines.append(
        f"- Calibration/final-eval split: `{round(CALIBRATION_SPLIT_FRACTION*100)}%`/"
        f"`{round((1-CALIBRATION_SPLIT_FRACTION)*100)}%`"
    )
    lines.append(f"- RNG seed: `{seed}`")
    lines.append("")
    lines.append("## Primary — empirical coverage per dataset\n")
    lines.append(
        "| Dataset | Minority rate | n_cal | n_eval | Empirical coverage "
        "[95% CI] | Within ±tol of target | Status |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for r in results:
        cov_cell = (
            f"{_fmt(r.empirical_coverage, 3)} "
            f"[{_fmt(r.coverage_ci_low, 3)}, {_fmt(r.coverage_ci_high, 3)}]"
            if r.empirical_coverage is not None
            else "—"
        )
        tol_cell = "yes" if r.within_tolerance else "no" if r.within_tolerance is not None else "—"
        lines.append(
            f"| {r.label} | {r.minority_rate_pct:.2f}% | {r.n_calibration or '—'} "
            f"| {r.n_final_eval or '—'} | {cov_cell} | {tol_cell} | {_status_marker(r)} |"
        )
    lines.append("")

    lines.append("## Secondary — interval widths\n")
    lines.append("| Dataset | Mean width | Median width |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.label} | {_fmt(r.mean_interval_width, 4)} | "
            f"{_fmt(r.median_interval_width, 4)} |"
        )
    lines.append("")

    lines.append("## Tertiary — commit rates (θ = {})\n".format(theta))
    lines.append(
        "| Dataset | Point-score commit rate | Conservative commit rate | "
        "Δ (conservative − point) |"
    )
    lines.append("|---|---|---|---|")
    for r in results:
        delta = (
            None
            if r.conservative_commit_rate is None or r.point_commit_rate is None
            else r.conservative_commit_rate - r.point_commit_rate
        )
        lines.append(
            f"| {r.label} | {_fmt(r.point_commit_rate, 4)} | "
            f"{_fmt(r.conservative_commit_rate, 4)} | {_fmt(delta, 4)} |"
        )
    lines.append("")
    lines.append(
        "> Expectation: conservative commit rate ≤ point-score commit rate "
        "(negative Δ), because the conservative rule fires only when "
        "`lower_bound ≥ θ` which is strictly stricter than `point ≥ θ`.\n"
    )

    # Notes for skipped datasets
    skipped = [r for r in results if r.status != "ok"]
    if skipped:
        lines.append("## Notes on skipped / errored datasets\n")
        for r in skipped:
            lines.append(f"- **{r.label}** — {r.status}: {r.note}")
        lines.append("")

    lines.append("## Reproducibility\n")
    lines.append(
        "This measurement reuses the Slice 7.5 `calibration.npz` artefacts "
        "written by the training scripts (`scripts/reproduce_table2.py`, "
        "`scripts/train_gtm_ensemble.py`, `scripts/train_xeducation_ensemble.py`)."
    )
    lines.append(
        "To reproduce: train the ensembles that produce model pickles + "
        "`calibration.npz`, then run this script. No retraining happens inside "
        "this measurement script; it only conformalises existing models."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Slice 7.5 conformal coverage across all four "
                    "cross-domain datasets for the IEEE Cyber-AI 2026 "
                    "camera-ready paper revisions pass.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Miscoverage rate (default: 0.10 for 90% target coverage).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Bootstrap samples for the 95% CI (default: 200).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Tolerance band for coverage-holds verdict (default: 0.05).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=DEFAULT_THETA,
        help="Governance threshold for commit rates (default: 0.75).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed (default: 42).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root for resolving models/ paths (default: cwd).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs") / "conformal_coverage_report.md",
        help="Output report path relative to --repo-root (default: "
             "docs/conformal_coverage_report.md).",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Optional JSON dump path for machine-readable results.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    LOGGER.info(
        "Starting conformal coverage measurement: alpha=%.3f, "
        "n_bootstrap=%d, tolerance=%.3f, theta=%.3f, seed=%d",
        args.alpha, args.n_bootstrap, args.tolerance, args.theta, args.seed,
    )

    results: List[DatasetCoverageResult] = []
    for cfg in DATASETS:
        LOGGER.info("Processing dataset: %s (%s)", cfg["key"], cfg["label"])
        with warnings.catch_warnings():
            # Slice 7.5 drift warnings are informational for this script;
            # suppress them to keep the console tidy.
            warnings.simplefilter("ignore", RuntimeWarning)
            result = measure_dataset_coverage(
                dataset_cfg=cfg,
                repo_root=args.repo_root,
                alpha=args.alpha,
                n_bootstrap=args.n_bootstrap,
                tolerance=args.tolerance,
                theta=args.theta,
                seed=args.seed,
            )
        results.append(result)
        if result.status == "ok":
            LOGGER.info(
                "  coverage=%.3f [CI %.3f, %.3f] within_tol=%s",
                result.empirical_coverage,
                result.coverage_ci_low,
                result.coverage_ci_high,
                result.within_tolerance,
            )

    # Write the markdown report
    report = render_markdown_report(
        results,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        tolerance=args.tolerance,
        theta=args.theta,
        seed=args.seed,
    )
    report_path = args.repo_root / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)

    # Optional JSON dump for downstream tooling
    if args.json_path is not None:
        json_path = args.repo_root / args.json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "protocol": {
                        "alpha": args.alpha,
                        "n_bootstrap": args.n_bootstrap,
                        "tolerance": args.tolerance,
                        "theta": args.theta,
                        "seed": args.seed,
                        "calibration_split_fraction": CALIBRATION_SPLIT_FRACTION,
                    },
                    "results": [r.as_dict() for r in results],
                },
                fh,
                indent=2,
            )
        LOGGER.info("Wrote %s", json_path)

    # Exit code: 0 if ALL ok-status datasets are within tolerance (or none ok)
    ok_results = [r for r in results if r.status == "ok"]
    if ok_results and all(r.within_tolerance for r in ok_results):
        return 0
    if not ok_results:
        LOGGER.warning(
            "No datasets produced coverage measurements. "
            "Run the training scripts first to populate models/ + calibration.npz."
        )
        return 0  # Not an error — just nothing measured yet
    return 1  # At least one ok dataset failed the tolerance check


if __name__ == "__main__":
    sys.exit(main())
