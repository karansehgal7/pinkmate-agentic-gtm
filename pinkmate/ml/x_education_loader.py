"""
X Education Lead Scoring v1 dataset loader (Zenodo 23791277, CC BY 4.0,
mirrored from the Kaggle X Education lead-scoring dataset).

Slice 5.1 scope:
    This loader is the real-data counterpart to leadforge_loader.py. X
    Education is the first REAL labelled B2B tabular lead-scoring
    dataset in the PinkMate build programme (EdTech domain, ~38.54%
    conversion rate, 9240 leads, 37 raw columns).

Defensive leakage-trap exclusion:
    X Education's `Tags` column is a well-documented sales-team-assigned
    post-hoc label (e.g., "Closed by Horizon", "Interested in other
    courses") that drives the suspicious 0.98+ ROC-AUC reported in prior
    published work (Aqil & Halim 2026). We exclude it alongside three
    other defensive leakage-trap columns:

        - Tags (THE leakage column — sales post-hoc labels)
        - Lead Quality (sometimes filled post-sales-interaction)
        - Lead Profile (similar temporal concern)
        - Last Notable Activity (potentially post-conversion-decision)

    This matches the LeadForge honesty bar — exclude anything that could
    possibly leak, even defensively. Expected post-exclusion ROC-AUC
    sits in the 0.70-0.85 range; any model exceeding 0.95 fails the
    methodology-integrity test (enforced in test_xeducation_training.py).

Domain-extension honesty bar:
    Slice 5.1 adds X Education as the first REAL-DATA validation point
    in the PinkMate build programme. Combined with Taiwan Bankruptcy
    (3.23% minority, financial-distress) and LeadForge intermediate
    (20.98%, B2B GTM synthetic), the σ_SHAP attribution-stability
    directions are now tested across the full 3-38% imbalance range
    and across three domain contexts (financial, B2B GTM synthetic,
    EdTech real).

    The paper's §VI.C SMOTE-direction observation is methodologically
    EXPECTED to weaken at near-balanced classes — SMOTE has less
    synthetic positives to generate as class balance approaches 50%.
    A DRIFT verdict on X Education at 38.5% minority is a nuanced
    methodological-boundary finding, not a failure.

Reference:
    - Zenodo record: https://zenodo.org/records/23791277
    - Aqil & Halim (2026): "Automated Sales Lead Scoring and
      Prioritization: A CRISP-DM Comparison of Four Machine Learning
      Classifiers" (CC BY 4.0, DOI 10.5281/zenodo.23791277)
    - X Education dataset reference EDA:
      https://analyticsvidhya.com/blog/2022/09/
      exploratory-data-analysis-eda-on-lead-scoring-dataset

Network note:
    The Zenodo files endpoint (zenodo.org) is reachable from most
    developer machines but may be blocked in some sandboxed CI
    environments. This loader uses `urllib.request.urlopen` (standard
    library only, no new dependencies, respects HTTPS_PROXY env var)
    for the primary path, and exposes `make_synthetic_xeducation_like()`
    as a hermetic fallback that produces data matching the loader's
    encoded schema for environments without network access.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# X Education schema constants
# ---------------------------------------------------------------------------

XEDU_ZENODO_RECORD_ID: str = "23791277"
XEDU_ZENODO_RECORD_URL: str = "https://zenodo.org/records/23791277"
XEDU_ZENODO_CSV_FILENAME: str = "Lead Scoring (1).csv"
# URL-encoded direct download — the filename contains a space and parentheses
XEDU_ZENODO_CSV_URL: str = (
    "https://zenodo.org/records/23791277/files/"
    "Lead%20Scoring%20%281%29.csv"
)

XEDU_N_LEADS: int = 9240
XEDU_N_RAW_COLUMNS: int = 37
# ~38.54% conversion per Analytics Vidhya reference
XEDU_APPROX_MINORITY_RATE: float = 0.3854
XEDU_RANDOM_SEED: int = 42
XEDU_TRAIN_SIZE: float = 0.80

TARGET_COLUMN: str = "Converted"

# Identity columns — drop
IDENTITY_COLUMNS: Tuple[str, ...] = ("Prospect ID", "Lead Number")

# Defensive leakage-trap exclusions (Decision A — all four dropped)
LEAKAGE_COLUMNS: Tuple[str, ...] = (
    "Tags",                   # THE leakage column — sales post-hoc labels
    "Lead Quality",           # Sometimes filled post-sales-interaction
    "Lead Profile",           # Similar temporal concern
    "Last Notable Activity",  # Potentially post-conversion-decision
)

# Near-zero-variance columns documented in X Education dataset.
# These are near-constant (>95% a single value), so they contribute
# essentially nothing to model discrimination while inflating feature
# count — drop them before encoding.
NEAR_ZERO_VARIANCE_COLUMNS: Tuple[str, ...] = (
    "Magazine",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
)

CATEGORICAL_COLUMNS: Tuple[str, ...] = (
    "Lead Origin",
    "Lead Source",
    "Last Activity",
    "Country",
    "Specialization",
    "How did you hear about X Education",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "City",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
)

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
)

BOOLEAN_YESNO_COLUMNS: Tuple[str, ...] = (
    "Do Not Email",
    "Do Not Call",
    "Search",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
    "A free copy of Mastering The Interview",
)

# X Education uses "Select" as a placeholder meaning the user didn't choose
# a value — treat as NaN so median/mode imputation handles it later.
SELECT_PLACEHOLDER_VALUE: str = "Select"


# Canonical value sets for the synthetic fixture. These do NOT constrain
# the real-data loader — real data is one-hot-encoded with whatever values
# appear via pd.get_dummies.
_SYNTHETIC_CATEGORICAL_VALUES = {
    "Lead Origin": ("API", "Landing Page Submission", "Lead Add Form", "Lead Import"),
    "Lead Source": (
        "Olark Chat", "Organic Search", "Direct Traffic", "Google",
        "Reference", "Referral Sites", "Welingak Website",
    ),
    "Last Activity": (
        "Email Opened", "SMS Sent", "Olark Chat Conversation",
        "Page Visited on Website", "Converted to Lead", "Email Bounced",
        "Email Link Clicked", "Form Submitted on Website",
    ),
    "Country": ("India", "United States", "United Arab Emirates", "Singapore", "Australia"),
    "Specialization": (
        "Finance Management", "Human Resource Management",
        "Marketing Management", "Operations Management",
        "Business Administration", "IT Projects Management",
    ),
    "How did you hear about X Education": (
        "Online Search", "Word Of Mouth", "Student of SomeSchool",
        "Other", "Multiple Sources",
    ),
    "What is your current occupation": (
        "Unemployed", "Working Professional", "Student",
        "Businessman", "Housewife",
    ),
    "What matters most to you in choosing a course": (
        "Better Career Prospects", "Flexibility & Convenience", "Other",
    ),
    "City": (
        "Mumbai", "Thane & Outskirts", "Other Cities",
        "Other Cities of Maharashtra", "Other Metro Cities",
    ),
    "Asymmetrique Activity Index": ("01.High", "02.Medium", "03.Low"),
    "Asymmetrique Profile Index": ("01.High", "02.Medium", "03.Low"),
}


# ---------------------------------------------------------------------------
# Public dataset type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XEducationDataset:
    """
    Loaded X Education lead-scoring dataset, leakage-cleaned + encoded.

    Attributes:
        X: Feature matrix (n_rows, ~70-130 depending on real-data cardinality).
            Columns are one-hot-encoded categoricals + raw numeric + boolean.
            Identity, leakage, near-zero-variance, and target columns are
            excluded.
        y: Target vector (n_rows,). 1 = converted, 0 = not converted.
        feature_names: Column names of the encoded feature matrix.
        source: Where the data came from — 'zenodo' | 'cached' | 'synthetic'.
        sha256: Hex digest of the serialised feature matrix + target.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: Tuple[str, ...]
    source: str
    sha256: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    """Deterministic SHA-256 of features + target serialised to CSV."""
    hasher = hashlib.sha256()
    hasher.update(X.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    hasher.update(y.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return hasher.hexdigest()


def _drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that must not become model features:
        - Identity (Prospect ID, Lead Number)
        - Leakage (Tags, Lead Quality, Lead Profile, Last Notable Activity)
        - Near-zero-variance (Magazine, etc.)
        - Target column (handled separately)

    Logs which columns were dropped for transparency.
    """
    drop_cols = (
        list(IDENTITY_COLUMNS)
        + list(LEAKAGE_COLUMNS)
        + list(NEAR_ZERO_VARIANCE_COLUMNS)
        + [TARGET_COLUMN]
    )
    present = [c for c in drop_cols if c in df.columns]
    missing = [c for c in drop_cols if c not in df.columns]

    LOGGER.info(
        "X Education loader dropping %d non-feature columns: %s",
        len(present), present,
    )
    if missing:
        LOGGER.warning(
            "Expected columns not found (may be fine for synthetic fixture): %s",
            missing,
        )
    return df.drop(columns=present)


def _normalize_select_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace X Education's 'Select' placeholder (meaning 'user did not choose
    a value') with NaN so median/mode imputation handles it downstream.

    Applied to all object-dtype columns.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(SELECT_PLACEHOLDER_VALUE, np.nan)
    return df


def _normalize_yesno_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No strings in known boolean columns to Int8 (1/0). Preserves
    NaN as NaN.
    """
    df = df.copy()
    for col in BOOLEAN_YESNO_COLUMNS:
        if col not in df.columns:
            continue
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
        df[col] = df[col].map(mapping).astype("Int8")
    return df


def _one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encode categorical columns. Uses pd.get_dummies with drop_first=False
    so the full value set is encoded (model regularisation handles the
    multicollinearity, same strategy as leadforge_loader.py).

    Boolean columns stay as Int8 (they're already numeric).
    Numeric columns pass through untouched.
    """
    present_cats = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if not present_cats:
        return df.copy()

    # Fill NaN with a sentinel so get_dummies has something to encode;
    # downstream imputation at the preprocess() stage handles sentinels
    # by treating them as their own category (already encoded).
    df = df.copy()
    for col in present_cats:
        df[col] = df[col].astype(str).replace("nan", "missing")

    return pd.get_dummies(df, columns=present_cats, drop_first=False, dummy_na=False)


def _fetch_zenodo_csv(url: str, timeout: int = 60) -> str:
    """
    Fetch the Zenodo CSV via urllib (stdlib-only, respects HTTPS_PROXY).

    Returns the CSV content as a UTF-8 string.

    Raises urllib.error.URLError or ConnectionError on network failure —
    callers decide whether to raise or fall back to synthetic.
    """
    LOGGER.info("Fetching X Education CSV from Zenodo: %s", url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PinkMate-Slice-5.1/1.0 (research artefact)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8")


# ---------------------------------------------------------------------------
# Public dataset load
# ---------------------------------------------------------------------------

def load_x_education_lead_scoring(
    cache_dir: Optional[Path] = None,
    allow_cache: bool = True,
    fallback_to_synthetic: bool = False,
    timeout: int = 60,
) -> XEducationDataset:
    """
    Fetch the X Education Lead Scoring dataset from the Zenodo CC BY 4.0
    mirror, apply defensive leakage-trap exclusion + schema normalization +
    one-hot encoding, and return as a frozen XEducationDataset.

    Args:
        cache_dir: Local cache directory. Defaults to data/x_education/.
        allow_cache: If True, reuse an existing local cache instead of
            re-fetching from Zenodo.
        fallback_to_synthetic: If True and Zenodo is unreachable (403,
            network error), fall back to make_synthetic_xeducation_like()
            with a warning. If False, re-raise the original exception.
        timeout: urllib timeout in seconds for the Zenodo fetch.

    Returns:
        XEducationDataset with the leakage-cleaned + encoded feature matrix,
        target vector, feature names, source tag, and SHA-256.

    Raises:
        ConnectionError / urllib.error.URLError: If Zenodo is unreachable
            and fallback_to_synthetic=False.
    """
    cache_dir = cache_dir or Path("data/x_education")
    features_csv = cache_dir / "features.csv"
    target_csv = cache_dir / "target.csv"

    # Cache hit — skip the network call
    if allow_cache and features_csv.exists() and target_csv.exists():
        LOGGER.info("Loading X Education from cache at %s", cache_dir)
        X = pd.read_csv(features_csv)
        y = pd.read_csv(target_csv).iloc[:, 0]
        source = "cached"
    else:
        # Primary path — Zenodo CC-BY mirror
        try:
            csv_text = _fetch_zenodo_csv(XEDU_ZENODO_CSV_URL, timeout=timeout)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if fallback_to_synthetic:
                LOGGER.warning(
                    "Zenodo fetch failed (%s: %s). Falling back to "
                    "synthetic X Education-like fixture.",
                    type(e).__name__, str(e)[:200],
                )
                return make_synthetic_xeducation_like()
            raise

        df_full = pd.read_csv(StringIO(csv_text))
        LOGGER.info(
            "X Education loaded from Zenodo — %d rows × %d raw columns",
            len(df_full), df_full.shape[1],
        )

        # Separate target before dropping non-feature columns
        y = df_full[TARGET_COLUMN].astype(int)

        X_raw = _drop_non_feature_columns(df_full)
        X_raw = _normalize_select_placeholders(X_raw)
        X_raw = _normalize_yesno_booleans(X_raw)
        X = _one_hot_encode_categoricals(X_raw)

        source = "zenodo"

        if allow_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            X.to_csv(features_csv, index=False)
            y.to_frame().to_csv(target_csv, index=False)
            LOGGER.info("Cached X Education to %s", cache_dir)

    sha256 = _compute_sha256(X, y)

    LOGGER.info(
        "X Education ready — %d rows, %d encoded features, minority rate %.4f",
        len(X), X.shape[1], float(y.mean()),
    )

    return XEducationDataset(
        X=X,
        y=y,
        feature_names=tuple(X.columns),
        source=source,
        sha256=sha256,
    )


# ---------------------------------------------------------------------------
# Synthetic fixture — hermetic test substitute for Zenodo-blocked environments
# ---------------------------------------------------------------------------

def make_synthetic_xeducation_like(
    n_rows: int = XEDU_N_LEADS,
    minority_rate: float = XEDU_APPROX_MINORITY_RATE,
    seed: int = XEDU_RANDOM_SEED,
) -> XEducationDataset:
    """
    Build a synthetic dataset matching X Education's post-encoded schema —
    categoricals (reduced value sets), numeric, boolean — with configurable
    size and class balance.

    Calibration:
        Positive-class separability is tuned to produce logistic-regression
        ROC-AUC roughly in the 0.75-0.88 range (matching real X Education's
        post-leakage-removal performance per Aqil & Halim 2026's baseline
        once the Tags leakage column is dropped). The methodology-integrity
        AUC guard (tests/test_xeducation_training.py::
        test_methodology_integrity_auc_guard) asserts all AUCs < 0.95,
        so the generator parameters below are deliberately conservative
        to stay well under that ceiling even at K=10 noise.

    NOT a scientific substitute for the real X Education data. The σ_SHAP
    numbers computed on this fixture reflect the generator's structure, not
    real EdTech conversion dynamics. Use only for CI / test environments
    where Zenodo is unreachable.

    Returns XEducationDataset with source='synthetic'.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_rows * minority_rate))
    n_neg = n_rows - n_pos

    def _sample_block(n: int, converted_bias: float = 0.0) -> dict:
        """
        Build one block of synthetic rows. converted_bias > 0 skews toward
        higher-signal values. All signal strengths are tuned conservatively
        so logistic regression lands in the 0.75-0.88 AUC band (realistic
        for X Education post-leakage-removal).
        """
        block: dict = {}

        # Categoricals — softer skew (1.3 → 1.0 instead of 2.0 → 1.0).
        # Positives slightly favour first values, but not aggressively.
        for col, values in _SYNTHETIC_CATEGORICAL_VALUES.items():
            if converted_bias > 0:
                p = np.linspace(1.3, 1.0, len(values))
            else:
                p = np.linspace(1.0, 1.3, len(values))
            p = p / p.sum()
            block[col] = rng.choice(values, size=n, p=p)

        # Numeric — tighter intensity ratio (1.25 vs 0.90 instead of 1.6
        # vs 0.8) plus additive Gaussian noise on the continuous columns
        # to prevent perfect gamma-tail separation.
        intensity = 1.25 if converted_bias > 0 else 0.90
        block["TotalVisits"] = rng.poisson(lam=4.0 * intensity, size=n)

        time_signal = rng.gamma(shape=2.0, scale=300.0 * intensity, size=n)
        time_noise = rng.normal(loc=0.0, scale=250.0, size=n)
        block["Total Time Spent on Website"] = np.clip(
            time_signal + time_noise, a_min=0, a_max=None
        ).astype(int)

        pv_signal = rng.gamma(shape=2.0, scale=1.5 * intensity, size=n)
        pv_noise = rng.normal(loc=0.0, scale=0.6, size=n)
        block["Page Views Per Visit"] = np.clip(
            pv_signal + pv_noise, a_min=0.0, a_max=None
        )

        # Asymmetrique Score ranges tightened — overlap bands are the norm
        block["Asymmetrique Activity Score"] = rng.integers(
            low=6 if converted_bias > 0 else 7,
            high=17 if converted_bias > 0 else 16,
            size=n,
        )
        block["Asymmetrique Profile Score"] = rng.integers(
            low=11 if converted_bias > 0 else 10,
            high=19 if converted_bias > 0 else 18,
            size=n,
        )

        # Boolean Yes/No cols — HALVED signal strength (0.18 vs 0.12
        # instead of 0.30 vs 0.10). With 9 boolean columns at 0.18/0.12
        # the combined per-row log-likelihood ratio is ~3.6 bits instead
        # of ~14 bits, which is realistic for soft lead signals.
        for col in BOOLEAN_YESNO_COLUMNS:
            p_yes = 0.18 if converted_bias > 0 else 0.12
            block[col] = rng.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes])

        return block

    pos_block = _sample_block(n_pos, converted_bias=1.0)
    neg_block = _sample_block(n_neg, converted_bias=0.0)

    df_pos = pd.DataFrame(pos_block)
    df_neg = pd.DataFrame(neg_block)
    df_pos[TARGET_COLUMN] = 1
    df_neg[TARGET_COLUMN] = 0

    df_full = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)
    df_full = df_full.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    y = df_full[TARGET_COLUMN].astype(int)
    X_raw = df_full.drop(columns=[TARGET_COLUMN])

    # Run the same pipeline as the real loader
    X_raw = _normalize_select_placeholders(X_raw)
    X_raw = _normalize_yesno_booleans(X_raw)
    X = _one_hot_encode_categoricals(X_raw)

    sha256 = _compute_sha256(X, y)

    return XEducationDataset(
        X=X,
        y=y,
        feature_names=tuple(X.columns),
        source="synthetic",
        sha256=sha256,
    )
