"""
LeadForge Lead Scoring v1 dataset loader (HuggingFace: shaypal5/leadforge-
lead-scoring-v1, MIT licence, synthetic B2B SaaS procurement scenario).

Slice 5 scope:
    This loader is the GTM-domain counterpart to `dataset.py`'s Taiwan
    Bankruptcy loader. It handles the domain-extension validation per
    Slice 5 of the PinkMate build programme.

    The dataset is published as a flat `leads` table with 31 columns
    covering account firmographics (industry, region, employee_band,
    estimated_revenue_band, process_maturity_band), contact attributes
    (role_function, seniority, buyer_role), lead meta (source,
    created_at), engagement signals (touch counts, session counts, page
    views, timing), funnel markers (activity_count, opportunity_created,
    has_open_opportunity), ACV estimates, a documented leakage trap
    (`total_touches_all`, counts touches across the full 90-day horizon),
    and the target label `converted_within_90_days` (bool, derived from
    `closed_won` events within the 90-day window from the lead's day-30
    snapshot anchor).

Domain-extension honesty bar:
    This is a METHODOLOGY VALIDATION on synthetic B2B-shaped data. The
    σ_SHAP numbers produced by the Slice 5 training script and committed
    to docs/gtm_training_report.md do NOT constitute performance claims
    about PinkMate on real B2B GTM leads. Real-data validation is
    deferred to Slice 5.1+ once a labelled real corpus is available
    (candidates: pink-mate.co.uk landing-page capture over 6-12 months,
    academic partnership via Kent Business School, specific industry
    engagement).

    This matches the Slice 2 honesty bar — Taiwan Bankruptcy was the
    paper's declared validation dataset, and the paper never claimed
    Taiwan-trained models should be used for B2B scoring. LeadForge
    plays the same methodology-validation role for the product-arc
    demonstration.

Reference:
    - LeadForge dataset card: https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1
    - Feature dictionary: https://leadforge-dev.github.io/leadforge/docs/dataset/features
    - Generation method: https://leadforge-dev.github.io/leadforge/docs/dataset/generation-method

Network note:
    The HuggingFace Hub endpoint (huggingface.co) may be blocked in
    some CI / sandbox environments. This loader primarily depends on
    `datasets.load_dataset()` for the real data path, but also exposes
    `make_synthetic_leadforge_like()` as a hermetic fallback that
    produces data with the same schema (31 columns, same dtypes, same
    categorical value sets) for environments without HF access. Tests
    use the synthetic fixture to stay network-free; the real training
    script uses the HF path.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LeadForge schema constants (from the published feature dictionary)
# ---------------------------------------------------------------------------

LEADFORGE_HF_REPO: str = "shaypal5/leadforge-lead-scoring-v1"
LEADFORGE_TIERS: Tuple[str, ...] = ("intro", "intermediate", "advanced")
LEADFORGE_DEFAULT_TIER: str = "intermediate"
LEADFORGE_RANDOM_SEED: int = 42
LEADFORGE_TRAIN_SIZE: float = 0.80

LEADFORGE_N_LEADS: int = 5000
LEADFORGE_N_ACCOUNTS: int = 1500
LEADFORGE_N_CONTACTS: int = 4200

# Conversion rate band is not published numerically in the dataset docs
# (declared only as "target band" in the acceptance gates YAML which is
# not exposed in v1). We approximate ~20% based on B2B GTM benchmarks for
# MQL → closed_won conversion over a 90-day horizon (industry range
# 15-25%). This is only used by the synthetic fixture; the real HF
# dataset has its own empirical conversion rate the loader does not
# need to know in advance.
LEADFORGE_APPROX_MINORITY_RATE: float = 0.20

TARGET_COLUMN: str = "converted_within_90_days"

IDENTITY_COLUMNS: Tuple[str, ...] = ("lead_id", "account_id", "contact_id")
TIMING_COLUMNS: Tuple[str, ...] = ("lead_created_at",)
LEAKAGE_COLUMNS: Tuple[str, ...] = ("total_touches_all",)  # Documented leakage trap

CATEGORICAL_COLUMNS: Tuple[str, ...] = (
    "lead_source",
    "industry",
    "region",
    "employee_band",
    "estimated_revenue_band",
    "process_maturity_band",
    "role_function",
    "seniority",
    "buyer_role",
)

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "touch_count",
    "inbound_touch_count",
    "outbound_touch_count",
    "session_count",
    "pricing_page_views",
    "demo_page_views",
    "total_session_duration_seconds",
    "touches_days_0_7",
    "touches_last_7_days",
    "days_since_first_touch",
    "days_since_last_touch",
    "activity_count",
    "opportunity_estimated_acv",
    "expected_acv",
)

BOOLEAN_COLUMNS: Tuple[str, ...] = (
    "opportunity_created",
    "has_open_opportunity",
)

# Categorical value sets, from the LeadForge feature dictionary where
# published. Bands (`employee_band`, `estimated_revenue_band`,
# `process_maturity_band`) are not fully enumerated in the public docs;
# the values below are reasonable choices that the synthetic fixture
# samples from and that the real HF dataset's bands are expected to be
# members of (the real data is one-hot-encoded consistently either way).

INDUSTRY_VALUES: Tuple[str, ...] = (
    "manufacturing", "logistics", "professional_services", "healthcare_non_clinical",
)
REGION_VALUES: Tuple[str, ...] = ("US", "UK")
LEAD_SOURCE_VALUES: Tuple[str, ...] = (
    "inbound_marketing", "sdr_outbound", "partner_referral",
)
ROLE_FUNCTION_VALUES: Tuple[str, ...] = ("finance", "ops", "it", "procurement")
SENIORITY_VALUES: Tuple[str, ...] = (
    "c_suite", "vp", "director", "manager", "individual_contributor",
)
BUYER_ROLE_VALUES: Tuple[str, ...] = (
    "economic_buyer", "champion", "technical_evaluator", "end_user",
)
EMPLOYEE_BAND_VALUES: Tuple[str, ...] = (
    "1-49", "50-199", "200-999", "1000-1999", "2000+",
)
REVENUE_BAND_VALUES: Tuple[str, ...] = (
    "<$1M", "$1M-$10M", "$10M-$50M", "$50M-$200M", "$200M+",
)
MATURITY_BAND_VALUES: Tuple[str, ...] = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Public dataset type (parallels TaiwanBankruptcyDataset)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeadForgeDataset:
    """
    Loaded LeadForge Lead Scoring v1 dataset, denormalised + leakage-cleaned.

    Attributes:
        X: Feature matrix (n_rows, ~26). Columns are one-hot-encoded
            categoricals + raw numeric + boolean columns. Identity,
            timing, and the ``total_touches_all`` leakage trap are
            excluded.
        y: Target vector (n_rows,). 1 = converted within 90 days,
            0 = not converted.
        feature_names: Column names of the encoded feature matrix.
        source: Where the data was loaded from
            ('huggingface' or 'cached' or 'synthetic').
        tier: 'intro' / 'intermediate' / 'advanced'.
        sha256: Hex digest of the serialised feature matrix + target.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: Tuple[str, ...]
    source: str
    tier: str
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
        - Identity columns (lead_id, account_id, contact_id)
        - Timing column (lead_created_at — would leak simulation clock)
        - Leakage trap (total_touches_all — counts 90-day horizon)
        - Target column (handled separately)
    """
    drop_cols = (
        list(IDENTITY_COLUMNS) + list(TIMING_COLUMNS) +
        list(LEAKAGE_COLUMNS) + [TARGET_COLUMN]
    )
    present = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=present)


def _one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encode the known categorical columns with the full published
    value set, so train/test splits produce consistent columns even if
    one partition happens to miss a category.

    Returns a DataFrame with:
        - Original numeric and boolean columns preserved
        - Each categorical column replaced by K dummies (prefix=column name)
    """
    df = df.copy()

    # Map categorical column → canonical value set
    categorical_value_sets = {
        "industry": INDUSTRY_VALUES,
        "region": REGION_VALUES,
        "lead_source": LEAD_SOURCE_VALUES,
        "role_function": ROLE_FUNCTION_VALUES,
        "seniority": SENIORITY_VALUES,
        "buyer_role": BUYER_ROLE_VALUES,
        "employee_band": EMPLOYEE_BAND_VALUES,
        "estimated_revenue_band": REVENUE_BAND_VALUES,
        "process_maturity_band": MATURITY_BAND_VALUES,
    }

    for col, values in categorical_value_sets.items():
        if col not in df.columns:
            continue
        # Force the column to the canonical categorical dtype so unseen
        # values become NaN rather than silently adding new columns
        df[col] = pd.Categorical(df[col], categories=list(values))

    # Dummies — drop_first=False so the full set is encoded; model
    # regularisation handles the multicollinearity.
    present_cats = [c for c in categorical_value_sets if c in df.columns]
    if present_cats:
        df = pd.get_dummies(df, columns=present_cats, drop_first=False, dummy_na=False)

    # Convert boolean columns to int (some sklearn models prefer int)
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int8")

    return df


# ---------------------------------------------------------------------------
# Public dataset load
# ---------------------------------------------------------------------------

def load_leadforge_lead_scoring(
    tier: Literal["intro", "intermediate", "advanced"] = LEADFORGE_DEFAULT_TIER,
    cache_dir: Optional[Path] = None,
    allow_cache: bool = True,
    fallback_to_synthetic: bool = False,
) -> LeadForgeDataset:
    """
    Load the LeadForge Lead Scoring v1 dataset for the specified tier via
    ``datasets.load_dataset()``, with optional on-disk caching.

    Args:
        tier: Difficulty tier. 'intermediate' is the canonical choice per
            the LeadForge documentation. 'intro' is less noisy; 'advanced'
            injects more missingness + outliers.
        cache_dir: Local cache directory. Defaults to ``./data/leadforge/<tier>``.
        allow_cache: If a cached copy exists, read from disk instead of
            calling the HuggingFace Hub.
        fallback_to_synthetic: If True and the HF Hub is unreachable (403,
            network error, or `datasets` not installed), fall back to
            ``make_synthetic_leadforge_like()`` with a warning. If False
            (default), raise the original exception. Intended for CI
            environments with blocked network access.

    Returns:
        LeadForgeDataset with the denormalised + leakage-cleaned feature
        matrix, target vector, feature names, and SHA-256.

    Raises:
        ValueError: If ``tier`` is not one of the supported literals.
        ImportError: If ``datasets`` is not installed and
            ``fallback_to_synthetic=False``.
        ConnectionError / HTTPError: If HuggingFace is unreachable and
            ``fallback_to_synthetic=False``.
    """
    if tier not in LEADFORGE_TIERS:
        raise ValueError(
            f"Unsupported tier: {tier!r}. Choose from {LEADFORGE_TIERS}."
        )

    cache_dir = cache_dir or Path(f"data/leadforge/{tier}")
    features_csv = cache_dir / "features.csv"
    target_csv = cache_dir / "target.csv"

    # Cache path (fast re-runs; also sidesteps HF network for repeat training)
    if allow_cache and features_csv.exists() and target_csv.exists():
        LOGGER.info("Loading LeadForge '%s' from cache at %s", tier, cache_dir)
        X = pd.read_csv(features_csv)
        y = pd.read_csv(target_csv).iloc[:, 0]
        source = "cached"
    else:
        # Primary path — HuggingFace Hub
        try:
            from datasets import load_dataset
        except ImportError as e:
            if fallback_to_synthetic:
                LOGGER.warning(
                    "`datasets` library not available; falling back to synthetic "
                    "LeadForge-like fixture. Install with: pip install datasets"
                )
                return make_synthetic_leadforge_like(tier=tier)
            raise ImportError(
                "`datasets` is required to fetch LeadForge from HuggingFace. "
                "Install with: pip install datasets"
            ) from e

        LOGGER.info("Fetching LeadForge '%s' tier from HuggingFace: %s", tier, LEADFORGE_HF_REPO)
        try:
            ds = load_dataset(LEADFORGE_HF_REPO, tier)
        except Exception as e:
            if fallback_to_synthetic:
                LOGGER.warning(
                    "HuggingFace fetch failed (%s: %s). Falling back to "
                    "synthetic LeadForge-like fixture.",
                    type(e).__name__, str(e)[:200],
                )
                return make_synthetic_leadforge_like(tier=tier)
            raise

        # The dataset ships with train / validation / test splits. For
        # Slice 5 training we concatenate all splits into a single
        # DataFrame and let `preprocess()` do its own stratified 80/20
        # split with seed=42 (same methodology as Slice 2 Taiwan).
        frames = [ds[split].to_pandas() for split in ds.keys()]
        df_full = pd.concat(frames, axis=0, ignore_index=True)

        # Separate target BEFORE dropping non-feature columns
        y = df_full[TARGET_COLUMN].astype(int)
        X_raw = _drop_non_feature_columns(df_full)
        X = _one_hot_encode_categoricals(X_raw)

        source = "huggingface"

        if allow_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            X.to_csv(features_csv, index=False)
            y.to_frame().to_csv(target_csv, index=False)
            LOGGER.info("Cached LeadForge '%s' to %s", tier, cache_dir)

    sha256 = _compute_sha256(X, y)

    return LeadForgeDataset(
        X=X,
        y=y,
        feature_names=tuple(X.columns),
        source=source,
        tier=tier,
        sha256=sha256,
    )


# ---------------------------------------------------------------------------
# Synthetic fixture — hermetic test substitute for HF-blocked environments
# ---------------------------------------------------------------------------

def make_synthetic_leadforge_like(
    n_rows: int = LEADFORGE_N_LEADS,
    minority_rate: float = LEADFORGE_APPROX_MINORITY_RATE,
    seed: int = LEADFORGE_RANDOM_SEED,
    tier: str = LEADFORGE_DEFAULT_TIER,
) -> LeadForgeDataset:
    """
    Build a synthetic dataset that matches LeadForge v1's schema and
    approximate shape (~5000 leads, ~26 features post-encoding, ~20%
    conversion rate).

    NOT a scientific substitute for the real LeadForge data. The σ_SHAP
    numbers produced on this fixture reflect the fixture's generator
    structure, NOT real B2B conversion dynamics or the LeadForge v1
    hidden-world generator. Use only for CI / cloud testing where the
    HuggingFace endpoint is unreachable.

    Returns:
        LeadForgeDataset with source='synthetic'.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_rows * minority_rate))
    n_neg = n_rows - n_pos

    def _sample_categoricals(n: int, converted_bias: float = 0.0) -> dict:
        """Sample categorical columns; `converted_bias` shifts the pos class
        toward higher-signal values (e.g., economic_buyer, c_suite)."""
        # Seniority probabilities — converted leads skew more senior
        if converted_bias > 0:
            seniority_p = [0.25, 0.30, 0.25, 0.15, 0.05]  # c_suite more likely
        else:
            seniority_p = [0.05, 0.10, 0.15, 0.35, 0.35]  # IC more likely
        buyer_role_p = [0.40, 0.25, 0.20, 0.15] if converted_bias > 0 else [
            0.10, 0.20, 0.30, 0.40
        ]
        return {
            "industry": rng.choice(INDUSTRY_VALUES, n),
            "region": rng.choice(REGION_VALUES, n, p=[0.7, 0.3]),
            "lead_source": rng.choice(LEAD_SOURCE_VALUES, n),
            "employee_band": rng.choice(EMPLOYEE_BAND_VALUES, n),
            "estimated_revenue_band": rng.choice(REVENUE_BAND_VALUES, n),
            "process_maturity_band": rng.choice(MATURITY_BAND_VALUES, n),
            "role_function": rng.choice(ROLE_FUNCTION_VALUES, n),
            "seniority": rng.choice(SENIORITY_VALUES, n, p=seniority_p),
            "buyer_role": rng.choice(BUYER_ROLE_VALUES, n, p=buyer_role_p),
        }

    def _sample_engagement(n: int, intensity: float = 1.0) -> dict:
        """Sample engagement features; `intensity` scales touches / sessions."""
        touch = rng.poisson(lam=4.0 * intensity, size=n)
        inbound = rng.binomial(touch, 0.6)
        outbound = touch - inbound
        session_count = rng.poisson(lam=2.0 * intensity, size=n)
        return {
            "touch_count": touch,
            "inbound_touch_count": inbound,
            "outbound_touch_count": outbound,
            "session_count": session_count,
            "pricing_page_views": rng.poisson(lam=0.6 * intensity, size=n),
            "demo_page_views": rng.poisson(lam=0.3 * intensity, size=n),
            "total_session_duration_seconds": (
                rng.gamma(shape=2.0, scale=120.0 * intensity, size=n)
            ).astype(int),
            "touches_days_0_7": rng.poisson(lam=1.5 * intensity, size=n),
            "touches_last_7_days": rng.poisson(lam=1.0 * intensity, size=n),
            "days_since_first_touch": rng.uniform(0, 30, n),
            "days_since_last_touch": rng.uniform(0, 7, n),
            "activity_count": rng.poisson(lam=3.0 * intensity, size=n),
            "opportunity_estimated_acv": rng.gamma(
                shape=2.0, scale=15000.0 * intensity, size=n
            ),
            "expected_acv": rng.gamma(
                shape=2.0, scale=12000.0 * intensity, size=n
            ),
        }

    # Positive class (converted) — richer engagement + senior buyer bias
    pos_categoricals = _sample_categoricals(n_pos, converted_bias=1.0)
    pos_engagement = _sample_engagement(n_pos, intensity=1.8)
    pos_opp_created = rng.binomial(1, 0.70, n_pos).astype(bool)
    pos_opp_open = (pos_opp_created & rng.binomial(1, 0.50, n_pos).astype(bool))

    # Negative class (not converted) — thinner engagement + junior buyer bias
    neg_categoricals = _sample_categoricals(n_neg, converted_bias=0.0)
    neg_engagement = _sample_engagement(n_neg, intensity=0.7)
    neg_opp_created = rng.binomial(1, 0.15, n_neg).astype(bool)
    neg_opp_open = (neg_opp_created & rng.binomial(1, 0.30, n_neg).astype(bool))

    def _build_block(cat, eng, opp_c, opp_o, n):
        block = {**cat, **eng}
        block["opportunity_created"] = opp_c
        block["has_open_opportunity"] = opp_o
        return pd.DataFrame(block)

    df_pos = _build_block(pos_categoricals, pos_engagement, pos_opp_created, pos_opp_open, n_pos)
    df_neg = _build_block(neg_categoricals, neg_engagement, neg_opp_created, neg_opp_open, n_neg)

    df_pos[TARGET_COLUMN] = True
    df_neg[TARGET_COLUMN] = False

    df_full = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)
    # Shuffle rows so classes interleave
    df_full = df_full.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    y = df_full[TARGET_COLUMN].astype(int)
    X_raw = df_full.drop(columns=[TARGET_COLUMN])
    X = _one_hot_encode_categoricals(X_raw)

    sha256 = _compute_sha256(X, y)

    return LeadForgeDataset(
        X=X,
        y=y,
        feature_names=tuple(X.columns),
        source="synthetic",
        tier=tier,
        sha256=sha256,
    )
