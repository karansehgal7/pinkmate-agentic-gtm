"""
Taiwan Bankruptcy Prediction dataset loader (UCI Machine Learning Repository
ID 572; CC BY 4.0 license).

Reference:
    §VI.A, §VIII of Sehgal & Bhatti (2026). The paper fixes:
        - UCI ID: 572
        - Row count: 6,819 firm observations
        - Feature count: 95 financial features
        - Minority class prevalence: 3.23% (220 bankrupt firms)
        - Dataset source: Taiwan Economic Journal (1999–2009)
        - Reproducibility seed: 42
        - Train/test split: 80/20 stratified

Dataset access path:
    Primary: `ucimlrepo` library (fetch_ucirepo(id=572))
    Verification: optional SHA-256 of the fetched CSV content against the
                  value declared in the paper's Reproducibility Statement.

Preprocessing (per §VI.A):
    1. Median imputation of missing values
    2. Standard scaling (zero-mean, unit variance) fit on train partition only
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-declared constants (§VI.A, §VIII)
# ---------------------------------------------------------------------------

TAIWAN_UCI_ID: int = 572
TAIWAN_N_ROWS: int = 6819
TAIWAN_N_FEATURES: int = 95
TAIWAN_MINORITY_RATE: float = 0.0323
TAIWAN_RANDOM_SEED: int = 42
TAIWAN_TRAIN_SIZE: float = 0.80

# The paper's Reproducibility Statement declares a partial SHA-256 prefix
# of the CSV mirror used ("06e29b39b97543c5..."). We expose the known
# prefix as a soft check; a mismatch is logged as a warning (not an error)
# because the UCI CSV serialisation can vary across library versions.
TAIWAN_PAPER_SHA256_PREFIX: str = "06e29b39b97543c5"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaiwanBankruptcyDataset:
    """
    Loaded Taiwan Bankruptcy dataset.

    Attributes:
        X: Feature matrix (n_rows, 95). Columns preserve paper feature order.
        y: Target vector (n_rows,). 1 = bankrupt, 0 = not bankrupt.
        feature_names: List of 95 feature names from UCI 572.
        source: Where the data was loaded from ('uci' or 'cached' or 'synthetic').
        sha256: Hex digest of the serialised CSV content (features + target).
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: Tuple[str, ...]
    source: str
    sha256: str


def _compute_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    """Deterministic SHA-256 of the features + target serialised to CSV."""
    hasher = hashlib.sha256()
    hasher.update(X.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    hasher.update(y.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return hasher.hexdigest()


def load_taiwan_bankruptcy(
    cache_dir: Optional[Path] = None,
    allow_cache: bool = True,
) -> TaiwanBankruptcyDataset:
    """
    Load the Taiwan Bankruptcy dataset (UCI 572) via `ucimlrepo`, with
    optional on-disk cache.

    Args:
        cache_dir: Directory to cache features and target CSVs between runs.
            Defaults to ``./data/taiwan/`` if not provided.
        allow_cache: If True and a cached copy exists under cache_dir, read
            from disk instead of calling the UCI API.

    Returns:
        TaiwanBankruptcyDataset with the full 6,819-row feature matrix,
        target vector, feature names and SHA-256 of the fetched content.

    Raises:
        ImportError: If ``ucimlrepo`` is not installed and no cache is
            available.
        ConnectionError: If the UCI API cannot be reached and no cache is
            available.
    """
    cache_dir = cache_dir or Path("data/taiwan")
    features_csv = cache_dir / "features.csv"
    target_csv = cache_dir / "target.csv"

    if allow_cache and features_csv.exists() and target_csv.exists():
        LOGGER.info("Loading Taiwan Bankruptcy dataset from cache at %s", cache_dir)
        X = pd.read_csv(features_csv)
        y = pd.read_csv(target_csv).iloc[:, 0]
        source = "cached"
    else:
        LOGGER.info("Fetching Taiwan Bankruptcy dataset from UCI ID %d", TAIWAN_UCI_ID)
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as e:
            raise ImportError(
                "ucimlrepo is required to fetch the Taiwan Bankruptcy dataset "
                "from UCI. Install with: pip install ucimlrepo"
            ) from e

        ds = fetch_ucirepo(id=TAIWAN_UCI_ID)
        X = ds.data.features.copy()
        y = ds.data.targets.iloc[:, 0].copy()
        source = "uci"

        if allow_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            X.to_csv(features_csv, index=False)
            y.to_frame().to_csv(target_csv, index=False)
            LOGGER.info("Cached dataset to %s", cache_dir)

    # Shape sanity checks against paper-declared constants.
    if len(X) != TAIWAN_N_ROWS:
        LOGGER.warning(
            "Fetched Taiwan Bankruptcy dataset has %d rows; paper declares %d.",
            len(X), TAIWAN_N_ROWS,
        )
    if X.shape[1] != TAIWAN_N_FEATURES:
        LOGGER.warning(
            "Fetched Taiwan Bankruptcy dataset has %d features; paper declares %d.",
            X.shape[1], TAIWAN_N_FEATURES,
        )

    sha256 = _compute_sha256(X, y)
    if not sha256.startswith(TAIWAN_PAPER_SHA256_PREFIX):
        LOGGER.info(
            "SHA-256 prefix mismatch: paper declares %s..., fetched %s... "
            "(this is expected if the UCI CSV serialisation has changed; "
            "the paper's downstream results depend on content, not byte-exact format).",
            TAIWAN_PAPER_SHA256_PREFIX, sha256[:16],
        )

    return TaiwanBankruptcyDataset(
        X=X,
        y=y,
        feature_names=tuple(X.columns),
        source=source,
        sha256=sha256,
    )


# ---------------------------------------------------------------------------
# Preprocessing per §VI.A
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedSplit:
    """Preprocessed train/test split ready for model training."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: Tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    seed: int


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = TAIWAN_RANDOM_SEED,
    train_size: float = TAIWAN_TRAIN_SIZE,
) -> PreprocessedSplit:
    """
    Apply the paper's §VI.A preprocessing pipeline:

        1. Stratified 80/20 train/test split at ``seed`` (default 42).
        2. Median imputation (fit on train, applied to both).
        3. Standard scaling (fit on train, applied to both).

    Returns the preprocessed arrays, the fitted imputer and scaler, and the
    feature-name order (preserved from the input DataFrame).

    SMOTE is NOT applied here — the paper requires it exclusively on the
    training partition after this preprocessing step. See
    :func:`pinkmate.ml.imbalance.apply_smote`.
    """
    feature_names = tuple(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values,
        y.values,
        test_size=1.0 - train_size,
        stratify=y.values,
        random_state=seed,
    )

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return PreprocessedSplit(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr.astype(int),
        y_test=y_te.astype(int),
        feature_names=feature_names,
        imputer=imputer,
        scaler=scaler,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Synthetic test fixture
# ---------------------------------------------------------------------------

def make_synthetic_taiwan_like(
    n_rows: int = TAIWAN_N_ROWS,
    n_features: int = TAIWAN_N_FEATURES,
    minority_rate: float = TAIWAN_MINORITY_RATE,
    seed: int = TAIWAN_RANDOM_SEED,
) -> TaiwanBankruptcyDataset:
    """
    Build a synthetic dataset that mirrors Taiwan Bankruptcy's shape:
    6,819 rows, 95 features, 3.23% minority class, standardised features.

    Used ONLY for unit testing the pipeline mechanics in environments
    without UCI access. NOT a scientific substitute — the paper's Table I
    and Table II numbers are reproducible only with the real UCI 572
    dataset.

    Returns:
        TaiwanBankruptcyDataset with source='synthetic'.
    """
    rng = np.random.default_rng(seed)
    n_minority = int(round(n_rows * minority_rate))
    n_majority = n_rows - n_minority

    # Majority-class features: centred, lower variance (healthy firms)
    maj = rng.normal(loc=0.0, scale=1.0, size=(n_majority, n_features))

    # Minority-class features: shifted + higher variance on a subset of
    # 'leverage'-like features (distressed firms). Mimics the signal structure
    # that gradient boosting picks up in the real dataset.
    min_ = rng.normal(loc=0.0, scale=1.0, size=(n_minority, n_features))
    distress_features = rng.choice(n_features, size=10, replace=False)
    min_[:, distress_features] += rng.normal(loc=1.5, scale=0.5, size=(n_minority, 10))

    X_arr = np.vstack([maj, min_])
    y_arr = np.concatenate([np.zeros(n_majority, dtype=int), np.ones(n_minority, dtype=int)])

    # Shuffle rows to interleave classes
    perm = rng.permutation(n_rows)
    X_arr = X_arr[perm]
    y_arr = y_arr[perm]

    feature_names = tuple(f"synth_feat_{i:02d}" for i in range(n_features))
    X = pd.DataFrame(X_arr, columns=list(feature_names))
    y = pd.Series(y_arr, name="Bankrupt?")

    sha256 = _compute_sha256(X, y)
    return TaiwanBankruptcyDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        source="synthetic",
        sha256=sha256,
    )
