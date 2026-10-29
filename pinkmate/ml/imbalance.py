"""
SMOTE-based imbalance handling, per paper §III.C.

Reference equation:
    x_new = x_i + λ(x_nn − x_i),   λ ~ U(0, 1)

Operational constraints per §VI.A:
    SMOTE is applied exclusively to the training partition to preserve
    evaluation integrity and avoid train/test information leakage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from imblearn.over_sampling import SMOTE

SMOTE_K_NEIGHBORS: int = 5
"""
Default k_neighbors for SMOTE nearest-neighbour interpolation.

The paper does not fix k_neighbors; 5 is the imbalanced-learn library
default and the value used in the companion reproducibility artefact.
"""


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to a training partition.

    Args:
        X_train: Training feature matrix (shape: n_samples, n_features).
        y_train: Training target vector (shape: n_samples,).
        seed: Random seed propagated to SMOTE (paper default 42, §VI.A).
        k_neighbors: Number of nearest neighbours for the SMOTE interpolation.

    Returns:
        (X_resampled, y_resampled). Minority class is upsampled to match
        the majority class count.

    Raises:
        ValueError: If the training data has only one class or if the
            minority class has fewer samples than k_neighbors.
    """
    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        raise ValueError(
            f"apply_smote requires at least 2 classes in y_train; got {n_classes}."
        )

    # Classes with fewer samples than k_neighbors + 1 cannot be oversampled
    # by SMOTE's k-NN approach. imbalanced-learn raises its own error; we
    # surface a clearer one here.
    _, counts = np.unique(y_train, return_counts=True)
    if counts.min() <= k_neighbors:
        raise ValueError(
            f"Minority class has {counts.min()} samples; SMOTE requires at "
            f"least {k_neighbors + 1} to interpolate with k_neighbors={k_neighbors}."
        )

    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res.astype(int)
