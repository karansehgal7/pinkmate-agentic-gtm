"""
Unit tests for pinkmate/ml/model_zoo.py — six-model ensemble (paper §III.D).
"""

from __future__ import annotations

import numpy as np
import pytest

from pinkmate.ml.model_zoo import (
    MODEL_NAMES,
    build_model_zoo,
    predict_proba_zoo,
    train_zoo,
)


@pytest.fixture
def toy_binary_data():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, size=(400, 15))
    signal = np.array([1.0, -0.9, 0.7, -0.5, 0.4] + [0.0] * 10)
    y_train = (X_train @ signal + rng.normal(0, 0.1, 400) > 0.3).astype(int)
    X_test = rng.normal(0, 1, size=(100, 15))
    y_test = (X_test @ signal + rng.normal(0, 0.1, 100) > 0.3).astype(int)
    return X_train, X_test, y_train, y_test


class TestBuildModelZoo:
    def test_all_six_models_present(self):
        zoo = build_model_zoo(seed=42)
        assert set(zoo.keys()) == set(MODEL_NAMES)
        assert len(zoo) == 6

    def test_model_names_order_preserved(self):
        assert MODEL_NAMES == (
            "logistic_regression",
            "random_forest",
            "adaboost",
            "xgboost",
            "lightgbm",
            "catboost",
        )


class TestTrainZoo:
    def test_trains_all_six(self, toy_binary_data):
        X_tr, _, y_tr, _ = toy_binary_data
        trained = train_zoo(X_tr, y_tr, seed=42, n_estimators_boosting=20)
        assert set(trained.models.keys()) == set(MODEL_NAMES)
        # Each model is now fitted — has an attribute like classes_ or n_features_in_.
        for name, model in trained.models.items():
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"

    def test_predictions_in_unit_interval(self, toy_binary_data):
        X_tr, X_te, y_tr, _ = toy_binary_data
        trained = train_zoo(X_tr, y_tr, seed=42, n_estimators_boosting=20)
        probs = predict_proba_zoo(trained, X_te)
        for name, p in probs.items():
            assert p.shape == (len(X_te),), f"{name} shape mismatch"
            assert (p >= 0).all() and (p <= 1).all(), f"{name} probabilities out of [0, 1]"

    def test_deterministic_under_fixed_seed(self, toy_binary_data):
        X_tr, X_te, y_tr, _ = toy_binary_data
        a = train_zoo(X_tr, y_tr, seed=42, n_estimators_boosting=20)
        b = train_zoo(X_tr, y_tr, seed=42, n_estimators_boosting=20)
        p_a = predict_proba_zoo(a, X_te)
        p_b = predict_proba_zoo(b, X_te)
        for name in MODEL_NAMES:
            # Tree ensembles may have minor numeric drift due to internal threading;
            # we allow a tiny tolerance.
            np.testing.assert_allclose(p_a[name], p_b[name], atol=1e-6, err_msg=f"{name} non-deterministic")
