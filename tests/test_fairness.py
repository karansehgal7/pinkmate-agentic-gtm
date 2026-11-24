"""Tests for pinkmate.ml.fairness — Slice 7.5c."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pinkmate.ml.fairness import (
    FairnessReport,
    FairnessThresholds,
    audit_fairness,
    audit_fairness_multi,
    summarise_fairness_flags,
)


CANONICAL_SEED: int = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(CANONICAL_SEED)


@pytest.fixture
def population(rng):
    n = 600
    employee_band = rng.choice(
        ["SMB", "Mid", "Enterprise"], size=n, p=[0.5, 0.3, 0.2]
    )
    industry = rng.choice(
        ["SaaS", "FinServ", "Manufacturing", "Healthcare"], size=n
    )
    return {"n": n, "employee_band": employee_band, "industry": industry}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_fairness_thresholds_defaults():
    th = FairnessThresholds()
    assert th.warn == 0.10
    assert th.alert == 0.20


def test_fairness_thresholds_classify():
    th = FairnessThresholds()
    assert th.classify(0.05) == "green"
    assert th.classify(0.15) == "yellow"
    assert th.classify(0.25) == "red"


def test_fairness_thresholds_rejects_invalid():
    with pytest.raises(ValueError, match="warn.*alert"):
        FairnessThresholds(warn=0.3, alert=0.1)


# ---------------------------------------------------------------------------
# audit_fairness — happy paths
# ---------------------------------------------------------------------------


def test_audit_fair_predictions_are_green(rng, population):
    # random predictions should give small disparity
    y_pred = rng.random(population["n"]) > 0.5
    report = audit_fairness(y_pred, population["employee_band"])
    assert report.level == "green"
    assert report.demographic_parity_difference < 0.10
    assert set(report.group_selection_rates.keys()) == {
        "SMB", "Mid", "Enterprise"
    }


def test_audit_detects_biased_predictions(rng, population):
    # Always predict 0 for Enterprise -> max disparity on SMB/Mid vs Enterprise
    y_pred = np.where(
        population["employee_band"] == "Enterprise",
        0,
        (rng.random(population["n"]) > 0.3).astype(int),
    )
    report = audit_fairness(y_pred, population["employee_band"])
    assert report.level == "red"
    assert report.demographic_parity_difference > 0.3
    assert report.group_selection_rates["Enterprise"] == pytest.approx(0.0)


def test_audit_without_y_true_omits_equalized_odds(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    report = audit_fairness(y_pred, population["employee_band"])
    assert report.equalized_odds_difference is None
    assert report.equalized_odds_ratio is None
    assert report.group_true_positive_rates is None
    assert report.group_false_positive_rates is None


def test_audit_with_y_true_populates_equalized_odds(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    y_true = rng.choice([0, 1], size=population["n"], p=[0.6, 0.4])
    report = audit_fairness(y_pred, population["employee_band"], y_true=y_true)
    assert report.equalized_odds_difference is not None
    assert report.equalized_odds_ratio is not None
    assert report.group_true_positive_rates is not None
    assert report.group_false_positive_rates is not None
    assert set(report.group_true_positive_rates.keys()) == {
        "SMB", "Mid", "Enterprise"
    }


def test_audit_default_attribute_name_is_employee_band(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    report = audit_fairness(y_pred, population["employee_band"])
    assert report.attribute == "employee_band"


def test_audit_custom_attribute_name(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    report = audit_fairness(
        y_pred,
        population["industry"],
        attribute_name="industry_vertical",
    )
    assert report.attribute == "industry_vertical"


def test_audit_accepts_probability_predictions(rng, population):
    # Pass continuous probabilities -> module should threshold internally
    y_pred_probas = rng.random(population["n"])
    report = audit_fairness(y_pred_probas, population["employee_band"])
    assert 0.0 <= report.demographic_parity_difference <= 1.0
    for rate in report.group_selection_rates.values():
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


def test_audit_rejects_mismatched_lengths(rng, population):
    y_pred = rng.random(population["n"])
    short_sens = population["employee_band"][: -5]
    with pytest.raises(ValueError, match="length"):
        audit_fairness(y_pred, short_sens)


def test_audit_rejects_mismatched_y_true(rng, population):
    y_pred = rng.random(population["n"])
    y_true = np.zeros(population["n"] - 5)
    with pytest.raises(ValueError, match="length"):
        audit_fairness(y_pred, population["employee_band"], y_true=y_true)


def test_audit_rejects_empty_inputs():
    with pytest.raises(ValueError, match="Empty"):
        audit_fairness(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Multi-attribute auditing
# ---------------------------------------------------------------------------


def test_audit_fairness_multi_returns_one_report_per_attribute(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    reports = audit_fairness_multi(
        y_pred,
        {
            "employee_band": population["employee_band"],
            "industry": population["industry"],
        },
    )
    assert len(reports) == 2
    attrs = {r.attribute for r in reports}
    assert attrs == {"employee_band", "industry"}


def test_audit_fairness_multi_empty_returns_empty(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    reports = audit_fairness_multi(y_pred, {})
    assert reports == []


def test_summarise_fairness_flags(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    reports = audit_fairness_multi(
        y_pred,
        {
            "employee_band": population["employee_band"],
            "industry": population["industry"],
        },
    )
    flags = summarise_fairness_flags(reports)
    assert len(flags) == 2
    assert all(":" in f for f in flags)
    assert all(f.split(":", 1)[0] in {"green", "yellow", "red"} for f in flags)


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------


def test_fairness_report_as_dict_is_json_serialisable(rng, population):
    y_pred = rng.random(population["n"]) > 0.5
    y_true = rng.choice([0, 1], size=population["n"])
    report = audit_fairness(y_pred, population["employee_band"], y_true=y_true)

    serialised = json.dumps(report.as_dict())
    recovered = json.loads(serialised)
    assert recovered["attribute"] == "employee_band"
    assert set(recovered["group_selection_rates"].keys()) == {
        "SMB", "Mid", "Enterprise"
    }
    assert recovered["equalized_odds_difference"] is not None
    assert recovered["level"] in {"green", "yellow", "red"}
