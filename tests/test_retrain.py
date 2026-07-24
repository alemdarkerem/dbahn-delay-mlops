"""Tests for promotion rules and new-month detection."""

from typing import Any

import pytest

from dbahn_delay.models import retrain
from dbahn_delay.models.retrain import promotion_verdict

CHAMPION = {"pr_auc": 0.50, "pinball_p90": 1.60, "ece": 0.02}


def challenger(**overrides: float) -> dict[str, float]:
    return {**CHAMPION, **overrides}


def test_equal_challenger_is_promoted() -> None:
    promoted, failures = promotion_verdict(CHAMPION, challenger())
    assert promoted and not failures


def test_worse_pr_auc_blocks_promotion() -> None:
    promoted, failures = promotion_verdict(CHAMPION, challenger(pr_auc=0.49))
    assert not promoted
    assert any("pr_auc" in f for f in failures)


def test_tiny_pr_auc_dip_within_tolerance_is_fine() -> None:
    promoted, _ = promotion_verdict(CHAMPION, challenger(pr_auc=0.499))
    assert promoted


def test_worse_pinball_blocks_promotion() -> None:
    promoted, failures = promotion_verdict(CHAMPION, challenger(pinball_p90=1.70))
    assert not promoted
    assert any("pinball" in f for f in failures)


def test_calibration_regression_blocks_promotion() -> None:
    promoted, failures = promotion_verdict(CHAMPION, challenger(pr_auc=0.55, ece=0.05))
    assert not promoted  # better AUC does NOT excuse worse calibration
    assert any("ece" in f for f in failures)


def test_score_boosters_uses_probabilities_not_labels() -> None:
    """Regression: sklearn classifiers return LABELS from predict().

    The first dry run scored a challenger with 0/1 labels as 'probabilities'
    (AUC collapsed 0.80 -> 0.62). score_boosters must use predict_proba.
    """
    import numpy as np
    import polars as pl

    from dbahn_delay.models.retrain import score_boosters

    class SklearnStyleClf:
        def predict(self, x: object) -> "np.ndarray[Any, Any]":
            raise AssertionError("predict() must not be used on sklearn classifiers")

        def predict_proba(self, x: object) -> "np.ndarray[Any, Any]":
            return np.array([[0.3, 0.7], [0.9, 0.1]])

    class Reg:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, x: object) -> "np.ndarray[Any, Any]":
            return np.array([self.value, self.value])

    from dbahn_delay.models.train import FEATURES

    val = pl.DataFrame(
        {
            **{
                f: [None, None]
                for f in FEATURES
                if f not in ("station_name", "train_type", "train_number")
            },
            "station_name": ["A", "B"],
            "train_type": ["ICE", "S"],
            "train_number": ["1", "2"],
            "target_delayed6": [True, False],
            "target_delay_min": [10, 0],
        }
    )
    metrics = score_boosters(SklearnStyleClf(), Reg(5.0), Reg(20.0), val)
    assert metrics["roc_auc"] == 1.0  # 0.7 for the positive, 0.1 for the negative


def test_new_months_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrain, "local_months", lambda: ["2026-05", "2026-06"])
    monkeypatch.setattr(retrain, "remote_months", lambda: ["2026-05", "2026-06", "2026-07"])
    assert retrain.new_months() == ["2026-07"]

    monkeypatch.setattr(retrain, "remote_months", lambda: ["2026-05", "2026-06"])
    assert retrain.new_months() == []
