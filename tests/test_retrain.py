"""Tests for promotion rules and new-month detection."""

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


def test_new_months_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrain, "local_months", lambda: ["2026-05", "2026-06"])
    monkeypatch.setattr(retrain, "remote_months", lambda: ["2026-05", "2026-06", "2026-07"])
    assert retrain.new_months() == ["2026-07"]

    monkeypatch.setattr(retrain, "remote_months", lambda: ["2026-05", "2026-06"])
    assert retrain.new_months() == []
