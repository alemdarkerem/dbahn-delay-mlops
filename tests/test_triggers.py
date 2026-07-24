"""Tests for retraining trigger rules over the daily metric series."""

import polars as pl

from dbahn_delay.live.evaluate_day import trigger_reasons


def series(ece: list[float], auc: list[float]) -> pl.DataFrame:
    days = [f"2026-07-{20 + i:02d}" for i in range(len(ece))]
    return pl.DataFrame({"day": days, "ece": ece, "roc_auc": auc})


def test_healthy_series_triggers_nothing() -> None:
    s = series([0.02, 0.03, 0.02], [0.78, 0.77, 0.79])
    assert trigger_reasons(s) == []


def test_three_bad_ece_days_trigger() -> None:
    s = series([0.02, 0.09, 0.10, 0.09], [0.78] * 4)
    reasons = trigger_reasons(s)
    assert any("ece" in r for r in reasons)


def test_two_bad_days_do_not_trigger_yet() -> None:
    s = series([0.02, 0.09, 0.10], [0.78] * 3)
    assert trigger_reasons(s) == []  # last three are 0.02/0.09/0.10 -> not ALL bad


def test_auc_floor_trigger() -> None:
    s = series([0.02] * 3, [0.65, 0.66, 0.64])
    reasons = trigger_reasons(s)
    assert any("roc_auc" in r for r in reasons)


def test_freshness_trigger_independent_of_series_length() -> None:
    s = series([0.02], [0.78])  # only one day of history
    assert trigger_reasons(s, feature_freshness_days=50) != []
    assert trigger_reasons(s, feature_freshness_days=10) == []
