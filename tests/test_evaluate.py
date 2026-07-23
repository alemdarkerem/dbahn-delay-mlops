"""Sanity tests for the evaluation metrics on hand-checkable arrays."""

import numpy as np
import pytest

from dbahn_delay.models.evaluate import (
    coverage,
    expected_calibration_error,
    pinball_loss,
    quantile_metrics,
)


def test_pinball_loss_penalizes_underprediction_more_at_high_quantiles() -> None:
    y = np.array([10.0])
    under = pinball_loss(y, np.array([0.0]), alpha=0.9)  # 0.9 * 10 = 9
    over = pinball_loss(y, np.array([20.0]), alpha=0.9)  # 0.1 * 10 = 1
    assert under == pytest.approx(9.0)
    assert over == pytest.approx(1.0)


def test_coverage_matches_hand_count() -> None:
    y = np.array([1.0, 5.0, 10.0, 20.0])
    q = np.array([2.0, 4.0, 15.0, 25.0])
    assert coverage(y, q) == 0.75  # 5.0 > 4.0 is the only miss


def test_ece_zero_for_perfectly_calibrated() -> None:
    rng = np.random.default_rng(42)
    prob = np.full(10_000, 0.3)
    y = (rng.random(10_000) < 0.3).astype(float)
    assert expected_calibration_error(y, prob) < 0.02


def test_quantile_metrics_keys() -> None:
    y = np.array([0.0, 5.0, 12.0])
    m = quantile_metrics(y, p50=np.array([1.0, 4.0, 10.0]), p90=np.array([5.0, 9.0, 30.0]))
    assert set(m) == {"pinball_p50", "pinball_p90", "mae_p50", "coverage_p90"}
