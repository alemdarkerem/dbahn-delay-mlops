"""Tests for the walk-forward splitter and the naive baselines."""

import polars as pl

from dbahn_delay.models.baseline import fit_baseline, predict_baseline
from dbahn_delay.models.cv import month_range, walk_forward_folds


def test_month_range_crosses_year_boundary() -> None:
    assert month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_walk_forward_folds_are_chronological_and_leak_free() -> None:
    folds = walk_forward_folds(["2026-01", "2026-02"], train_window_months=12)
    assert folds[0].train_months[0] == "2025-01"
    assert folds[0].train_months[-1] == "2025-12"
    assert folds[0].val_month == "2026-01"
    # Rolling window slides with the validation month
    assert folds[1].train_months[0] == "2025-02"
    assert folds[1].train_months[-1] == "2026-01"
    # No fold ever contains its validation month in training
    for fold in folds:
        assert fold.val_month not in fold.train_months
        assert all(m < fold.val_month for m in fold.train_months)


def feature_frame(rows: list[dict[str, object]]) -> pl.LazyFrame:
    base: dict[str, object] = {
        "station_name": "Berlin Hbf",
        "train_type": "ICE",
        "target_delayed6": False,
        "target_delay_min": 0,
    }
    return pl.LazyFrame([base | r for r in rows])


def test_baseline_learns_group_rates_and_falls_back() -> None:
    train = feature_frame(
        [
            {"target_delayed6": True, "target_delay_min": 20},
            {"target_delayed6": True, "target_delay_min": 10},
            {"target_delayed6": False, "target_delay_min": 0},
            {"target_delayed6": False, "target_delay_min": 0},
            # different type to make fallback distinguishable
            {"train_type": "S", "target_delayed6": False, "target_delay_min": 1},
        ]
    )
    tables = fit_baseline(train)

    val = feature_frame(
        [
            {},  # known station x type -> group rate 0.5
            {"station_name": "Unknown Hbf"},  # unseen station -> ICE fallback 0.5
            {"station_name": "Unknown Hbf", "train_type": "X"},  # fully unseen -> global
        ]
    )
    out = predict_baseline(tables, val)
    assert out["baseline_prob"][0] == 0.5
    assert out["baseline_prob"][1] == 0.5
    assert out["baseline_prob"][2] == tables.global_rate
    # p90 of the ICE group must dominate its p50
    assert out["baseline_p90"][0] >= out["baseline_p50"][0]
