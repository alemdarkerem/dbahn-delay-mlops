"""Tests for the feature pipeline — above all: NO LEAKAGE.

The leak tests construct tiny frames where "today" has an extreme delay and
assert that today's rolling features cannot see it.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from dbahn_delay.features.build import (
    add_calendar_features,
    add_rolling_stats,
    add_scheduled_time,
    build_feature_frame,
)


def stops_frame(rows: list[dict[str, object]]) -> pl.LazyFrame:
    base: dict[str, object] = {
        "station_name": "Berlin Hbf",
        "xml_station_name": "Berlin Hbf",
        "eva": "8011160",
        "train_number": "1601",
        "line_number": None,
        "final_destination_station": "München Hbf",
        "delay_in_min": 0,
        "time": datetime(2025, 1, 15, 8, 30),
        "is_canceled": False,
        "train_type": "ICE",
        "train_line_ride_id": "ride-1",
        "train_line_station_num": 3,
        "arrival_planned_time": None,
        "arrival_change_time": None,
        "departure_planned_time": datetime(2025, 1, 15, 8, 28),
        "departure_change_time": None,
        "id": "stop-1",
    }
    out = []
    for i, row in enumerate(rows):
        merged = base | row
        merged.setdefault("id", f"stop-{i}")
        out.append(merged)
    ts_cols = (
        "time",
        "arrival_planned_time",
        "arrival_change_time",
        "departure_planned_time",
        "departure_change_time",
    )
    # Interpret naive datetimes as Berlin wall-clock, exactly like ingest does.
    return pl.LazyFrame(
        out,
        schema_overrides={
            "delay_in_min": pl.Int32,
            "train_line_station_num": pl.Int32,
            **{c: pl.Datetime("us") for c in ts_cols},
        },
    ).with_columns(pl.col(c).dt.replace_time_zone("Europe/Berlin") for c in ts_cols)


def at(day: int, hour: int = 8) -> dict[str, object]:
    return {
        "time": datetime(2025, 1, day, hour, 30),
        "departure_planned_time": datetime(2025, 1, day, hour, 28),
    }


def test_scheduled_time_prefers_planned_departure() -> None:
    out = stops_frame([{}]).pipe(add_scheduled_time).collect()
    assert out["scheduled_time"][0] == out["departure_planned_time"][0]


def test_scheduled_time_falls_back_to_actual_minus_delay() -> None:
    out = (
        stops_frame([{"departure_planned_time": None, "delay_in_min": 10}])
        .pipe(add_scheduled_time)
        .collect()
    )
    # 08:30 actual - 10 min delay = 08:20 scheduled (Berlin wall-clock)
    assert out["scheduled_time"][0] == datetime(
        2025, 1, 15, 8, 20, tzinfo=ZoneInfo("Europe/Berlin")
    )


def test_calendar_features_holiday_and_weekend() -> None:
    rows: list[dict[str, object]] = [
        {"departure_planned_time": datetime(2025, 1, 1, 10, 0)},  # New Year, Wednesday
        {"departure_planned_time": datetime(2025, 1, 4, 10, 0)},  # Saturday
    ]
    out = stops_frame(rows).pipe(add_scheduled_time).pipe(add_calendar_features).collect()
    assert out["is_holiday"].to_list() == [True, False]
    assert out["is_weekend"].to_list() == [False, True]


def test_rolling_stats_exclude_same_day() -> None:
    """The leak test: today's 100-min delay must NOT appear in today's stats."""
    rows = [
        at(day=10) | {"delay_in_min": 4},
        at(day=11) | {"delay_in_min": 100},  # today's extreme delay
    ]
    out = (
        stops_frame(rows)
        .pipe(add_scheduled_time)
        .pipe(add_calendar_features)
        .pipe(add_rolling_stats)
        .collect()
        .sort("scheduled_time")
    )
    # Day 11's window covers day 10 only: mean must be 4, not (4+100)/2
    assert out["train_mean_delay_14d"][1] == 4.0
    # Day 10 has no history at all -> null, not 4
    assert out["train_mean_delay_14d"][0] is None


def test_rolling_stats_respect_window_length() -> None:
    """A delay 20 days ago is outside the 14d window but inside the 30d one."""
    rows = [
        {"delay_in_min": 60, **at(day=1)},
        {"delay_in_min": 0, **at(day=21)},
    ]
    out = (
        stops_frame(rows)
        .pipe(add_scheduled_time)
        .pipe(add_calendar_features)
        .pipe(add_rolling_stats)
        .collect()
        .sort("scheduled_time")
    )
    row = out.filter(pl.col("scheduled_time").dt.day() == 21)
    assert row["train_mean_delay_14d"][0] is None  # nothing within 14d
    assert row["train_mean_delay_30d"][0] == 60.0  # day 1 within 30d


def test_canceled_rows_excluded_from_history_and_targets() -> None:
    rows = [
        at(day=10) | {"delay_in_min": 50, "is_canceled": True},
        at(day=10, hour=9) | {"delay_in_min": 2},
        at(day=11) | {"delay_in_min": 0},
    ]
    out = stops_frame(rows).pipe(build_feature_frame).collect().sort("scheduled_time")
    # Canceled row is not in the output at all
    assert out.height == 2
    # Day 11 history saw only the non-canceled 2-min row, not the canceled 50
    assert out["train_mean_delay_14d"][1] == 2.0


def test_feature_frame_has_targets_and_fold_key() -> None:
    out = stops_frame([at(day=10) | {"delay_in_min": 7}]).pipe(build_feature_frame).collect()
    assert out["target_delayed6"][0] is True
    assert out["target_delay_min"][0] == 7
    assert out["fold_month"][0] == "2025-01"
