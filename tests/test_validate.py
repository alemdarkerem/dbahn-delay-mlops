"""Unit tests for the validation checks, using small synthetic frames."""

from datetime import datetime

import polars as pl

from dbahn_delay.data.validate import (
    EXPECTED_SCHEMA,
    check_delay_range,
    check_duplicate_ids,
    check_required_not_null,
    check_schema,
    check_time_range,
    validate,
)


def make_valid_frame() -> pl.LazyFrame:
    """One well-formed row matching the expected raw schema."""
    row = {
        "station_name": "Berlin Hbf",
        "xml_station_name": "Berlin Hbf",
        "eva": "8011160",
        "train_name": "ICE 123",
        "final_destination_station": "München Hbf",
        "delay_in_min": 4,
        "time": datetime(2025, 1, 15, 8, 30),
        "is_canceled": False,
        "train_type": "ICE",
        "train_line_ride_id": "ride-1",
        "train_line_station_num": 3,
        "arrival_planned_time": datetime(2025, 1, 15, 8, 26),
        "arrival_change_time": datetime(2025, 1, 15, 8, 30),
        "departure_planned_time": datetime(2025, 1, 15, 8, 28),
        "departure_change_time": datetime(2025, 1, 15, 8, 32),
        "id": "stop-1",
    }
    return pl.LazyFrame([row], schema=EXPECTED_SCHEMA)


def test_valid_frame_passes_all_checks() -> None:
    report = validate(
        make_valid_frame(),
        "synthetic",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 2, 1),
    )
    assert report.passed, report.summary()


def test_schema_check_catches_missing_and_mistyped_columns() -> None:
    missing = make_valid_frame().drop("eva")
    assert not check_schema(missing).passed

    mistyped = make_valid_frame().with_columns(pl.col("delay_in_min").cast(pl.Float64))
    assert not check_schema(mistyped).passed


def test_required_not_null_catches_null_station() -> None:
    lf = make_valid_frame().with_columns(station_name=pl.lit(None, dtype=pl.String))
    assert not check_required_not_null(lf).passed


def test_required_not_null_fails_on_empty_frame() -> None:
    lf = make_valid_frame().filter(pl.lit(False))
    assert not check_required_not_null(lf).passed


def test_delay_range_catches_absurd_delay() -> None:
    lf = make_valid_frame().with_columns(delay_in_min=pl.lit(100_000, dtype=pl.Int64))
    result = check_delay_range(lf)
    assert not result.passed
    assert "1 rows" in result.details


def test_duplicate_ids_detected() -> None:
    lf = pl.concat([make_valid_frame(), make_valid_frame()])
    assert not check_duplicate_ids(lf).passed


def test_time_range_catches_event_outside_window() -> None:
    result = check_time_range(
        make_valid_frame(), start=datetime(2025, 3, 1), end=datetime(2025, 4, 1)
    )
    assert not result.passed
