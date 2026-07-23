"""Unit tests for the ingest transforms on synthetic data."""

from datetime import datetime

import polars as pl

from dbahn_delay.data.ingest import dedupe, drop_bad_rows, station_panel, to_berlin_time


def frame(rows: list[dict[str, object]]) -> pl.LazyFrame:
    base: dict[str, object] = {
        "station_name": "Berlin Hbf",
        "train_type": "ICE",
        "delay_in_min": 0,
        "time": datetime(2025, 1, 15, 8, 30),
        "arrival_planned_time": None,
        "arrival_change_time": None,
        "departure_planned_time": None,
        "departure_change_time": None,
        "id": "stop-1",
    }
    return pl.LazyFrame(
        [base | row for row in rows],
        schema_overrides={
            "delay_in_min": pl.Int32,
            "time": pl.Datetime("ns"),
            "arrival_planned_time": pl.Datetime("ns"),
            "arrival_change_time": pl.Datetime("ns"),
            "departure_planned_time": pl.Datetime("ns"),
            "departure_change_time": pl.Datetime("ns"),
        },
    )


def test_to_berlin_time_attaches_timezone() -> None:
    out = frame([{}]).pipe(to_berlin_time).collect()
    assert out["time"].dtype == pl.Datetime("us", "Europe/Berlin")
    # Wall-clock value is preserved, only the zone is attached
    assert out["time"][0].hour == 8


def test_to_berlin_time_resolves_ambiguous_fall_back_hour() -> None:
    # 2025-10-26 02:30 happens twice in Berlin; we resolve to the earliest (CEST)
    out = frame([{"time": datetime(2025, 10, 26, 2, 30)}]).pipe(to_berlin_time).collect()
    assert str(out["time"][0].utcoffset()) == "2:00:00"


def test_drop_bad_rows_removes_nulls_and_absurd_delays() -> None:
    lf = frame(
        [
            {},
            {"station_name": None},
            {"delay_in_min": 10_000},
            {"delay_in_min": -500},
        ]
    )
    assert drop_bad_rows(lf).collect().height == 1


def test_dedupe_keeps_one_row_per_id() -> None:
    lf = frame([{"id": "a"}, {"id": "a"}, {"id": "b"}])
    assert dedupe(lf).collect().height == 2


def test_station_panel_keeps_only_stations_present_every_month() -> None:
    lf = frame(
        [
            {"station_name": "Always", "time": datetime(2025, 1, 1)},
            {"station_name": "Always", "time": datetime(2025, 2, 1)},
            {"station_name": "OnlyFeb", "time": datetime(2025, 2, 1)},
        ]
    )
    assert station_panel(lf) == ["Always"]
