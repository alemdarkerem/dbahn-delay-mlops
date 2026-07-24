"""Tests for the live loop: XML parsing, sealed logging, daily evaluation."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from dbahn_delay.live.evaluate_day import evaluate
from dbahn_delay.live.fetch import append_new_predictions, upsert_changes
from dbahn_delay.live.parse import Change, parse_changes, parse_plan

BERLIN = ZoneInfo("Europe/Berlin")

# Trimmed real responses captured from the Timetables API (2026-07-23).
PLAN_XML = """<?xml version='1.0' encoding='UTF-8'?>
<timetable station='Berlin Hbf'>
  <s id="-1854738006003716349-2607231847-1">
    <tl f="F" t="p" o="80" c="ICE" n="542"/>
    <dp pt="2607231847" pp="7" fb="ICE 542" ppth="Wolfsburg Hbf|Koeln Hbf"/>
  </s>
  <s id="2173268028498717892-2607231634-2">
    <tl f="F" t="p" o="80" c="ICE" n="605"/>
    <ar pt="2607231822" pp="3" fb="ICE 605" ppth="Hamburg Hbf"/>
  </s>
  <s id="re7-line-label">
    <tl f="N" t="p" o="800165" c="RE" n="17025"/>
    <dp pt="2607231850" pp="2" l="7" ppth="Karlsruhe Hbf"/>
  </s>
  <s id="broken-no-time"><tl c="RE" n="1"/><dp pp="1"/></s>
</timetable>"""

CHANGES_XML = """<?xml version='1.0' encoding='UTF-8'?>
<timetable station='Berlin Hbf'>
  <s id="stop-delayed"><dp ct="2607231859" pt="2607231847"/></s>
  <s id="stop-canceled"><dp cs="c" pt="2607231900"/></s>
  <s id="stop-no-ct"><ar pp="4"/></s>
</timetable>"""


def test_parse_plan_extracts_stops_and_skips_broken() -> None:
    stops = parse_plan(PLAN_XML)
    assert len(stops) == 3  # broken row without pt skipped
    re7 = next(st for st in stops if st.stop_id == "re7-line-label")
    assert re7.line == "7"  # passenger-facing label captured
    assert re7.train_number == "17025"
    ice542 = stops[0]
    assert ice542.train_type == "ICE"
    assert ice542.train_number == "542"
    assert ice542.scheduled_time == datetime(2026, 7, 23, 18, 47, tzinfo=BERLIN)
    assert ice542.has_departure
    assert stops[0].line is None  # long-distance trains often carry no line
    assert not stops[1].has_departure  # arrival-only stop (terminus)


def test_parse_changes_delay_and_cancellation() -> None:
    changes = parse_changes(CHANGES_XML)
    by_id = {c.stop_id: c for c in changes}
    assert by_id["stop-delayed"].changed_time == datetime(2026, 7, 23, 18, 59, tzinfo=BERLIN)
    assert not by_id["stop-delayed"].is_canceled
    assert by_id["stop-canceled"].is_canceled
    assert by_id["stop-no-ct"].changed_time is None


def prediction_frame(stop_ids: list[str]) -> pl.DataFrame:
    now = datetime(2026, 7, 23, 16, 0, tzinfo=BERLIN)
    return pl.DataFrame(
        [
            {
                "stop_id": sid,
                "station_name": "Berlin Hbf",
                "train_type": "ICE",
                "train_number": "542",
                "scheduled_time": datetime(2026, 7, 23, 18, 47, tzinfo=BERLIN),
                "delay_probability": 0.5,
                "delay_p50_min": 4.0,
                "delay_p90_min": 20.0,
                "coverage": "train",
                "model_version": "test",
                "predicted_at": now,
            }
            for sid in stop_ids
        ]
    )


def test_append_new_predictions_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    day = "2026-07-23"
    assert append_new_predictions(prediction_frame(["a", "b"]), day) == 2
    # Second cycle sees the same stops plus one new -> only the new one lands
    assert append_new_predictions(prediction_frame(["a", "b", "c"]), day) == 1
    stored = pl.read_parquet(tmp_path / "predictions" / f"{day}.parquet")
    assert stored.height == 3
    assert stored["stop_id"].n_unique() == 3


def test_upsert_changes_keeps_latest_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    day = "2026-07-23"
    t1 = datetime(2026, 7, 23, 17, 0, tzinfo=BERLIN)
    t2 = datetime(2026, 7, 23, 18, 0, tzinfo=BERLIN)
    upsert_changes([Change("s1", datetime(2026, 7, 23, 18, 50, tzinfo=BERLIN), False)], t1, day)
    upsert_changes([Change("s1", datetime(2026, 7, 23, 18, 59, tzinfo=BERLIN), False)], t2, day)
    stored = pl.read_parquet(tmp_path / "changes" / f"{day}.parquet")
    assert stored.height == 1
    assert stored["changed_time"][0] == datetime(2026, 7, 23, 18, 59, tzinfo=BERLIN)


def test_predict_stops_uses_injected_predictor_and_skips_failures() -> None:
    from dbahn_delay.live.fetch import predict_stops
    from dbahn_delay.live.parse import PlannedStop

    stops = [
        PlannedStop(
            "s1", "Berlin Hbf", "ICE", "542", datetime(2026, 7, 23, 18, 47, tzinfo=BERLIN), True
        ),
        PlannedStop(
            "s2", "Berlin Hbf", "S", "7", datetime(2026, 7, 23, 18, 50, tzinfo=BERLIN), True
        ),
    ]

    def stub_predict(stop: PlannedStop) -> dict[str, object] | None:
        if stop.stop_id == "s2":
            return None  # API hiccup: row skipped, cycle survives
        return {
            "delay_probability": 0.42,
            "delay_p50_min": 3.0,
            "delay_p90_min": 21.0,
            "coverage": "train",
            "model_version": "test",
        }

    out = predict_stops(stub_predict, stops, datetime(2026, 7, 23, 16, 0, tzinfo=BERLIN))
    assert out.height == 1
    assert out["stop_id"][0] == "s1"
    assert out["delay_probability"][0] == 0.42


def test_evaluate_day_metrics_hand_checked() -> None:
    predictions = prediction_frame(["on-time", "delayed", "canceled"])
    changes = pl.DataFrame(
        [
            {  # 12 min late: 18:47 -> 18:59
                "stop_id": "delayed",
                "changed_time": datetime(2026, 7, 23, 18, 59, tzinfo=BERLIN),
                "is_canceled": False,
                "observed_at": datetime(2026, 7, 23, 19, 0, tzinfo=BERLIN),
            },
            {
                "stop_id": "canceled",
                "changed_time": None,
                "is_canceled": True,
                "observed_at": datetime(2026, 7, 23, 19, 0, tzinfo=BERLIN),
            },
        ],
        schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
    )
    metrics = evaluate(predictions, changes)
    assert metrics["n_predictions"] == 3.0
    assert metrics["n_canceled"] == 1.0
    assert metrics["n_evaluated"] == 2.0
    # one on-time (0 min, no change record) + one 12-min delayed -> base rate 0.5
    assert metrics["base_rate"] == 0.5
    # p90 was 20: covers 0 and 12 -> coverage 1.0
    assert metrics["coverage_p90"] == 1.0
    # MAE of p50=4: |0-4|=4 and |12-4|=8 -> mean 6
    assert metrics["mae_p50"] == 6.0
