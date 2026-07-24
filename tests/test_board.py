"""Tests for the station-board endpoints and page."""

import importlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from fastapi.testclient import TestClient

FIXTURE_BUNDLE = Path(__file__).parent / "fixtures" / "bundle"
BERLIN = ZoneInfo("Europe/Berlin")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    os.environ["DBAHN_MODEL_DIR"] = str(FIXTURE_BUNDLE)
    import dbahn_delay.serving.app as app_module

    importlib.reload(app_module)
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    return TestClient(app_module.app)


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "DB Delay Predictions" in response.text


def test_stations_lists_panel(client: TestClient) -> None:
    body = client.get("/stations").json()
    assert len(body["stations"]) == 105
    assert "Berlin Hauptbahnhof" in body["stations"]


def test_board_without_data_returns_note(client: TestClient) -> None:
    body = client.get("/board/Berlin Hauptbahnhof").json()
    assert body["upcoming"] == []
    assert body["departed"] == []
    assert "note" in body


def test_board_dedupes_wing_train_rows(client: TestClient, tmp_path: Path) -> None:
    """Two stop ids for the same train+minute render as one board row."""
    now = datetime.now(tz=BERLIN)
    day = now.strftime("%Y-%m-%d")
    base = {
        "station_name": "Berlin Hauptbahnhof",
        "train_type": "ICE",
        "train_number": "241",
        "scheduled_time": now + timedelta(hours=1),
        "delay_probability": 0.45,
        "delay_p50_min": 7.0,
        "delay_p90_min": 31.0,
        "coverage": "train",
        "model_version": "fixture-0",
        "predicted_at": now,
    }
    rows = [base | {"stop_id": "wing-a"}, base | {"stop_id": "wing-b"}]
    (tmp_path / "predictions").mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(tmp_path / "predictions" / f"{day}.parquet")

    body = client.get("/board/Berlin Hauptbahnhof").json()
    assert len(body["upcoming"]) == 1


def test_board_splits_upcoming_and_departed(client: TestClient, tmp_path: Path) -> None:
    now = datetime.now(tz=BERLIN)
    day = now.strftime("%Y-%m-%d")
    rows = []
    for offset, sid in ((-2, "gone"), (2, "soon")):
        rows.append(
            {
                "stop_id": sid,
                "station_name": "Berlin Hauptbahnhof",
                "train_type": "ICE",
                "train_number": "1601",
                "scheduled_time": now + timedelta(hours=offset),
                "delay_probability": 0.3,
                "delay_p50_min": 2.0,
                "delay_p90_min": 15.0,
                "coverage": "train",
                "model_version": "fixture-0",
                "predicted_at": now - timedelta(hours=3),
            }
        )
    (tmp_path / "predictions").mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(tmp_path / "predictions" / f"{day}.parquet")
    (tmp_path / "changes").mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "stop_id": "gone",
                "changed_time": now - timedelta(hours=2) + timedelta(minutes=12),
                "is_canceled": False,
                "observed_at": now,
            }
        ],
        schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
    ).write_parquet(tmp_path / "changes" / f"{day}.parquet")

    body = client.get("/board/Berlin Hauptbahnhof").json()
    assert [r["train_number"] for r in body["upcoming"]] == ["1601"]
    assert body["departed"][0]["actual_delay_min"] == 12
    # the observed 12-min delay sits inside the predicted p90 of 15
    assert body["departed"][0]["actual_delay_min"] <= body["departed"][0]["delay_p90_min"]
