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


def test_board_upcoming_live_db_delay(client: TestClient, tmp_path: Path) -> None:
    """Upcoming rows expose DB's current estimate; null means no report yet."""
    now = datetime.now(tz=BERLIN)
    day = now.strftime("%Y-%m-%d")
    rows = []
    for sid in ("reported", "silent", "axed"):
        rows.append(
            {
                "stop_id": sid,
                "station_name": "Berlin Hauptbahnhof",
                "train_type": "ICE",
                "train_number": sid,
                "scheduled_time": now + timedelta(hours=1),
                "delay_probability": 0.3,
                "delay_p50_min": 2.0,
                "delay_p90_min": 15.0,
                "coverage": "train",
                "model_version": "fixture-0",
                "predicted_at": now - timedelta(hours=1),
            }
        )
    (tmp_path / "predictions").mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(tmp_path / "predictions" / f"{day}.parquet")
    (tmp_path / "changes").mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "stop_id": "reported",
                "changed_time": now + timedelta(hours=1, minutes=25),
                "is_canceled": False,
                "observed_at": now,
            },
            {
                "stop_id": "axed",
                "changed_time": None,
                "is_canceled": True,
                "observed_at": now,
            },
        ],
        schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
    ).write_parquet(tmp_path / "changes" / f"{day}.parquet")

    upcoming = {
        r["train_number"]: r for r in client.get("/board/Berlin Hauptbahnhof").json()["upcoming"]
    }
    assert upcoming["reported"]["db_live_delay_min"] == 25
    assert upcoming["silent"]["db_live_delay_min"] is None  # no report != on time
    assert upcoming["axed"]["is_canceled"] is True


def test_board_survives_midnight(client: TestClient, tmp_path: Path) -> None:
    """Trains sealed in YESTERDAY's file (e.g. the 00:30 train sealed at
    23:05) must still appear after the day flips, and a stop sealed in both
    files must keep its FIRST prediction."""
    now = datetime.now(tz=BERLIN)
    today, yesterday = now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")

    def row(sid: str, sched: datetime, predicted: datetime, prob: float) -> dict[str, object]:
        return {
            "stop_id": sid,
            "station_name": "Berlin Hauptbahnhof",
            "train_type": "ICE",
            "train_number": sid,
            "scheduled_time": sched,
            "delay_probability": prob,
            "delay_p50_min": 1.0,
            "delay_p90_min": 9.0,
            "coverage": "train",
            "model_version": "fixture-0",
            "predicted_at": predicted,
        }

    (tmp_path / "predictions").mkdir(parents=True)
    pl.DataFrame(
        [
            # departed 20 min ago but sealed yesterday -> was invisible before
            row("night-owl", now - timedelta(minutes=20), now - timedelta(hours=3), 0.2),
            # sealed in BOTH files: yesterday's (first) prediction must win
            row("double", now + timedelta(hours=1), now - timedelta(hours=2), 0.11),
        ]
    ).write_parquet(tmp_path / "predictions" / f"{yesterday}.parquet")
    pl.DataFrame(
        [
            row("double", now + timedelta(hours=1), now - timedelta(hours=1), 0.99),
            row("fresh", now + timedelta(hours=2), now - timedelta(minutes=30), 0.3),
        ]
    ).write_parquet(tmp_path / "predictions" / f"{today}.parquet")

    body = client.get("/board/Berlin Hauptbahnhof").json()
    assert [r["train_number"] for r in body["departed"]] == ["night-owl"]
    upcoming = {r["train_number"]: r for r in body["upcoming"]}
    assert set(upcoming) == {"double", "fresh"}
    assert upcoming["double"]["delay_probability"] == 0.11  # first seal wins
