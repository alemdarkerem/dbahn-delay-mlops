"""Tests for the daily output recalibration (builder + serving store)."""

import importlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from dbahn_delay.live import recalibrate
from dbahn_delay.serving.recalibration import RecalibrationStore

BERLIN = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 7, 29, 6, 15, tzinfo=BERLIN)


def seed_biased_day(live_dir: Path, n: int = 300) -> None:
    """One completed day where the model said 20% but 50% were late,
    and p90=10 while the true 90th percentile of delays is ~20."""
    day = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    (live_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (live_dir / "changes").mkdir(parents=True, exist_ok=True)
    sched = NOW - timedelta(days=1, hours=-10)  # 16:15 yesterday, long settled
    pl.DataFrame(
        [
            {
                "stop_id": f"s{i}",
                "station_name": "Berlin Hbf",
                "train_type": "ICE",
                "train_number": str(i),
                "scheduled_time": sched,
                "delay_probability": 0.2,
                "delay_p50_min": 1.0,
                "delay_p90_min": 10.0,
                "coverage": "train",
                "model_version": "test",
                "predicted_at": sched - timedelta(hours=2),
            }
            for i in range(n)
        ]
    ).write_parquet(live_dir / "predictions" / f"{day}.parquet")
    # half the stops get a 20-min delay; the other half no change (= on time)
    pl.DataFrame(
        [
            {
                "stop_id": f"s{i}",
                "changed_time": sched + timedelta(minutes=20),
                "is_canceled": False,
                "observed_at": sched + timedelta(hours=1),
            }
            for i in range(0, n, 2)
        ],
        schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
    ).write_parquet(live_dir / "changes" / f"{day}.parquet")


def test_rebuild_corrects_systematic_optimism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    monkeypatch.setattr(recalibrate, "MIN_SAMPLES", 100)
    seed_biased_day(tmp_path)

    calibration = recalibrate.rebuild(now=NOW)
    assert calibration is not None
    assert calibration["n_samples"] == 300

    store = RecalibrationStore(recalibrate.calibration_path())
    prob, p50, p90 = store.apply(0.2, 1.0, 10.0)
    assert abs(prob - 0.5) < 0.01  # "20%" now reads as the realized 50%
    assert p50 == 1.0  # p50 untouched
    # 90% of actuals are <= 20 -> delta pushes p90 from 10 toward 20
    assert 19.0 <= p90 <= 21.0
    assert store.info() is not None and store.info()["n_samples"] == 300  # type: ignore[index]


def test_rebuild_keeps_identity_below_min_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    seed_biased_day(tmp_path, n=50)  # < MIN_SAMPLES (2000)
    assert recalibrate.rebuild(now=NOW) is None
    assert not recalibrate.calibration_path().exists()


def test_store_identity_without_file(tmp_path: Path) -> None:
    store = RecalibrationStore(tmp_path / "calibration.json")
    assert store.apply(0.3, 2.0, 15.0) == (0.3, 2.0, 15.0)
    assert store.info() is None


def test_store_reloads_and_survives_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"curve_x": [0.0, 1.0], "curve_y": [0.1, 0.9], "p90_delta_min": 5}))
    store = RecalibrationStore(path)
    prob, _, p90 = store.apply(0.5, 1.0, 10.0)
    assert abs(prob - 0.5) < 1e-9  # midpoint of the linear curve
    assert p90 == 15.0

    path.write_text("{broken")
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    assert store.apply(0.5, 1.0, 10.0) == (0.5, 1.0, 10.0)  # falls back to identity


def test_predict_applies_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: /predict output shifts once a calibration file exists."""
    os.environ["DBAHN_MODEL_DIR"] = str(Path(__file__).parent / "fixtures" / "bundle")
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    import dbahn_delay.serving.app as app_module

    importlib.reload(app_module)
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    request = {
        "station_name": "Berlin Hbf",
        "train_type": "ICE",
        "train_number": "1601",
        "scheduled_time": "2026-07-02T17:30:00",
        "train_line_station_num": 3,
    }
    raw = client.post("/predict", json=request).json()

    calib_dir = tmp_path / "recalibration"
    calib_dir.mkdir(parents=True)
    (calib_dir / "calibration.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-29T06:15:00+02:00",
                "n_samples": 5000,
                "curve_x": [0.0, 1.0],
                "curve_y": [0.5, 1.0],  # blunt upward correction
                "p90_delta_min": 7.0,
            }
        )
    )
    corrected = client.post("/predict", json=request).json()
    assert corrected["delay_probability"] > raw["delay_probability"]
    assert corrected["delay_p90_min"] == round(raw["delay_p90_min"] + 7.0, 1)
    info = client.get("/model-info").json()["recalibration"]
    assert info["n_samples"] == 5000

    # reload for other test modules: this module's app instance pinned live_dir
    importlib.reload(app_module)
