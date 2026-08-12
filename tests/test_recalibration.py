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


def test_calibration_is_scoped_to_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A curve fitted for the previous model must not correct a new one.

    After a promotion the old bias would push the fresh model the wrong way,
    so pairs are filtered to the newest model version and the serving store
    ignores a curve stamped with a different version.
    """
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    monkeypatch.setattr(recalibrate, "MIN_SAMPLES", 100)
    seed_biased_day(tmp_path)

    # a handful of stops already sealed by the newly promoted model
    day = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    path = tmp_path / "predictions" / f"{day}.parquet"
    old = pl.read_parquet(path)
    fresh = old.head(120).with_columns(
        stop_id=pl.col("stop_id") + "-new", model_version=pl.lit("v2-newer")
    )
    pl.concat([old, fresh]).write_parquet(path)

    calibration = recalibrate.rebuild(now=NOW)
    assert calibration is not None
    assert calibration["model_version"] == "v2-newer"
    assert calibration["n_samples"] == 120  # old model's pairs excluded

    matching = RecalibrationStore(recalibrate.calibration_path(), model_version="v2-newer")
    assert matching.apply(0.2, 1.0, 10.0)[0] != 0.2  # correction applies

    stale = RecalibrationStore(recalibrate.calibration_path(), model_version="v3-promoted")
    assert stale.apply(0.2, 1.0, 10.0) == (0.2, 1.0, 10.0)  # raw outputs
    assert stale.info() is None


def test_fit_uses_raw_outputs_not_corrected_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the curve must learn from what the MODEL said, not from
    what yesterday's correction turned it into. Fitting on corrected values
    and applying the result to raw ones under-corrects, and the daily loop
    oscillates (ECE 0.021 -> 0.066 over a week, live 2026-08)."""
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    monkeypatch.setattr(recalibrate, "MIN_SAMPLES", 100)
    seed_biased_day(tmp_path)

    # Rows sealed under an active calibration: the model said 20%, the
    # correction shipped 50%, and 50% of them were indeed late.
    day = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    path = tmp_path / "predictions" / f"{day}.parquet"
    pl.read_parquet(path).with_columns(
        delay_probability=pl.lit(0.5),
        delay_p90_min=pl.lit(20.0),
        raw_delay_probability=pl.lit(0.2),
        raw_delay_p90_min=pl.lit(10.0),
    ).write_parquet(path)

    recalibrate.rebuild(now=NOW)
    store = RecalibrationStore(recalibrate.calibration_path())
    prob, _, p90 = store.apply(0.2, 1.0, 10.0)
    # Fitted on the raw 20% -> still maps to the realized 50% (a fit on the
    # corrected 50% would have produced a near-identity curve instead).
    assert abs(prob - 0.5) < 0.01
    assert 19.0 <= p90 <= 21.0


def test_rebuild_drops_calibration_left_by_previous_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promotion with too few fresh pairs must not keep serving the old
    model's correction — the stale file is removed rather than left behind."""
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    seed_biased_day(tmp_path, n=50)  # fewer pairs than MIN_SAMPLES
    path = recalibrate.calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_version": "previous-model",
                "curve_x": [0.0, 1.0],
                "curve_y": [0.5, 1.0],
                "p90_delta_min": 9.0,
            }
        )
    )

    assert recalibrate.rebuild(now=NOW) is None  # cannot refit yet
    assert not path.exists()  # but the stale curve is gone


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
