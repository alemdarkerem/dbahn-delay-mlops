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


def seed_day(
    live_dir: Path,
    days_back: int,
    n: int,
    *,
    said: float = 0.2,
    p90: float = 10.0,
    late_share: float = 0.5,
    late_minutes: int = 20,
    model_version: str = "test",
    prefix: str = "s",
    raw: tuple[float, float] | None = None,
) -> None:
    """One settled day of sealed predictions + observed outcomes.

    Default shape: the model said 20% and p90=10, but half the trains were
    20 minutes late — i.e. systematically optimistic in both directions.
    """
    day = (NOW - timedelta(days=days_back)).strftime("%Y-%m-%d")
    (live_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (live_dir / "changes").mkdir(parents=True, exist_ok=True)
    sched = NOW - timedelta(days=days_back, hours=-10)  # 16:15 that day
    row: dict[str, object] = {
        "station_name": "Berlin Hbf",
        "train_type": "ICE",
        "scheduled_time": sched,
        "delay_probability": said,
        "delay_p50_min": 1.0,
        "delay_p90_min": p90,
        "coverage": "train",
        "model_version": model_version,
        "predicted_at": sched - timedelta(hours=2),
    }
    if raw is not None:
        row |= {"raw_delay_probability": raw[0], "raw_delay_p90_min": raw[1]}
    pl.DataFrame(
        [row | {"stop_id": f"{prefix}{days_back}-{i}", "train_number": str(i)} for i in range(n)]
    ).write_parquet(live_dir / "predictions" / f"{day}.parquet")
    every = max(1, round(1 / late_share))
    pl.DataFrame(
        [
            {
                "stop_id": f"{prefix}{days_back}-{i}",
                "changed_time": sched + timedelta(minutes=late_minutes),
                "is_canceled": False,
                "observed_at": sched + timedelta(hours=1),
            }
            for i in range(0, n, every)
        ],
        schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
    ).write_parquet(live_dir / "changes" / f"{day}.parquet")


def seed_biased_day(live_dir: Path, n: int = 300, **kwargs: object) -> None:
    """Two settled days of the same bias: an older one to fit the curve on
    and yesterday's to validate it against doing nothing."""
    seed_day(live_dir, days_back=2, n=n // 2, **kwargs)  # type: ignore[arg-type]
    seed_day(live_dir, days_back=1, n=n - n // 2, **kwargs)  # type: ignore[arg-type]


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

    # the newly promoted model has been sealing predictions on both days too
    for days_back in (2, 1):
        day = (NOW - timedelta(days=days_back)).strftime("%Y-%m-%d")
        pred_path = tmp_path / "predictions" / f"{day}.parquet"
        change_path = tmp_path / "changes" / f"{day}.parquet"
        old_pred, old_changes = pl.read_parquet(pred_path), pl.read_parquet(change_path)
        seed_day(tmp_path, days_back=days_back, n=120, model_version="v2-newer", prefix="new")
        pl.concat([old_pred, pl.read_parquet(pred_path)]).write_parquet(pred_path)
        pl.concat([old_changes, pl.read_parquet(change_path)]).write_parquet(change_path)

    calibration = recalibrate.rebuild(now=NOW)
    assert calibration is not None
    assert calibration["model_version"] == "v2-newer"
    assert calibration["n_samples"] == 240  # old model's pairs excluded

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


def test_correction_stands_down_when_the_model_is_already_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The layer must not "fix" a healthy model.

    Live on 2026-08-15, right after a train/serve skew was repaired, the raw
    model scored ECE 0.009 while the stale curve pushed it to 0.088. A
    correction now has to beat raw outputs on a held-out day or it is
    dropped — and an existing file goes with it.
    """
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    monkeypatch.setattr(recalibrate, "MIN_SAMPLES", 100)
    # honest model: says 50% and 50% are late; p90=20 covers every actual
    seed_biased_day(tmp_path, n=400, said=0.5, p90=20.0, late_share=0.5, late_minutes=10)
    recalibrate.calibration_path().parent.mkdir(parents=True, exist_ok=True)
    recalibrate.calibration_path().write_text(
        json.dumps(
            {
                "model_version": "test",
                "curve_x": [0.0, 1.0],
                "curve_y": [0.4, 1.0],  # yesterday's aggressive upward push
                "p90_delta_min": 8.0,
            }
        )
    )

    assert recalibrate.rebuild(now=NOW) is None
    assert not recalibrate.calibration_path().exists()  # harmful curve removed
    assert RecalibrationStore(recalibrate.calibration_path()).apply(0.5, 1.0, 20.0) == (
        0.5,
        1.0,
        20.0,
    )


def test_validation_numbers_are_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the correction does earn its place, the report says by how much."""
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    monkeypatch.setattr(recalibrate, "MIN_SAMPLES", 100)
    seed_biased_day(tmp_path, n=400)

    calibration = recalibrate.rebuild(now=NOW)
    assert calibration is not None
    checks = calibration["validation"]
    assert checks["probability_correction_helps"] and checks["p90_correction_helps"]
    assert checks["ece_corrected"] < checks["ece_raw"]
    assert abs(checks["coverage_corrected"] - 0.9) < abs(checks["coverage_raw"] - 0.9)
