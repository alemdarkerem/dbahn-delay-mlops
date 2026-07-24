"""Tests for the live snapshot overlay: build, serve-side priority, reload."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from dbahn_delay.live.refresh_snapshot import refresh
from dbahn_delay.serving.overlay import OverlayStore

BERLIN = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 7, 24, 6, 30, tzinfo=BERLIN)


def seed_live_data(live_dir: Path) -> None:
    """Two days of observations for one train: delays 10 and 20 min."""
    (live_dir / "predictions").mkdir(parents=True)
    (live_dir / "changes").mkdir(parents=True)
    for day_offset, delay in ((2, 10), (1, 20)):
        day = (NOW - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        scheduled = NOW - timedelta(days=day_offset, hours=-10)  # 16:30 that day
        pl.DataFrame(
            [
                {
                    "stop_id": f"s{day_offset}",
                    "station_name": "Berlin Hauptbahnhof",
                    "train_type": "ICE",
                    "train_number": "1601",
                    "scheduled_time": scheduled,
                    "delay_probability": 0.5,
                    "delay_p50_min": 5.0,
                    "delay_p90_min": 20.0,
                    "coverage": "train",
                    "model_version": "test",
                    "predicted_at": scheduled - timedelta(hours=2),
                }
            ]
        ).write_parquet(live_dir / "predictions" / f"{day}.parquet")
        pl.DataFrame(
            [
                {
                    "stop_id": f"s{day_offset}",
                    "changed_time": scheduled + timedelta(minutes=delay),
                    "is_canceled": False,
                    "observed_at": scheduled + timedelta(hours=1),
                }
            ],
            schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
        ).write_parquet(live_dir / "changes" / f"{day}.parquet")


def test_refresh_builds_hand_checkable_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    seed_live_data(tmp_path)
    summary = refresh(now=NOW)
    assert summary["entities_written"] > 0

    store = OverlayStore(tmp_path / "snapshot_overlay")
    entry = store.get("train", ("ICE", "1601"))
    assert entry is not None
    # two observations, delays 10 and 20 -> mean 15, both >= 6 -> rate 1.0
    assert entry["train_mean_delay_30d"] == 15.0
    assert entry["train_delayed_rate_30d"] == 1.0
    # window ends yesterday -> available today
    assert entry["join_date"] == NOW.date()


def test_overlay_reloads_on_mtime_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    seed_live_data(tmp_path)
    refresh(now=NOW)
    store = OverlayStore(tmp_path / "snapshot_overlay")
    assert store.get("train", ("ICE", "1601")) is not None

    # Rewrite overlay with different data; store must pick it up
    path = tmp_path / "snapshot_overlay" / "train_stats.parquet"
    table = pl.read_parquet(path).with_columns(train_mean_delay_30d=pl.lit(99.0))
    import os
    import time

    table.write_parquet(path)
    os.utime(path, (time.time() + 5, time.time() + 5))  # force distinct mtime
    entry = store.get("train", ("ICE", "1601"))
    assert entry is not None and entry["train_mean_delay_30d"] == 99.0


def test_assemble_features_prefers_fresh_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config
    from dbahn_delay.serving.features import assemble_features
    from dbahn_delay.serving.loader import ModelBundle

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    seed_live_data(tmp_path)
    refresh(now=NOW)

    bundle = ModelBundle.load(Path(__file__).parent / "fixtures" / "bundle")
    store = OverlayStore(tmp_path / "snapshot_overlay")
    row, coverage = assemble_features(
        bundle,
        station_name="Berlin Hbf",  # fixture bundle knows this station
        train_type="ICE",
        train_number="1601",
        scheduled_time=NOW + timedelta(hours=3),
        train_line_station_num=None,
        overlay=store,
    )
    assert coverage == "train"
    # Overlay value (15.0) must beat the fixture bundle's canned value
    assert row["train_mean_delay_30d"] == 15.0
    # Granularity absent from overlay falls back to the bundle snapshot
    assert row["station_type_mean_delay_30d"] is not None


def test_refresh_excludes_todays_partial_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's half-reported observations must not enter the stats.

    First live run regression: partial today pushed join_date to tomorrow
    (negative freshness) and biased stats optimistically.
    """
    from dbahn_delay import config

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    seed_live_data(tmp_path)
    # add a TODAY observation with an absurd delay; it must be ignored
    day = NOW.strftime("%Y-%m-%d")
    scheduled = NOW - timedelta(hours=2)
    pl.DataFrame(
        [
            {
                "stop_id": "today",
                "station_name": "Berlin Hauptbahnhof",
                "train_type": "ICE",
                "train_number": "1601",
                "scheduled_time": scheduled,
                "delay_probability": 0.5,
                "delay_p50_min": 5.0,
                "delay_p90_min": 20.0,
                "coverage": "train",
                "model_version": "test",
                "predicted_at": scheduled - timedelta(hours=2),
            }
        ]
    ).write_parquet(tmp_path / "predictions" / f"{day}.parquet")

    refresh(now=NOW)
    store = OverlayStore(tmp_path / "snapshot_overlay")
    entry = store.get("train", ("ICE", "1601"))
    assert entry is not None
    assert entry["train_mean_delay_30d"] == 15.0  # unchanged: today excluded
    assert entry["join_date"] <= NOW.date()  # freshness can never be negative
