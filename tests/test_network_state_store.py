"""Tests for the serving-side NetworkStateStore (hand-checked aggregates)."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from dbahn_delay.serving.network_state import NetworkStateStore

BERLIN = ZoneInfo("Europe/Berlin")


def write_live(
    live_dir: Path, preds: list[dict[str, object]], changes: list[dict[str, object]]
) -> None:
    now = datetime.now(tz=BERLIN)
    day = now.strftime("%Y-%m-%d")
    (live_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (live_dir / "changes").mkdir(parents=True, exist_ok=True)
    base = {
        "station_name": "Berlin Hbf",
        "train_type": "ICE",
        "train_number": "1",
        "delay_probability": 0.5,
        "delay_p50_min": 1.0,
        "delay_p90_min": 9.0,
        "coverage": "train",
        "model_version": "test",
        "predicted_at": now - timedelta(hours=3),
    }
    pl.DataFrame([base | p for p in preds]).write_parquet(
        live_dir / "predictions" / f"{day}.parquet"
    )
    if changes:
        pl.DataFrame(
            [{"observed_at": now, "is_canceled": False} | c for c in changes],
            schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
        ).write_parquet(live_dir / "changes" / f"{day}.parquet")


def test_store_aggregates_hand_checked(tmp_path: Path) -> None:
    now = datetime.now(tz=BERLIN)
    write_live(
        tmp_path,
        preds=[
            # departed 30 min ago, no change record -> on time (0)
            {"stop_id": "a", "scheduled_time": now - timedelta(minutes=30)},
            # departed with a 30-min delay observed 20 min ago
            {"stop_id": "b", "scheduled_time": now - timedelta(minutes=50)},
            # departed 90 min ago -> outside the 60-min window
            {"stop_id": "old", "scheduled_time": now - timedelta(minutes=90)},
            # canceled -> excluded
            {"stop_id": "cx", "scheduled_time": now - timedelta(minutes=10)},
            # not departed yet (DB reports a future delay) -> excluded
            {"stop_id": "future", "scheduled_time": now + timedelta(minutes=30)},
            # another station and type, on time 10 min ago
            {
                "stop_id": "re",
                "scheduled_time": now - timedelta(minutes=10),
                "station_name": "Köln Hbf",
                "train_type": "RE",
            },
        ],
        changes=[
            {"stop_id": "b", "changed_time": now - timedelta(minutes=20)},
            {"stop_id": "cx", "changed_time": None, "is_canceled": True},
            {"stop_id": "future", "changed_time": now + timedelta(minutes=50)},
        ],
    )
    store = NetworkStateStore(tmp_path)
    row = store.features("Berlin Hbf", "ICE")
    assert row["station_live_mean_delay_60m"] == 15.0  # (0 + 30) / 2
    assert row["station_live_delayed_share_60m"] == 0.5
    assert row["station_live_n_60m"] == 2
    assert row["type_live_mean_delay_60m"] == 15.0  # both ICEs
    # network: 0, 30, 0 -> mean 10, one of three delayed
    assert row["network_live_mean_delay_60m"] == 10.0
    assert abs(row["network_live_delayed_share_60m"] - 1 / 3) < 1e-9

    other = store.features("Köln Hbf", "RE")
    assert other["station_live_mean_delay_60m"] == 0.0
    assert other["station_live_n_60m"] == 1
    assert other["type_live_delayed_share_60m"] == 0.0


def test_store_empty_dir_returns_missing(tmp_path: Path) -> None:
    row = NetworkStateStore(tmp_path).features("Berlin Hbf", "ICE")
    assert row["station_live_mean_delay_60m"] is None
    assert row["station_live_n_60m"] == 0
    assert row["network_live_mean_delay_60m"] is None


def test_store_reloads_when_files_change(tmp_path: Path) -> None:
    now = datetime.now(tz=BERLIN)
    write_live(
        tmp_path,
        preds=[{"stop_id": "a", "scheduled_time": now - timedelta(minutes=30)}],
        changes=[],
    )
    store = NetworkStateStore(tmp_path)
    assert store.features("Berlin Hbf", "ICE")["station_live_n_60m"] == 1

    write_live(
        tmp_path,
        preds=[
            {"stop_id": "a", "scheduled_time": now - timedelta(minutes=30)},
            {"stop_id": "b", "scheduled_time": now - timedelta(minutes=15)},
        ],
        changes=[],
    )
    # force a visible mtime bump even on coarse filesystems
    day = now.strftime("%Y-%m-%d")
    pred_path = tmp_path / "predictions" / f"{day}.parquet"
    stat = pred_path.stat()
    os.utime(pred_path, (stat.st_atime + 5, stat.st_mtime + 5))
    assert store.features("Berlin Hbf", "ICE")["station_live_n_60m"] == 2
