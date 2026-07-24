"""Live network-state at serving time: the trailing hour, right now.

Training computes these features as of the simulated seal time; in
production the seal moment IS "now" (the :05 fetch cycle calls /predict
while sealing), so the store aggregates observations from the last 60
minutes of today's (+ yesterday's, for the window crossing midnight) live
files. Same ground-truth rules as evaluate_day: no change record => on
time, canceled excluded, latest change per stop wins. An observation
counts from its realized departure time (changed_time if delayed, else
scheduled_time) — mirroring training's "a delay is known only once it
happened".

Missing live files (local dev, tests, cold start) => every lookup returns
None => the model sees NaN, exactly like a quiet station at 4 am.

Recomputation is cheap (two day-sized parquet files) but not free, so the
aggregate is cached and refreshed when a source file's mtime changes or a
5-minute wall-clock bucket passes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from dbahn_delay.features.build import DELAYED_THRESHOLD_MIN
from dbahn_delay.features.network_state import WINDOW_MINUTES

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")


@dataclass(frozen=True)
class _State:
    station: dict[str, tuple[float, float, int]]  # name -> (mean, share, n)
    train_type: dict[str, tuple[float, float]]
    network: tuple[float, float] | None


_EMPTY = _State(station={}, train_type={}, network=None)


def _observations(live_dir: Path, now: datetime) -> pl.DataFrame | None:
    """Observed departures in [now - 60m, now) from the live files."""
    days = [(now - timedelta(days=1)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")]
    preds = [
        pl.read_parquet(p)
        for d in days
        if (p := live_dir / "predictions" / f"{d}.parquet").exists()
    ]
    if not preds:
        return None
    predictions = pl.concat(preds, how="diagonal_relaxed")
    changes = [
        pl.read_parquet(p) for d in days if (p := live_dir / "changes" / f"{d}.parquet").exists()
    ]
    if changes:
        latest = pl.concat(changes).sort("observed_at").unique(subset="stop_id", keep="last")
        joined = predictions.join(
            latest.select("stop_id", "changed_time", "is_canceled"), on="stop_id", how="left"
        )
    else:
        joined = predictions.with_columns(
            changed_time=pl.lit(None, dtype=pl.Datetime("us", "Europe/Berlin")),
            is_canceled=pl.lit(None, dtype=pl.Boolean),
        )
    return (
        joined.with_columns(
            is_canceled=pl.col("is_canceled").fill_null(False),
            obs_time=pl.coalesce(pl.col("changed_time"), pl.col("scheduled_time")),
            delay_in_min=(pl.col("changed_time") - pl.col("scheduled_time"))
            .dt.total_minutes()
            .fill_null(0),
        )
        .filter(
            ~pl.col("is_canceled"),
            pl.col("obs_time") >= now - timedelta(minutes=WINDOW_MINUTES),
            pl.col("obs_time") < now,
        )
        .select("station_name", "train_type", "delay_in_min")
    )


def _aggregate(obs: pl.DataFrame) -> _State:
    def stats(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
        return df.group_by(keys).agg(
            mean=pl.col("delay_in_min").mean(),
            share=(pl.col("delay_in_min") >= DELAYED_THRESHOLD_MIN).mean(),
            n=pl.len(),
        )

    station = {
        r["station_name"]: (float(r["mean"]), float(r["share"]), int(r["n"]))
        for r in stats(obs, ["station_name"]).iter_rows(named=True)
    }
    train_type = {
        r["train_type"]: (float(r["mean"]), float(r["share"]))
        for r in stats(obs, ["train_type"]).iter_rows(named=True)
    }
    network = None
    if obs.height:
        network = (
            float(obs["delay_in_min"].mean()),  # type: ignore[arg-type]
            float((obs["delay_in_min"] >= DELAYED_THRESHOLD_MIN).mean()),  # type: ignore[arg-type]
        )
    return _State(station=station, train_type=train_type, network=network)


class NetworkStateStore:
    """Cached 60-minute live aggregates for /predict."""

    def __init__(self, live_dir: Path) -> None:
        self._live_dir = live_dir
        self._state = _EMPTY
        self._cache_key: tuple[Any, ...] | None = None

    def _refresh_if_stale(self) -> None:
        now = datetime.now(tz=BERLIN)
        days = [(now - timedelta(days=1)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")]
        mtimes = tuple(
            p.stat().st_mtime if (p := self._live_dir / sub / f"{d}.parquet").exists() else None
            for sub in ("predictions", "changes")
            for d in days
        )
        bucket = now.timestamp() // 300  # advance at least every 5 minutes
        key = (*mtimes, bucket)
        if key == self._cache_key:
            return
        try:
            obs = _observations(self._live_dir, now)
            self._state = _aggregate(obs) if obs is not None else _EMPTY
        except Exception:
            logger.exception("network-state refresh failed; keeping previous state")
        self._cache_key = key

    def features(self, station_name: str, train_type: str) -> dict[str, Any]:
        """The 7 network-state feature values for one request (None = NaN)."""
        self._refresh_if_stale()
        st = self._state.station.get(station_name)
        ty = self._state.train_type.get(train_type)
        net = self._state.network
        return {
            "station_live_mean_delay_60m": st[0] if st else None,
            "station_live_delayed_share_60m": st[1] if st else None,
            "station_live_n_60m": st[2] if st else 0,
            "type_live_mean_delay_60m": ty[0] if ty else None,
            "type_live_delayed_share_60m": ty[1] if ty else None,
            "network_live_mean_delay_60m": net[0] if net else None,
            "network_live_delayed_share_60m": net[1] if net else None,
        }
