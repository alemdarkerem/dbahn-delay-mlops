"""Daily snapshot overlay: fresh rolling stats from our own live observations.

The bundle's feature snapshot ages with its training data (day-1 report:
stats 23 days old, ECE drifting). This job rebuilds the SAME statistics
(trailing 14/30-day mean delay / delayed rate per granularity) from the live
loop's sealed predictions joined with observed changes, and writes them as an
overlay the API prefers over the bundle snapshot. Model untouched; features
current. Runs daily on the VPS after the morning evaluation.

CLI: ``python -m dbahn_delay.live.refresh_snapshot`` (cron ``30 4 * * *`` UTC).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.features.build import GRANULARITIES, daily_aggregates, rolling_from_daily
from dbahn_delay.features.snapshot import SNAPSHOT_FILES

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
LOOKBACK_DAYS = 35  # covers the longest rolling window (30d) with margin


def overlay_dir() -> Path:
    return settings.live_dir / "snapshot_overlay"


def observed_stops(now: datetime) -> pl.DataFrame | None:
    """Live observations as a stops-like frame (same rules as evaluate_day).

    delay = changed - scheduled; never-seen-in-changes => on time (0);
    canceled excluded. Only the columns the rolling-stat builders need.
    """
    days = [(now.date() - timedelta(days=i)).isoformat() for i in range(LOOKBACK_DAYS + 1)]
    pred_frames = [
        pl.read_parquet(p)
        for d in days
        if (p := settings.live_dir / "predictions" / f"{d}.parquet").exists()
    ]
    if not pred_frames:
        return None
    predictions = pl.concat(pred_frames)
    change_frames = [
        pl.read_parquet(p)
        for d in days
        if (p := settings.live_dir / "changes" / f"{d}.parquet").exists()
    ]
    if change_frames:
        changes = pl.concat(change_frames).sort("observed_at").unique(subset="stop_id", keep="last")
        joined = predictions.join(
            changes.select("stop_id", "changed_time", "is_canceled"), on="stop_id", how="left"
        )
    else:
        joined = predictions.with_columns(
            changed_time=pl.lit(None, dtype=pl.Datetime("us", "Europe/Berlin")),
            is_canceled=pl.lit(None, dtype=pl.Boolean),
        )
    return (
        joined.with_columns(
            is_canceled=pl.col("is_canceled").fill_null(False),
            delay_in_min=(pl.col("changed_time") - pl.col("scheduled_time"))
            .dt.total_minutes()
            .fill_null(0)
            .cast(pl.Int32),
            event_date=pl.col("scheduled_time").dt.date(),
        )
        # only COMPLETED days: today's partial observations would bias the
        # stats optimistically (delays not yet reported) and push join_date
        # into tomorrow (negative freshness - caught on first live run)
        .filter(pl.col("event_date") < now.date())
        .select(
            "station_name",
            "train_type",
            "train_number",
            "delay_in_min",
            "is_canceled",
            "event_date",
        )
    )


def refresh(now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(tz=BERLIN)
    stops = observed_stops(now)
    if stops is None or stops.is_empty():
        logger.warning("no live observations yet - overlay not written")
        return {"entities_written": 0}

    out_dir = overlay_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for prefix, keys in GRANULARITIES.items():
        stats = rolling_from_daily(daily_aggregates(stops.lazy(), keys), keys, prefix)
        latest = stats.sort("join_date").group_by(keys, maintain_order=True).last().collect()
        latest.write_parquet(out_dir / SNAPSHOT_FILES[prefix])
        total += latest.height
        logger.info("overlay %s: %d entities", prefix, latest.height)
    return {"entities_written": total}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    refresh()
