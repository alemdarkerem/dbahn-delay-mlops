"""Feature snapshot: the serving-time mini feature store.

Exports, per granularity, each entity's LATEST trailing-window statistics
into small parquet files inside a model bundle. At request time the API
looks these up instead of scanning the 44M-row feature frame — same values,
same asof semantics ("most recent stats strictly before today"), no skew.

Usage: ``python -m dbahn_delay.features.snapshot <bundle_dir>``
(train.py also calls this when exporting a fresh bundle).
"""

import logging
import sys
from pathlib import Path

import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.features.build import (
    GRANULARITIES,
    add_calendar_features,
    add_scheduled_time,
    daily_aggregates,
    rolling_from_daily,
)

logger = logging.getLogger(__name__)

SNAPSHOT_FILES = {prefix: f"{prefix}_stats.parquet" for prefix in GRANULARITIES}


def latest_stats(lf: pl.LazyFrame, keys: list[str], prefix: str) -> pl.DataFrame:
    """One row per entity: its most recent trailing-window statistics."""
    stats = rolling_from_daily(daily_aggregates(lf, keys), keys, prefix)
    return stats.sort("join_date").group_by(keys, maintain_order=True).last().collect()


def export_snapshot(bundle_dir: Path) -> None:
    """Write per-granularity latest-stats tables into a bundle directory."""
    stops = (
        pl.scan_parquet(settings.stops_path)
        .filter(pl.col("is_canceled").not_())
        .pipe(add_scheduled_time)
        .pipe(add_calendar_features)
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for prefix, keys in GRANULARITIES.items():
        table = latest_stats(stops, keys, prefix)
        path = bundle_dir / SNAPSHOT_FILES[prefix]
        table.write_parquet(path)
        logger.info("wrote %s: %d entities", path, table.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_snapshot(Path(sys.argv[1]))
