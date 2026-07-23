"""Ingest pipeline: raw monthly files -> one canonical, clean stops dataset.

All transforms are pure LazyFrame -> LazyFrame functions so they can be unit
tested on synthetic data and reused verbatim by the live-serving path later
(one shared pipeline, no train/serve skew).

Usage: ``python -m dbahn_delay.data.ingest`` (or ``make ingest``).
"""

import logging

import polars as pl

from dbahn_delay.config import settings

logger = logging.getLogger(__name__)

TIMESTAMP_COLUMNS = (
    "time",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
)

# Delays outside these bounds are data errors, not operations (see EDA).
DELAY_MIN = -60
DELAY_MAX = 1_440


def to_berlin_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Attach the Europe/Berlin timezone to the naive local timestamps.

    The raw data stores local wall-clock times. During the fall-back DST
    transition the same wall-clock hour occurs twice; those ambiguous stamps
    are resolved to the earliest occurrence. Spring-forward gaps cannot occur
    in valid data, but any nonexistent stamp is nulled rather than crashing.
    """
    return lf.with_columns(
        pl.col(c)
        .dt.cast_time_unit("us")
        .dt.replace_time_zone("Europe/Berlin", ambiguous="earliest", non_existent="null")
        for c in TIMESTAMP_COLUMNS
    )


def drop_bad_rows(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove rows that cannot be used for training or evaluation."""
    return lf.filter(
        pl.col("station_name").is_not_null()
        & pl.col("train_type").is_not_null()
        & pl.col("time").is_not_null()
        & pl.col("delay_in_min").is_between(DELAY_MIN, DELAY_MAX)
    )


def dedupe(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep one row per stop id (files could overlap at month boundaries)."""
    return lf.unique(subset="id", keep="first")


def station_panel(lf: pl.LazyFrame, min_month_share: float = 1.0) -> list[str]:
    """Stations present in at least ``min_month_share`` of all months.

    The dataset covers only the top ~100 stations before 2025-11 and all
    stations afterwards. Restricting to a consistent panel avoids a fake
    "new stations appeared" drift signal in the training data.
    """
    months_per_station = (
        lf.select(
            station=pl.col("station_name"),
            month=pl.col("time").dt.strftime("%Y-%m"),
        )
        .unique()
        .group_by("station")
        .len("months")
    )
    n_months = lf.select(pl.col("time").dt.strftime("%Y-%m").n_unique()).collect().item()
    kept = (
        months_per_station.filter(pl.col("months") >= min_month_share * n_months)
        .select("station")
        .collect()["station"]
        .sort()
        .to_list()
    )
    return kept


def build_stops() -> None:
    """Run the full ingest: scan raw files, clean, restrict panel, write parquet."""
    raw = pl.scan_parquet(settings.monthly_raw_dir / "*.parquet")
    panel = station_panel(raw)
    logger.info("Station panel: %d stations", len(panel))

    stops = (
        raw.pipe(to_berlin_time)
        .pipe(drop_bad_rows)
        .pipe(dedupe)
        .filter(pl.col("station_name").is_in(panel))
    )

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    stops.sink_parquet(settings.stops_path)

    n = pl.scan_parquet(settings.stops_path).select(pl.len()).collect().item()
    logger.info("Wrote %s: %d rows", settings.stops_path, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_stops()
