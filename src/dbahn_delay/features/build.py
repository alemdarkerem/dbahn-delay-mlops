"""Leak-safe feature pipeline: canonical stops -> model-ready feature frame.

The hard rule: a prediction happens BEFORE the stop, so every feature must be
computable from schedule information plus strictly-past history. All rolling
statistics are aggregated per calendar day and joined "as of the previous
day" — a row scheduled on day D only ever sees aggregates of days < D.

Usage: ``python -m dbahn_delay.features.build`` (or ``make features``).
"""

import datetime as dt
import logging

import holidays
import polars as pl

from dbahn_delay.config import settings

logger = logging.getLogger(__name__)

# Trailing windows (in days) for the rolling history features.
ROLLING_WINDOWS = (14, 30)

# Granularities at which history is aggregated. Order matters: from most
# specific (best signal, worst coverage) to least specific (cold-start fallback).
GRANULARITIES: dict[str, list[str]] = {
    "train": ["train_type", "train_number"],  # "is ICE 1601 chronically late?"
    "station_type": ["station_name", "train_type"],
    "type": ["train_type"],
}

DELAYED_THRESHOLD_MIN = 6


def add_scheduled_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Reconstruct the scheduled (planned) event time.

    Preference order: planned departure, planned arrival, actual time minus
    delay. The scheduled time is what a passenger knows before the trip, so
    all calendar features derive from it.
    """
    return lf.with_columns(
        scheduled_time=pl.coalesce(
            pl.col("departure_planned_time"),
            pl.col("arrival_planned_time"),
            pl.col("time").dt.offset_by(pl.format("-{}m", pl.col("delay_in_min"))),
        )
    )


def add_calendar_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Schedule-derived features: hour, weekday, month, weekend, holiday."""
    years = range(2024, dt.date.today().year + 2)
    de_holidays = list(holidays.country_holidays("DE", years=years))  # national only
    return lf.with_columns(
        scheduled_hour=pl.col("scheduled_time").dt.hour(),
        weekday=pl.col("scheduled_time").dt.weekday(),
        month=pl.col("scheduled_time").dt.month(),
        is_weekend=pl.col("scheduled_time").dt.weekday() >= 6,
        is_holiday=pl.col("scheduled_time").dt.date().is_in(de_holidays),
        event_date=pl.col("scheduled_time").dt.date(),
    )


def daily_aggregates(lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """Per-day delay statistics for one granularity (train / station x type / type)."""
    return (
        lf.filter(pl.col("is_canceled").not_())
        .group_by([*keys, "event_date"])
        .agg(
            delay_sum=pl.col("delay_in_min").sum(),
            delayed_sum=(pl.col("delay_in_min") >= DELAYED_THRESHOLD_MIN).sum(),
            n=pl.len(),
        )
    )


def rolling_from_daily(daily: pl.LazyFrame, keys: list[str], prefix: str) -> pl.LazyFrame:
    """Turn daily aggregates into trailing-window stats keyed by availability date.

    The stats of the window ending on day D become *available* on D+1
    (``join_date``). Joining stops on ``event_date >= join_date`` (asof,
    backward) therefore can never see same-day data. That one-day shift is
    the leak barrier; the unit tests pin it down.
    """
    daily = daily.sort([*keys, "event_date"]).with_columns(
        # upsample-free trick: rolling over *rows* would be wrong with gaps,
        # so roll over a date-indexed window instead.
        pl.col("event_date").alias("_d")
    )
    out = daily
    for w in ROLLING_WINDOWS:
        out = out.with_columns(
            pl.col("delay_sum")
            .rolling_sum_by("_d", window_size=f"{w}d", closed="right")
            .over(keys)
            .alias(f"_ds{w}"),
            pl.col("delayed_sum")
            .rolling_sum_by("_d", window_size=f"{w}d", closed="right")
            .over(keys)
            .alias(f"_dd{w}"),
            pl.col("n")
            .rolling_sum_by("_d", window_size=f"{w}d", closed="right")
            .over(keys)
            .alias(f"_n{w}"),
        )
    # Shift by one day: join key becomes "the day AFTER the window ends".
    out = out.with_columns(join_date=pl.col("event_date").dt.offset_by("1d"))
    exprs = []
    for w in ROLLING_WINDOWS:
        exprs.append((pl.col(f"_ds{w}") / pl.col(f"_n{w}")).alias(f"{prefix}_mean_delay_{w}d"))
        exprs.append((pl.col(f"_dd{w}") / pl.col(f"_n{w}")).alias(f"{prefix}_delayed_rate_{w}d"))
    exprs.append(pl.col(f"_n{ROLLING_WINDOWS[-1]}").alias(f"{prefix}_count_{ROLLING_WINDOWS[-1]}d"))
    return out.select([*keys, "join_date", *exprs])


def add_rolling_stats(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Attach trailing history features at every granularity via asof joins.

    Semantics: a stop on day D receives the trailing-window stats as of the
    entity's most recent active day strictly before D, but only if that day is
    at most ``w`` days old (asof tolerance) — otherwise null. This matches how
    a live feature store would serve "latest known stats" at prediction time,
    and it handles entities that do not run every day (weekend-only trains).
    """
    out = lf
    for prefix, keys in GRANULARITIES.items():
        stats = rolling_from_daily(daily_aggregates(lf, keys), keys, prefix)
        for w in ROLLING_WINDOWS:
            cols = [f"{prefix}_mean_delay_{w}d", f"{prefix}_delayed_rate_{w}d"]
            if w == ROLLING_WINDOWS[-1]:
                cols.append(f"{prefix}_count_{w}d")
            sub = stats.select([*keys, "join_date", *cols]).sort("join_date")
            out = (
                out.sort("event_date")
                .join_asof(
                    sub,
                    left_on="event_date",
                    right_on="join_date",
                    by=keys,
                    strategy="backward",
                    tolerance=f"{w}d",
                    # both sides are sorted on the asof key right above;
                    # polars cannot verify it per-group and would warn
                    check_sortedness=False,
                )
                .drop("join_date")
            )
    return out


def build_feature_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    """stops -> model-ready frame with features, targets and fold key."""
    non_canceled = lf.filter(pl.col("is_canceled").not_())
    return (
        non_canceled.pipe(add_scheduled_time)
        .pipe(add_calendar_features)
        .pipe(add_rolling_stats)
        .with_columns(
            target_delayed6=(pl.col("delay_in_min") >= DELAYED_THRESHOLD_MIN),
            target_delay_min=pl.col("delay_in_min"),
            fold_month=pl.col("scheduled_time").dt.strftime("%Y-%m"),
        )
        .select(
            "id",
            "station_name",
            "train_type",
            "train_number",
            "train_line_station_num",
            "scheduled_time",
            "scheduled_hour",
            "weekday",
            "month",
            "is_weekend",
            "is_holiday",
            *[
                f"{prefix}_{stat}_{w}d"
                for prefix in GRANULARITIES
                for w in ROLLING_WINDOWS
                for stat in ("mean_delay", "delayed_rate")
            ],
            *[f"{prefix}_count_{ROLLING_WINDOWS[-1]}d" for prefix in GRANULARITIES],
            "target_delayed6",
            "target_delay_min",
            "fold_month",
        )
    )


def main() -> None:
    stops = pl.scan_parquet(settings.stops_path)
    features = build_feature_frame(stops)
    out_path = settings.processed_dir / "features.parquet"
    features.sink_parquet(out_path)
    n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    logger.info("Wrote %s: %d rows", out_path, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
