"""Naive baselines the ML models must beat.

Classification: historical delayed-rate per station x train_type.
Regression: historical p50/p90 delay per station x train_type.
Fallback chain for unseen combinations: train_type stats, then global stats.

Fitted strictly on a fold's train window — same time discipline as the models.
"""

from dataclasses import dataclass

import polars as pl

GROUP_KEYS = ["station_name", "train_type"]
FALLBACK_KEYS = ["train_type"]


@dataclass(frozen=True)
class BaselineTables:
    """Lookup tables produced from a train window."""

    group: pl.DataFrame  # per station x type
    fallback: pl.DataFrame  # per type
    global_rate: float
    global_p50: float
    global_p90: float


def fit_baseline(train: pl.LazyFrame) -> BaselineTables:
    aggs = [
        pl.col("target_delayed6").mean().alias("bl_rate"),
        pl.col("target_delay_min").quantile(0.5).alias("bl_p50"),
        pl.col("target_delay_min").quantile(0.9).alias("bl_p90"),
    ]
    group = train.group_by(GROUP_KEYS).agg(aggs).collect()
    fallback = train.group_by(FALLBACK_KEYS).agg(aggs).collect()
    glob = train.select(aggs).collect()
    return BaselineTables(
        group=group,
        fallback=fallback,
        global_rate=float(glob["bl_rate"][0]),
        global_p50=float(glob["bl_p50"][0]),
        global_p90=float(glob["bl_p90"][0]),
    )


def predict_baseline(tables: BaselineTables, data: pl.LazyFrame) -> pl.DataFrame:
    """Attach baseline predictions to validation rows (with fallback chain)."""
    fb = tables.fallback.rename(
        {"bl_rate": "fb_rate", "bl_p50": "fb_p50", "bl_p90": "fb_p90"}
    ).lazy()
    out = (
        data.join(tables.group.lazy(), on=GROUP_KEYS, how="left")
        .join(fb, on=FALLBACK_KEYS, how="left")
        .with_columns(
            baseline_prob=pl.coalesce(
                pl.col("bl_rate"), pl.col("fb_rate"), pl.lit(tables.global_rate)
            ),
            baseline_p50=pl.coalesce(pl.col("bl_p50"), pl.col("fb_p50"), pl.lit(tables.global_p50)),
            baseline_p90=pl.coalesce(pl.col("bl_p90"), pl.col("fb_p90"), pl.lit(tables.global_p90)),
        )
        .drop("bl_rate", "bl_p50", "bl_p90", "fb_rate", "fb_p50", "fb_p90")
    )
    return out.collect()
