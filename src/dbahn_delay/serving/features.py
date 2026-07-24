"""Request -> model feature row, reusing the training feature code path.

Calendar features come from the exact same ``add_calendar_features`` used at
training time; rolling-history features come from the bundle's snapshot with
the training asof semantics: stats are used only if their window ended within
tolerance, otherwise that granularity falls back (train -> station_type ->
type -> cold). The strongest granularity that answered defines the response's
``coverage`` field.
"""

from datetime import datetime
from typing import Any

import polars as pl

from dbahn_delay.features.build import GRANULARITIES, ROLLING_WINDOWS, add_calendar_features
from dbahn_delay.serving.loader import STAT_COLUMNS, ModelBundle
from dbahn_delay.serving.overlay import OverlayStore

# A snapshot entry older than the longest window carries no usable signal.
MAX_STALENESS_DAYS = max(ROLLING_WINDOWS)

COVERAGE_ORDER = [*GRANULARITIES, "cold"]


def calendar_row(scheduled_time: datetime) -> dict[str, Any]:
    """Calendar features for one scheduled time via the training pipeline."""
    frame = pl.LazyFrame({"scheduled_time": [scheduled_time]}).pipe(add_calendar_features)
    return frame.collect().row(0, named=True)


def assemble_features(
    bundle: ModelBundle,
    *,
    station_name: str,
    train_type: str,
    train_number: str,
    scheduled_time: datetime,
    train_line_station_num: int | None,
    overlay: OverlayStore | None = None,
) -> tuple[dict[str, Any], str]:
    """Build the model's feature dict and report the achieved coverage level.

    Per granularity the OVERLAY (daily live-derived stats) is consulted first,
    the bundle snapshot second — freshest usable window wins. Both go through
    the same staleness tolerance.
    """
    cal = calendar_row(scheduled_time)
    event_date = cal["event_date"]

    row: dict[str, Any] = {
        "station_name": station_name,
        "train_type": train_type,
        "train_number": train_number,
        "train_line_station_num": train_line_station_num,
        "scheduled_hour": cal["scheduled_hour"],
        "weekday": cal["weekday"],
        "month": cal["month"],
        "is_weekend": cal["is_weekend"],
        "is_holiday": cal["is_holiday"],
    }

    entity_keys = {
        "train": (train_type, train_number),
        "station_type": (station_name, train_type),
        "type": (train_type,),
    }

    def usable(entry: dict[str, Any] | None) -> bool:
        return entry is not None and (event_date - entry["join_date"]).days <= MAX_STALENESS_DAYS

    coverage = "cold"
    for prefix in GRANULARITIES:
        entry = overlay.get(prefix, entity_keys[prefix]) if overlay else None
        if not usable(entry):
            entry = bundle.stats[prefix].get(entity_keys[prefix])
        fresh = usable(entry)
        for col in STAT_COLUMNS[prefix]:
            row[col] = entry[col] if fresh and entry is not None else None
        if fresh and coverage == "cold":
            coverage = prefix  # first (most specific) granularity that answered
    return row, coverage
