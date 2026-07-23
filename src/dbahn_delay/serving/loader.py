"""Load a versioned model bundle: boosters, metadata and the feature snapshot."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from dbahn_delay.features.build import GRANULARITIES
from dbahn_delay.features.snapshot import SNAPSHOT_FILES

# Snapshot columns looked up per granularity (order matters for the model).
STAT_COLUMNS = {
    prefix: [
        f"{prefix}_mean_delay_14d",
        f"{prefix}_delayed_rate_14d",
        f"{prefix}_mean_delay_30d",
        f"{prefix}_delayed_rate_30d",
        f"{prefix}_count_30d",
    ]
    for prefix in GRANULARITIES
}


@dataclass(frozen=True)
class ModelBundle:
    """Everything the API needs, loaded once at startup."""

    version: str
    metadata: dict[str, Any]
    clf: lgb.Booster
    q50: lgb.Booster
    q90: lgb.Booster
    # granularity -> {entity key tuple -> {"join_date": date, stats...}}
    stats: dict[str, dict[tuple[str, ...], dict[str, Any]]]

    @classmethod
    def load(cls, bundle_dir: Path) -> "ModelBundle":
        metadata = json.loads((bundle_dir / "metadata.json").read_text())
        stats: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
        for prefix, keys in GRANULARITIES.items():
            table = pl.read_parquet(bundle_dir / SNAPSHOT_FILES[prefix])
            lookup: dict[tuple[str, ...], dict[str, Any]] = {}
            for row in table.iter_rows(named=True):
                entity = tuple(str(row[k]) for k in keys)
                lookup[entity] = row
            stats[prefix] = lookup
        return cls(
            version=metadata["version"],
            metadata=metadata,
            clf=lgb.Booster(model_file=str(bundle_dir / "clf.txt")),
            q50=lgb.Booster(model_file=str(bundle_dir / "q50.txt")),
            q90=lgb.Booster(model_file=str(bundle_dir / "q90.txt")),
            stats=stats,
        )

    def stats_age_days(self, today: date) -> int:
        """Age of the freshest snapshot entry — staleness signal for /health."""
        newest = max(
            (row["join_date"] for lookup in self.stats.values() for row in lookup.values()),
            default=today,
        )
        return (today - newest).days
