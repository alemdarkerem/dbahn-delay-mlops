"""Overlay store: prefer fresh live-derived stats over the bundle snapshot.

The overlay files are rewritten daily by the refresh job; the store reloads
lazily when a file's mtime changes (one os.stat per request — microseconds).
Missing overlay files are fine: lookups simply fall back to the bundle.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl

from dbahn_delay.features.build import GRANULARITIES
from dbahn_delay.features.snapshot import SNAPSHOT_FILES

logger = logging.getLogger(__name__)


class OverlayStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._mtimes: dict[str, float] = {}
        self._tables: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}

    def _reload_if_changed(self, prefix: str) -> None:
        path = self._dir / SNAPSHOT_FILES[prefix]
        if not path.exists():
            if prefix in self._tables:
                del self._tables[prefix]
                del self._mtimes[prefix]
            return
        mtime = path.stat().st_mtime
        if self._mtimes.get(prefix) == mtime:
            return
        keys = GRANULARITIES[prefix]
        lookup: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in pl.read_parquet(path).iter_rows(named=True):
            lookup[tuple(str(row[k]) for k in keys)] = row
        self._tables[prefix] = lookup
        self._mtimes[prefix] = mtime
        logger.info("overlay %s reloaded: %d entities", prefix, len(lookup))

    def get(self, prefix: str, entity: tuple[str, ...]) -> dict[str, Any] | None:
        self._reload_if_changed(prefix)
        return self._tables.get(prefix, {}).get(entity)

    def newest_join_date(self) -> Any | None:
        """Freshness signal for /health (max join_date across granularities)."""
        dates: list[Any] = []
        for prefix in GRANULARITIES:
            self._reload_if_changed(prefix)
            dates.extend(row["join_date"] for row in self._tables.get(prefix, {}).values())
        return max(dates) if dates else None
