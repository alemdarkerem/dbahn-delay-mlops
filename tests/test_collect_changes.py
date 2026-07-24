"""Test the changes-only cycle with a stubbed client."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

BERLIN = ZoneInfo("Europe/Berlin")

FCHG_XML = """<?xml version='1.0' encoding='UTF-8'?>
<timetable station='X'>
  <s id="stop-1"><dp ct="2607241230" pt="2607241215"/></s>
</timetable>"""


class StubClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_changes(self, eva: str) -> str:
        self.calls += 1
        if eva == "boom":
            raise RuntimeError("api hiccup")
        return FCHG_XML

    def close(self) -> None:
        pass


def test_collect_changes_survives_station_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbahn_delay import config
    from dbahn_delay.live import collect_changes

    monkeypatch.setattr(config.settings, "live_dir", tmp_path)
    stub = StubClient()
    monkeypatch.setattr(collect_changes, "TimetablesClient", lambda: stub)
    monkeypatch.setattr(
        collect_changes,
        "load_station_map",
        lambda: {"Good": {"eva": "1"}, "Bad": {"eva": "boom"}, "Also Good": {"eva": "2"}},
    )

    now = datetime(2026, 7, 24, 12, 30, tzinfo=BERLIN)
    summary = collect_changes.run(now=now)
    assert summary == {"stations_ok": 2, "stations_failed": 1, "changes_recorded": 2}
    stored = pl.read_parquet(tmp_path / "changes" / "2026-07-24.parquet")
    assert stored.height == 1  # same stop id from both stations -> upserted once
