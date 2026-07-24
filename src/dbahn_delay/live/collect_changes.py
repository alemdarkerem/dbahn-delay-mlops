"""Lightweight changes-only cycle: refresh observed delays/cancellations.

Runs between the hourly fetch cycles (e.g. every 15 minutes) so the board's
"actual delay" column stays fresh without re-predicting anything. One fchg
request per station (~105 requests, ~2 minutes paced) — no model, no plan
calls, tiny memory footprint.

Cron: ``python -m dbahn_delay.live.collect_changes`` at ``*/15 * * * *``
(the :00/:15/:30/:45 slots never overlap the :05-:11 fetch window, keeping
the combined request rate under the API's 60/min free-tier limit).
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from dbahn_delay.live.client import TimetablesClient
from dbahn_delay.live.fetch import upsert_changes
from dbahn_delay.live.parse import Change, parse_changes
from dbahn_delay.live.stations import load_station_map

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")


def run(now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(tz=BERLIN)
    stations = load_station_map()
    client = TimetablesClient()
    changes: list[Change] = []
    ok = failed = 0
    try:
        for name, info in stations.items():
            try:
                changes.extend(parse_changes(client.fetch_changes(info["eva"])))
                ok += 1
            except Exception:
                logger.exception("station %r failed, continuing", name)
                failed += 1
    finally:
        client.close()

    n = upsert_changes(changes, observed_at=now, day=now.strftime("%Y-%m-%d"))
    summary = {"stations_ok": ok, "stations_failed": failed, "changes_recorded": n}
    logger.info("changes cycle done: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
