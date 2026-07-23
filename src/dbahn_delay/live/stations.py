"""Resolve panel station names to live-API EVA numbers.

The live API's primary EVA can differ from the historical dataset's EVA
(meta-station groups — e.g. Berlin Hbf is 8098160 live vs 8011160 in the
HF data), so we resolve by NAME via the /station search endpoint and commit
the resulting map. Regenerate only if the panel changes:

    uv run python -m dbahn_delay.live.stations
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.live.client import TimetablesClient

logger = logging.getLogger(__name__)

STATION_MAP_PATH = Path(__file__).parent / "station_map.json"

# Historical names the live API spells differently. Values are search
# patterns — the /station endpoint also accepts an EVA number, which is the
# only way to find names its search chokes on (e.g. "/" breaks the URL path).
NAME_OVERRIDES = {
    "Berlin Hauptbahnhof": "Berlin Hbf",
    "Köln Messe/Deutz": "8003368",
    "Ludwigshafen (Rhein) Hbf": "8000236",
}


def name_variants(name: str) -> list[str]:
    """Search candidates, most likely first.

    API naming quirks: parentheses carry no space on the left but sometimes
    keep it on the right ("Freiburg(Breisgau) Hbf"), sometimes not
    ("Singen(Hohentwiel)").
    """
    variants = []
    if name in NAME_OVERRIDES:
        variants.append(NAME_OVERRIDES[name])
    variants.append(name)
    left_squeezed = name.replace(" (", "(")
    both_squeezed = left_squeezed.replace(") ", ")")
    variants.extend(v for v in (both_squeezed, left_squeezed) if v != name)
    return variants


def load_station_map() -> dict[str, dict[str, str]]:
    """name -> {"eva": ..., "api_name": ...} for every resolvable panel station."""
    return json.loads(STATION_MAP_PATH.read_text())  # type: ignore[no-any-return]


def panel_station_names() -> list[str]:
    """The names of the stations in the canonical processed dataset."""
    return (
        pl.scan_parquet(settings.stops_path)
        .select(pl.col("station_name").unique().sort())
        .collect()["station_name"]
        .to_list()
    )


def resolve_station(client: TimetablesClient, name: str) -> dict[str, str] | None:
    for candidate in name_variants(name):
        xml_text = client.search_station(candidate)
        root = ET.fromstring(xml_text)
        first = root.find("station")
        if first is not None:
            return {"eva": first.attrib["eva"], "api_name": first.attrib.get("name", name)}
    return None


def build_station_map() -> None:
    client = TimetablesClient()
    mapping: dict[str, dict[str, str]] = {}
    failed = []
    names = panel_station_names()
    try:
        for name in names:
            try:
                resolved = resolve_station(client, name)
            except Exception:
                logger.exception("lookup failed for %r", name)
                resolved = None
            if resolved:
                mapping[name] = resolved
            else:
                failed.append(name)
    finally:
        client.close()
    STATION_MAP_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    logger.info(
        "resolved %d/%d stations -> %s (failed: %s)",
        len(mapping),
        len(names),
        STATION_MAP_PATH,
        ", ".join(failed) or "none",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_station_map()
