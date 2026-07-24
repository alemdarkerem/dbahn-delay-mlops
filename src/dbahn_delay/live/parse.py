"""Parse Timetables API XML into typed records.

Pure functions over XML strings — fixture-tested against real captured
responses. Timestamps in the API are YYMMDDHHMM local Berlin wall-clock;
we convert them to tz-aware datetimes immediately (same policy as ingest).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")


@dataclass(frozen=True)
class PlannedStop:
    stop_id: str
    station_name: str
    train_type: str
    train_number: str
    scheduled_time: datetime  # departure preferred, else arrival (schedule semantics)
    has_departure: bool
    line: str | None = None  # passenger-facing line label ("7", "RE5", "FEX")


@dataclass(frozen=True)
class Change:
    stop_id: str
    changed_time: datetime | None  # departure ct preferred, else arrival ct
    is_canceled: bool


def parse_ts(value: str) -> datetime:
    """YYMMDDHHMM (Berlin wall-clock) -> tz-aware datetime."""
    return datetime.strptime(value, "%y%m%d%H%M").replace(tzinfo=BERLIN)


def parse_plan(xml_text: str) -> list[PlannedStop]:
    root = ET.fromstring(xml_text)
    station = root.attrib.get("station", "")
    stops = []
    for s in root.findall("s"):
        tl = s.find("tl")
        dp = s.find("dp")
        ar = s.find("ar")
        if tl is None or (dp is None and ar is None):
            continue
        event = dp if dp is not None else ar
        assert event is not None
        pt = event.attrib.get("pt")
        if not pt:
            continue
        stops.append(
            PlannedStop(
                stop_id=s.attrib["id"],
                station_name=station,
                train_type=tl.attrib.get("c", ""),
                train_number=tl.attrib.get("n", ""),
                scheduled_time=parse_ts(pt),
                has_departure=dp is not None,
                line=event.attrib.get("l"),
            )
        )
    return stops


def parse_changes(xml_text: str) -> list[Change]:
    """Latest known change per stop: changed time (ct) and cancellation.

    A stop is considered canceled when its event carries cs="c" (cancellation
    status) — either arrival or departure.
    """
    root = ET.fromstring(xml_text)
    changes = []
    for s in root.findall("s"):
        dp = s.find("dp")
        ar = s.find("ar")
        if dp is None and ar is None:
            continue
        canceled = any(e is not None and e.attrib.get("cs") == "c" for e in (dp, ar))
        ct_raw = None
        for event in (dp, ar):  # departure preferred, matching plan semantics
            if event is not None and event.attrib.get("ct"):
                ct_raw = event.attrib["ct"]
                break
        changes.append(
            Change(
                stop_id=s.attrib["id"],
                changed_time=parse_ts(ct_raw) if ct_raw else None,
                is_canceled=canceled,
            )
        )
    return changes
