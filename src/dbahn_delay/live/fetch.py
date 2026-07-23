"""Hourly live cycle: fetch upcoming stops, log sealed predictions + changes.

Predictions are made BEFORE the event and appended once per stop id (first
prediction wins — re-predicting closer to departure would leak an unfair
advantage into the accuracy reports). Observed changes (delays/cancellations)
are upserted so the daily evaluator can join them as ground truth.

Runs hourly via cron/Coolify Scheduled Tasks: ``python -m dbahn_delay.live.fetch``
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.live.client import TimetablesClient
from dbahn_delay.live.parse import Change, PlannedStop, parse_changes, parse_plan
from dbahn_delay.live.stations import load_station_map
from dbahn_delay.serving.app import feature_matrix
from dbahn_delay.serving.features import assemble_features
from dbahn_delay.serving.loader import ModelBundle

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
HOURS_AHEAD = (1, 2)  # predict for stops scheduled in the next two hour-slices


def predictions_path(day: str) -> Any:
    return settings.live_dir / "predictions" / f"{day}.parquet"


def changes_path(day: str) -> Any:
    return settings.live_dir / "changes" / f"{day}.parquet"


def predict_stops(
    bundle: ModelBundle, stops: list[PlannedStop], predicted_at: datetime
) -> pl.DataFrame:
    rows = []
    for stop in stops:
        features, coverage = assemble_features(
            bundle,
            station_name=stop.station_name,
            train_type=stop.train_type,
            train_number=stop.train_number,
            scheduled_time=stop.scheduled_time,
            train_line_station_num=None,
        )
        x = feature_matrix(bundle, features)
        rows.append(
            {
                "stop_id": stop.stop_id,
                "station_name": stop.station_name,
                "train_type": stop.train_type,
                "train_number": stop.train_number,
                "scheduled_time": stop.scheduled_time,
                "delay_probability": float(bundle.clf.predict(x)[0]),
                "delay_p50_min": max(0.0, float(bundle.q50.predict(x)[0])),
                "delay_p90_min": max(0.0, float(bundle.q90.predict(x)[0])),
                "coverage": coverage,
                "model_version": bundle.version,
                "predicted_at": predicted_at,
            }
        )
    df = pl.DataFrame(rows)
    return df.with_columns(delay_p90_min=pl.max_horizontal("delay_p90_min", "delay_p50_min"))


def append_new_predictions(new: pl.DataFrame, day: str) -> int:
    """Append rows whose stop_id is not logged yet (first prediction wins)."""
    path = predictions_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(path)
        new = new.filter(~pl.col("stop_id").is_in(existing["stop_id"]))
        if new.is_empty():
            return 0
        pl.concat([existing, new]).write_parquet(path)
    else:
        new.write_parquet(path)
    return new.height


def upsert_changes(changes: list[Change], observed_at: datetime, day: str) -> int:
    """Keep the LATEST observed change per stop id."""
    if not changes:
        return 0
    path = changes_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame(
        [
            {
                "stop_id": c.stop_id,
                "changed_time": c.changed_time,
                "is_canceled": c.is_canceled,
                "observed_at": observed_at,
            }
            for c in changes
        ],
        schema_overrides={"changed_time": pl.Datetime("us", "Europe/Berlin")},
    )
    combined = pl.concat([pl.read_parquet(path), new]) if path.exists() else new
    deduped = combined.sort("observed_at").unique(subset="stop_id", keep="last")
    deduped.write_parquet(path)
    return new.height


def run_cycle(now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(tz=BERLIN)
    bundle = ModelBundle.load(settings_model_dir())
    stations = load_station_map()
    client = TimetablesClient()

    all_stops: list[PlannedStop] = []
    all_changes: list[Change] = []
    ok = failed = 0
    try:
        for name, info in stations.items():
            try:
                for ahead in HOURS_AHEAD:
                    slot = now + timedelta(hours=ahead)
                    xml_text = client.fetch_plan(
                        info["eva"], slot.strftime("%y%m%d"), slot.strftime("%H")
                    )
                    for stop in parse_plan(xml_text):
                        # API returns its own station spelling; use the panel
                        # name so features match training vocabulary.
                        all_stops.append(
                            PlannedStop(
                                stop_id=stop.stop_id,
                                station_name=name,
                                train_type=stop.train_type,
                                train_number=stop.train_number,
                                scheduled_time=stop.scheduled_time,
                                has_departure=stop.has_departure,
                            )
                        )
                all_changes.extend(parse_changes(client.fetch_changes(info["eva"])))
                ok += 1
            except Exception:
                logger.exception("station %r failed, continuing", name)
                failed += 1
    finally:
        client.close()

    day = now.strftime("%Y-%m-%d")
    n_predictions = 0
    if all_stops:
        predictions = predict_stops(bundle, all_stops, predicted_at=now)
        n_predictions = append_new_predictions(predictions, day)
    n_changes = upsert_changes(all_changes, observed_at=now, day=day)

    summary = {
        "stations_ok": ok,
        "stations_failed": failed,
        "stops_fetched": len(all_stops),
        "new_predictions": n_predictions,
        "changes_recorded": n_changes,
    }
    logger.info("cycle done: %s", summary)
    return summary


def settings_model_dir() -> Any:
    import os
    from pathlib import Path

    bundle_dir = os.environ.get("DBAHN_MODEL_DIR", "")
    if bundle_dir:
        return Path(bundle_dir)
    candidates = sorted(Path("models").glob("*/metadata.json"))
    if not candidates:
        raise RuntimeError("no model bundle found - set DBAHN_MODEL_DIR")
    return candidates[-1].parent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cycle()
