"""FastAPI service exposing the delay model.

Endpoints: POST /predict, GET /health, GET /model-info.
The bundle directory comes from settings (DBAHN_MODEL_DIR).

Run locally: ``make serve`` (uvicorn dbahn_delay.serving.app:app).
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dbahn_delay.serving.features import assemble_features
from dbahn_delay.serving.loader import ModelBundle
from dbahn_delay.serving.overlay import OverlayStore

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")


class PredictRequest(BaseModel):
    station_name: str = Field(min_length=1, examples=["Berlin Hbf"])
    train_type: str = Field(min_length=1, examples=["ICE"])
    train_number: str = Field(min_length=1, examples=["1601"])
    scheduled_time: datetime = Field(
        description="Scheduled stop time; naive values are interpreted as Europe/Berlin",
        examples=["2026-07-24T09:34:00"],
    )
    train_line_station_num: int | None = Field(default=None, ge=0, le=100)


class PredictResponse(BaseModel):
    delay_probability: float = Field(description="P(delay >= 6 min)")
    delay_p50_min: float = Field(description="Median predicted delay in minutes")
    delay_p90_min: float = Field(description="90th-percentile predicted delay in minutes")
    coverage: str = Field(
        description="Feature coverage: train | station_type | type | cold (fallback depth)"
    )
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    stats_age_days: int  # bundle snapshot age (training-time features)
    feature_freshness_days: int  # freshest usable source (overlay wins if present)


def load_bundle() -> ModelBundle | None:
    bundle_dir = os.environ.get("DBAHN_MODEL_DIR", "")
    if not bundle_dir:
        candidates = sorted(Path("models").glob("*/metadata.json"))
        bundle_dir = str(candidates[-1].parent) if candidates else ""
    try:
        bundle = ModelBundle.load(Path(bundle_dir))
        logger.info("loaded model bundle %s from %s", bundle.version, bundle_dir)
        return bundle
    except (OSError, KeyError, ValueError):
        logger.exception("failed to load model bundle from %r", bundle_dir)
        return None


app = FastAPI(
    title="DB Delay Prediction API",
    description="Deutsche Bahn train delay predictions (data by Deutsche Bahn, CC BY 4.0)",
    version="0.1.0",
)
_bundle = load_bundle()


def _overlay_store() -> OverlayStore:
    from dbahn_delay.config import settings

    return OverlayStore(settings.live_dir / "snapshot_overlay")


_overlay = _overlay_store()


def bundle_or_503() -> ModelBundle:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model bundle not loaded")
    return _bundle


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    bundle = bundle_or_503()
    scheduled = request.scheduled_time
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=BERLIN)

    row, coverage = assemble_features(
        bundle,
        station_name=request.station_name,
        train_type=request.train_type,
        train_number=request.train_number,
        scheduled_time=scheduled,
        train_line_station_num=request.train_line_station_num,
        overlay=_overlay,
    )
    x = feature_matrix(bundle, row)
    prob = float(bundle.clf.predict(x)[0])
    p50 = max(0.0, float(bundle.q50.predict(x)[0]))
    p90 = max(0.0, float(bundle.q90.predict(x)[0]))
    # Quantile crossings can happen with independently trained models; never
    # return an inconsistent pair to the user.
    p90 = max(p90, p50)
    return PredictResponse(
        delay_probability=round(prob, 4),
        delay_p50_min=round(p50, 1),
        delay_p90_min=round(p90, 1),
        coverage=coverage,
        model_version=bundle.version,
    )


def feature_matrix(bundle: ModelBundle, row: dict[str, object]) -> "np.ndarray[Any, Any]":
    """Order features exactly as at training time; encode categoricals.

    Boosters trained through the sklearn API with pandas categoricals store
    their category lists; we rebuild codes from those lists so values map
    identically (unseen category -> NaN -> LightGBM missing).
    """
    features: list[str] = bundle.metadata["features"]
    categorical: list[str] = bundle.metadata["categorical_features"]
    pandas_categorical: list[list[str]] = bundle.clf.pandas_categorical or []
    cat_maps = {
        col: {v: i for i, v in enumerate(values)}
        for col, values in zip(categorical, pandas_categorical, strict=False)
    }
    values = []
    for name in features:
        v = row.get(name)
        if name in cat_maps:
            values.append(float(cat_maps[name].get(str(v), np.nan)))
        elif v is None:
            values.append(np.nan)
        else:
            values.append(float(v))  # type: ignore[arg-type,unused-ignore]
    return np.array([values], dtype=np.float64)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = bundle_or_503()
    today = datetime.now(tz=BERLIN).date()
    bundle_age = bundle.stats_age_days(today)
    overlay_newest = _overlay.newest_join_date()
    freshness = min(bundle_age, (today - overlay_newest).days) if overlay_newest else bundle_age
    return HealthResponse(
        status="ok",
        model_version=bundle.version,
        stats_age_days=bundle_age,
        feature_freshness_days=freshness,
    )


@app.get("/model-info")
def model_info() -> dict[str, object]:
    bundle = bundle_or_503()
    return {
        "metadata": bundle.metadata,
        "snapshot_entities": {k: len(v) for k, v in bundle.stats.items()},
    }


@app.get("/monitoring")
def monitoring() -> dict[str, object]:
    """Last 30 daily accountability reports (prediction vs actual)."""
    from dbahn_delay.config import settings

    metrics_path = settings.live_dir / "daily_metrics.parquet"
    if not metrics_path.exists():
        return {"days": [], "note": "no daily metrics recorded yet"}
    import polars as pl

    rows = pl.read_parquet(metrics_path).sort("day").tail(30)
    latest = rows.tail(1).to_dicts()[0] if rows.height else {}
    return {
        "days": rows.to_dicts(),
        "retraining_recommended": bool(latest.get("retraining_recommended", False)),
        "reasons": [r for r in str(latest.get("trigger_reasons", "")).split("; ") if r],
    }


@app.get("/stations")
def stations() -> dict[str, object]:
    """Panel stations available on the board."""
    from dbahn_delay.live.stations import load_station_map

    return {"stations": sorted(load_station_map())}


@app.get("/board/{station_name}")
def board(station_name: str, limit: int = 25) -> dict[str, object]:
    """Station board: upcoming predictions + departed trains with outcomes.

    Reads today's sealed-prediction and observed-change files. Departed rows
    include the observed delay (no change record => assumed on time), which
    makes the model's accuracy publicly visible per train.
    """
    import polars as pl

    from dbahn_delay.config import settings

    now = datetime.now(tz=BERLIN)
    day = now.strftime("%Y-%m-%d")
    pred_path = settings.live_dir / "predictions" / f"{day}.parquet"
    if not pred_path.exists():
        return {"station": station_name, "upcoming": [], "departed": [], "note": "no data yet"}

    preds = (
        pl.read_parquet(pred_path)
        .filter(pl.col("station_name") == station_name)
        # Wing trains / duplicate plan entries yield two stop ids for the same
        # train at the same minute; show one row (display only — the sealed
        # data and the daily evaluation keep both stop events).
        .unique(subset=["train_type", "train_number", "scheduled_time"], keep="first")
    )
    changes_path = settings.live_dir / "changes" / f"{day}.parquet"
    if changes_path.exists():
        changes = pl.read_parquet(changes_path)
        preds = preds.join(
            changes.select("stop_id", "changed_time", "is_canceled"), on="stop_id", how="left"
        )
    else:
        preds = preds.with_columns(
            changed_time=pl.lit(None, dtype=pl.Datetime("us", "Europe/Berlin")),
            is_canceled=pl.lit(None, dtype=pl.Boolean),
        )

    preds = preds.with_columns(
        is_canceled=pl.col("is_canceled").fill_null(False),
        actual_delay_min=(pl.col("changed_time") - pl.col("scheduled_time"))
        .dt.total_minutes()
        .fill_null(0),
    )
    columns = [
        "train_type",
        "train_number",
        "scheduled_time",
        "delay_probability",
        "delay_p50_min",
        "delay_p90_min",
        "coverage",
        "is_canceled",
        "actual_delay_min",
    ]
    upcoming = (
        preds.filter(pl.col("scheduled_time") >= now)
        .sort("scheduled_time")
        .head(limit)
        .select(columns)
    )
    departed = (
        preds.filter(pl.col("scheduled_time") < now)
        .sort("scheduled_time", descending=True)
        .head(limit)
        .select(columns)
    )
    return {
        "station": station_name,
        "generated_at": now.isoformat(),
        "upcoming": upcoming.to_dicts(),
        "departed": departed.to_dicts(),
    }


@app.get("/", include_in_schema=False)
def index() -> "FileResponse":
    return FileResponse(Path(__file__).parent / "static" / "index.html")
