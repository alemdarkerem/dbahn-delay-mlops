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
from pydantic import BaseModel, Field

from dbahn_delay.serving.features import assemble_features
from dbahn_delay.serving.loader import ModelBundle

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
    stats_age_days: int


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
    return HealthResponse(
        status="ok",
        model_version=bundle.version,
        stats_age_days=bundle.stats_age_days(datetime.now(tz=BERLIN).date()),
    )


@app.get("/model-info")
def model_info() -> dict[str, object]:
    bundle = bundle_or_503()
    return {
        "metadata": bundle.metadata,
        "snapshot_entities": {k: len(v) for k, v in bundle.stats.items()},
    }
