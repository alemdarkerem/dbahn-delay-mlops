"""Daily output recalibration: correct sealed outputs for regime drift.

Diagnosis (2026-07-29): the live network runs far hotter than the training
window (delayed share 0.25-0.33 vs 0.19) and a tree ensemble cannot
extrapolate beyond its training distribution — every probability bucket
came out ~10pp optimistic even with fresh features and live network state.
Monthly retraining absorbs a new regime eventually; this layer corrects
outputs every morning in between:

- classifier: isotonic curve fit on the trailing week of sealed
  probabilities vs realized outcomes, exported as a piecewise-linear
  mapping (portable JSON, no pickles);
- p90: conformal additive offset — the residual quantile that restores
  empirical coverage to the 0.90 target.

Model weights untouched. The API reloads the JSON on mtime change and
applies it inside /predict, so corrected numbers are what gets sealed and
graded. The feedback loop is intended: once sealed outputs are well
calibrated, the fitted curve approaches identity and the offset shrinks
to zero — a built-in self-test (watch it after each retrain).

Runs as the tail of the morning evaluation; standalone:
``python -m dbahn_delay.live.recalibrate``
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.features.build import DELAYED_THRESHOLD_MIN

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
LOOKBACK_DAYS = 7
MIN_SAMPLES = 2000  # below this a curve would fit noise; keep identity
TARGET_COVERAGE = 0.90
SETTLED_HOURS = 3  # outcome considered final this long after the scheduled time


def calibration_path() -> Path:
    return settings.live_dir / "recalibration" / "calibration.json"


def collect_pairs(now: datetime) -> pl.DataFrame | None:
    """Sealed (probability, p90) vs realized outcome, evaluate_day rules."""
    days = [(now.date() - timedelta(days=i)).isoformat() for i in range(1, LOOKBACK_DAYS + 1)]
    pred_frames = [
        pl.read_parquet(p)
        for d in days
        if (p := settings.live_dir / "predictions" / f"{d}.parquet").exists()
    ]
    if not pred_frames:
        return None
    predictions = pl.concat(pred_frames, how="diagonal_relaxed")
    change_days = [(now.date() - timedelta(days=i)).isoformat() for i in range(LOOKBACK_DAYS + 1)]
    change_frames = [
        pl.read_parquet(p)
        for d in change_days
        if (p := settings.live_dir / "changes" / f"{d}.parquet").exists()
    ]
    if change_frames:
        changes = (
            pl.concat(change_frames, how="diagonal_relaxed")
            .sort("observed_at")
            .unique(subset="stop_id", keep="last")
        )
        joined = predictions.join(
            changes.select("stop_id", "changed_time", "is_canceled"), on="stop_id", how="left"
        )
    else:
        joined = predictions.with_columns(
            changed_time=pl.lit(None, dtype=pl.Datetime("us", "Europe/Berlin")),
            is_canceled=pl.lit(None, dtype=pl.Boolean),
        )
    pairs = (
        joined.with_columns(
            is_canceled=pl.col("is_canceled").fill_null(False),
            actual_delay_min=(pl.col("changed_time") - pl.col("scheduled_time"))
            .dt.total_minutes()
            .fill_null(0),
        )
        .filter(
            ~pl.col("is_canceled"),
            # only settled outcomes (yesterday's file can hold early-today stops)
            pl.col("scheduled_time") < now - timedelta(hours=SETTLED_HOURS),
        )
        .select("model_version", "delay_probability", "delay_p90_min", "actual_delay_min")
    )
    # A calibration curve belongs to ONE model: after a promotion the old
    # model's bias would over-correct the new one. Keep only the newest
    # model's pairs (the min-sample guard then holds the layer at identity
    # until enough fresh evidence exists).
    if pairs.height:
        current = pairs.sort("model_version").tail(1)["model_version"][0]
        pairs = pairs.filter(pl.col("model_version") == current)
    return pairs


def fit_calibration(pairs: pl.DataFrame, now: datetime) -> dict[str, Any] | None:
    if pairs.height < MIN_SAMPLES:
        logger.warning("only %d settled pairs (< %d) - keeping identity", pairs.height, MIN_SAMPLES)
        return None
    from sklearn.isotonic import IsotonicRegression

    prob = pairs["delay_probability"].to_numpy().astype(np.float64)
    outcome = (pairs["actual_delay_min"] >= DELAYED_THRESHOLD_MIN).to_numpy().astype(np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(prob, outcome)
    residuals = pairs["actual_delay_min"].to_numpy().astype(np.float64) - pairs[
        "delay_p90_min"
    ].to_numpy().astype(np.float64)
    return {
        "created_at": now.isoformat(),
        "model_version": str(pairs["model_version"][0]),
        "n_samples": pairs.height,
        "lookback_days": LOOKBACK_DAYS,
        "curve_x": [round(float(v), 6) for v in iso.X_thresholds_],
        "curve_y": [round(float(v), 6) for v in iso.y_thresholds_],
        # negative when the model over-covers: shrinking is honest too
        "p90_delta_min": round(float(np.quantile(residuals, TARGET_COVERAGE)), 2),
        "target_coverage": TARGET_COVERAGE,
    }


def drop_stale_calibration(current_version: str) -> None:
    """Remove a calibration left behind by a previous model.

    Without this, a promotion that does not yet have enough fresh pairs to
    refit would keep serving the OLD model's correction — and sealed
    predictions are immutable, so those hours could never be repaired.
    """
    path = calibration_path()
    if not path.exists():
        return
    try:
        stamped = json.loads(path.read_text()).get("model_version")
    except Exception:
        stamped = None
    if stamped != current_version:
        path.unlink()
        logger.warning(
            "dropped calibration fitted for %s (now serving %s)",
            stamped or "unknown",
            current_version,
        )


def rebuild(now: datetime | None = None) -> dict[str, Any] | None:
    now = now or datetime.now(tz=BERLIN)
    pairs = collect_pairs(now)
    if pairs is None or pairs.is_empty():
        logger.warning("no live pairs yet - recalibration skipped")
        return None
    drop_stale_calibration(str(pairs["model_version"][0]))
    calibration = fit_calibration(pairs, now)
    if calibration is None:
        return None
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(calibration))
    tmp.replace(path)  # atomic: the API never sees a half-written file
    logger.info(
        "recalibration written: n=%d, p90_delta=%+.1f min",
        calibration["n_samples"],
        calibration["p90_delta_min"],
    )
    return calibration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rebuild()
