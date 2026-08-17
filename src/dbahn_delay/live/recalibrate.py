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
graded — but the fit always uses the model's RAW outputs, which are sealed
alongside them. Fitting on corrected outputs while applying the result to
raw ones is a composition error: each day's curve then measures only the
*residual* bias, under-corrects when applied to raw predictions, and the
loop oscillates (seen live 2026-08: ECE 0.021 -> 0.066 over a week).

With raw-fitting the curve is a genuine self-test: after a retrain absorbs
a regime, the fresh model needs less correction and the fitted curve
flattens on its own.

The correction also has to EARN its place. It is fitted on the older part
of the window and scored against doing nothing on the most recent day; a
half that does not beat raw outputs there is neutralised, and if neither
half helps the calibration file is removed entirely. Without that gate the
layer keeps "fixing" a model that has already been repaired — live on
2026-08-15 the raw model scored ECE 0.009 while the stale curve pushed it
to 0.088.

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
from dbahn_delay.models.evaluate import classification_metrics

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
    # Fit on the model's RAW outputs, never on the corrected ones it produced
    # under yesterday's curve: fitting on corrected values and applying the
    # result to raw ones under-corrects, and the daily loop oscillates
    # (observed 2026-08: ECE 0.021 -> 0.066 creep). Older rows have no raw
    # columns; there the sealed values ARE raw (no calibration was active).
    for col in ("raw_delay_probability", "raw_delay_p90_min"):
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    pairs = (
        joined.with_columns(
            is_canceled=pl.col("is_canceled").fill_null(False),
            actual_delay_min=(pl.col("changed_time") - pl.col("scheduled_time"))
            .dt.total_minutes()
            .fill_null(0),
            model_probability=pl.coalesce("raw_delay_probability", "delay_probability"),
            model_p90_min=pl.coalesce("raw_delay_p90_min", "delay_p90_min"),
        )
        .filter(
            ~pl.col("is_canceled"),
            # only settled outcomes (yesterday's file can hold early-today stops)
            pl.col("scheduled_time") < now - timedelta(hours=SETTLED_HOURS),
        )
        .select(
            "model_version",
            "scheduled_time",
            "model_probability",
            "model_p90_min",
            "actual_delay_min",
        )
    )
    # A calibration curve belongs to ONE model: after a promotion the old
    # model's bias would over-correct the new one. Keep only the newest
    # model's pairs (the min-sample guard then holds the layer at identity
    # until enough fresh evidence exists).
    if pairs.height:
        current = pairs.sort("model_version").tail(1)["model_version"][0]
        pairs = pairs.filter(pl.col("model_version") == current)
    return pairs


def fit_curve(pairs: pl.DataFrame) -> tuple[list[float], list[float], float]:
    """Isotonic probability curve + conformal p90 offset from these pairs."""
    from sklearn.isotonic import IsotonicRegression

    prob = pairs["model_probability"].to_numpy().astype(np.float64)
    outcome = (pairs["actual_delay_min"] >= DELAYED_THRESHOLD_MIN).to_numpy().astype(np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(prob, outcome)
    residuals = pairs["actual_delay_min"].to_numpy().astype(np.float64) - pairs[
        "model_p90_min"
    ].to_numpy().astype(np.float64)
    return (
        [round(float(v), 6) for v in iso.X_thresholds_],
        [round(float(v), 6) for v in iso.y_thresholds_],
        # negative when the model over-covers: shrinking is honest too
        round(float(np.quantile(residuals, TARGET_COVERAGE)), 2),
    )


def validate(
    holdout: pl.DataFrame, curve_x: list[float], curve_y: list[float], p90_delta: float
) -> dict[str, Any]:
    """Score the candidate correction against doing nothing, out of sample.

    The layer exists to fix a miscalibrated model; when the model is fine
    (after a retrain, or after a train/serve skew is repaired) the same
    correction actively harms — seen live 2026-08-15: raw ECE 0.009,
    corrected 0.088. So each half must beat raw on unseen days or it is
    dropped, exactly like the champion/challenger promotion gate.
    """
    prob = holdout["model_probability"].to_numpy().astype(np.float64)
    p90 = holdout["model_p90_min"].to_numpy().astype(np.float64)
    actual = holdout["actual_delay_min"].to_numpy().astype(np.float64)
    outcome = (actual >= DELAYED_THRESHOLD_MIN).astype(np.float64)

    ece_raw = classification_metrics(outcome, prob)["ece"]
    ece_corrected = classification_metrics(outcome, np.interp(prob, curve_x, curve_y))["ece"]
    cov_raw = float((actual <= p90).mean())
    cov_corrected = float((actual <= p90 + p90_delta).mean())
    return {
        "ece_raw": round(ece_raw, 4),
        "ece_corrected": round(ece_corrected, 4),
        "coverage_raw": round(cov_raw, 4),
        "coverage_corrected": round(cov_corrected, 4),
        "n_holdout": holdout.height,
        "probability_correction_helps": bool(ece_corrected < ece_raw),
        "p90_correction_helps": bool(
            abs(cov_corrected - TARGET_COVERAGE) < abs(cov_raw - TARGET_COVERAGE)
        ),
    }


def fit_calibration(pairs: pl.DataFrame, now: datetime) -> dict[str, Any] | None:
    if pairs.height < MIN_SAMPLES:
        logger.warning("only %d settled pairs (< %d) - keeping identity", pairs.height, MIN_SAMPLES)
        return None

    # Hold out the most recent day: a correction learned on older evidence
    # has to prove itself on days it has not seen.
    cutoff = now - timedelta(days=1)
    fit_part = pairs.filter(pl.col("scheduled_time") < cutoff)
    holdout = pairs.filter(pl.col("scheduled_time") >= cutoff)
    if fit_part.height < MIN_SAMPLES // 2 or holdout.height < MIN_SAMPLES // 4:
        logger.warning(
            "cannot split for validation (fit=%d, holdout=%d) - keeping identity",
            fit_part.height,
            holdout.height,
        )
        return None

    candidate_x, candidate_y, candidate_delta = fit_curve(fit_part)
    checks = validate(holdout, candidate_x, candidate_y, candidate_delta)
    if not (checks["probability_correction_helps"] or checks["p90_correction_helps"]):
        logger.warning("correction does not beat raw outputs on the holdout: %s", checks)
        return None

    # Earned its place: refit on everything, then neutralise the half that
    # did not help (identity curve / zero offset).
    curve_x, curve_y, p90_delta = fit_curve(pairs)
    if not checks["probability_correction_helps"]:
        curve_x, curve_y = [0.0, 1.0], [0.0, 1.0]
    if not checks["p90_correction_helps"]:
        p90_delta = 0.0
    return {
        "created_at": now.isoformat(),
        "model_version": str(pairs["model_version"][0]),
        "n_samples": pairs.height,
        "lookback_days": LOOKBACK_DAYS,
        "curve_x": curve_x,
        "curve_y": curve_y,
        "p90_delta_min": p90_delta,
        "target_coverage": TARGET_COVERAGE,
        "validation": checks,
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
        # No correction earned its place today: stand down rather than keep
        # yesterday's, which may now be doing harm.
        if calibration_path().exists():
            calibration_path().unlink()
            logger.warning("no correction beats raw outputs - calibration removed")
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
