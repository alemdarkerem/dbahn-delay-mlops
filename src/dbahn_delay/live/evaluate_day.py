"""Daily accountability report: yesterday's predictions vs what actually happened.

Ground-truth rule (documented assumption, same as the dataset collector's):
a stop with an observed change gets actual_delay = changed - scheduled; a stop
never seen in the change feed counts as on time (0 min). Canceled stops are
excluded from delay metrics (delay is meaningless for them, see EDA finding 4).

Runs every morning for the previous day:
``python -m dbahn_delay.live.evaluate_day [YYYY-MM-DD]``
"""

import logging
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.live.fetch import changes_path, predictions_path
from dbahn_delay.models.evaluate import classification_metrics, quantile_metrics

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
DELAYED_THRESHOLD_MIN = 6


def load_day(day: str) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    pred_path = predictions_path(day)
    if not pred_path.exists():
        logger.warning("no predictions logged for %s", day)
        return None
    predictions = pl.read_parquet(pred_path)
    # Changes for the day, plus the following day's file (late-night stops get
    # their final change observations after midnight).
    next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    frames = [pl.read_parquet(p) for p in (changes_path(day), changes_path(next_day)) if p.exists()]
    if frames:
        changes = pl.concat(frames).sort("observed_at").unique(subset="stop_id", keep="last")
    else:
        changes = pl.DataFrame(
            schema={
                "stop_id": pl.String,
                "changed_time": pl.Datetime("us", "Europe/Berlin"),
                "is_canceled": pl.Boolean,
                "observed_at": pl.Datetime("us", "Europe/Berlin"),
            }
        )
    return predictions, changes


def evaluate(predictions: pl.DataFrame, changes: pl.DataFrame) -> dict[str, float]:
    joined = predictions.join(changes, on="stop_id", how="left").with_columns(
        is_canceled=pl.col("is_canceled").fill_null(False),
        actual_delay_min=(
            (pl.col("changed_time") - pl.col("scheduled_time")).dt.total_minutes()
        ).fill_null(0),
    )
    n_canceled = int(joined["is_canceled"].sum())
    usable = joined.filter(~pl.col("is_canceled"))

    y_true = (usable["actual_delay_min"] >= DELAYED_THRESHOLD_MIN).to_numpy().astype(np.float64)
    y_prob = usable["delay_probability"].to_numpy().astype(np.float64)
    y_delay = usable["actual_delay_min"].to_numpy().astype(np.float64)
    p50 = usable["delay_p50_min"].to_numpy().astype(np.float64)
    p90 = usable["delay_p90_min"].to_numpy().astype(np.float64)

    metrics: dict[str, float] = {
        "n_predictions": float(len(joined)),
        "n_canceled": float(n_canceled),
        "n_evaluated": float(len(usable)),
        "share_with_observed_change": float(
            usable.select(pl.col("changed_time").is_not_null().mean()).item() or 0.0
        ),
        "coverage_train_share": float(
            usable.select((pl.col("coverage") == "train").mean()).item() or 0.0
        ),
    }
    if len(usable) > 0:
        if 0.0 < y_true.mean() < 1.0:
            metrics.update(classification_metrics(y_true, y_prob))
        else:
            metrics["base_rate"] = float(y_true.mean())
        metrics.update(quantile_metrics(y_delay, p50, p90))
    return metrics


# Retraining trigger rules, evaluated over the daily series. The trigger
# ALERTS (surfaced on /monitoring); a human runs `make retrain` — the
# promotion gate stays human-approved by design.
ECE_LIMIT = 0.08
AUC_FLOOR = 0.70
CONSECUTIVE_DAYS = 3
FRESHNESS_LIMIT_DAYS = 45


def trigger_reasons(series: pl.DataFrame, feature_freshness_days: int | None = None) -> list[str]:
    """Reasons to retrain, based on the last CONSECUTIVE_DAYS of metrics."""
    reasons = []
    tail = series.sort("day").tail(CONSECUTIVE_DAYS)
    if tail.height >= CONSECUTIVE_DAYS:
        if "ece" in tail.columns and bool((tail["ece"] > ECE_LIMIT).all()):
            reasons.append(f"ece > {ECE_LIMIT} for {CONSECUTIVE_DAYS} consecutive days")
        if "roc_auc" in tail.columns and bool((tail["roc_auc"] < AUC_FLOOR).all()):
            reasons.append(f"roc_auc < {AUC_FLOOR} for {CONSECUTIVE_DAYS} consecutive days")
    if feature_freshness_days is not None and feature_freshness_days > FRESHNESS_LIMIT_DAYS:
        reasons.append(f"feature freshness {feature_freshness_days}d > {FRESHNESS_LIMIT_DAYS}d")
    return reasons


def write_report(day: str, metrics: dict[str, float]) -> None:
    reports_dir = settings.live_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Daily report — {day}", ""]
    lines += [f"- **{k}**: {v:.4f}" for k, v in metrics.items()]
    (reports_dir / f"{day}.md").write_text("\n".join(lines) + "\n")

    metrics_path = settings.live_dir / "daily_metrics.parquet"
    row = pl.DataFrame([{"day": day, **metrics}])
    if metrics_path.exists():
        existing = pl.read_parquet(metrics_path).filter(pl.col("day") != day)
        series = pl.concat([existing, row], how="diagonal")
    else:
        series = row
    reasons = trigger_reasons(series)
    series = series.with_columns(
        retraining_recommended=pl.when(pl.col("day") == day)
        .then(pl.lit(bool(reasons)))
        .otherwise(
            pl.col("retraining_recommended")
            if "retraining_recommended" in series.columns
            else pl.lit(False)
        ),
        trigger_reasons=pl.when(pl.col("day") == day)
        .then(pl.lit("; ".join(reasons)))
        .otherwise(
            pl.col("trigger_reasons") if "trigger_reasons" in series.columns else pl.lit("")
        ),
    )
    series.sort("day").write_parquet(metrics_path)
    if reasons:
        logger.warning("RETRAINING RECOMMENDED: %s", "; ".join(reasons))


def main() -> None:
    day = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (datetime.now(tz=BERLIN).date() - timedelta(days=1)).isoformat()
    )
    loaded = load_day(day)
    if loaded is None:
        return
    predictions, changes = loaded
    metrics = evaluate(predictions, changes)
    write_report(day, metrics)
    logger.info("report for %s: %s", day, metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
