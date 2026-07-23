"""Training entry point: walk-forward CV, baselines vs LightGBM, MLflow, artifact.

Per fold: fit baselines + three LightGBM models (classifier, q50, q90) on the
train window, evaluate everything on the validation month, log to MLflow.
After CV: refit on the freshest window and export a versioned artifact bundle
for serving.

Usage: ``python -m dbahn_delay.models.train`` (or ``make train``).
Smoke mode for tests/CI: ``DBAHN_SMOKE_ROWS=<n>`` caps rows per split.
"""

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from dbahn_delay.config import settings
from dbahn_delay.models.baseline import fit_baseline, predict_baseline
from dbahn_delay.models.cv import Fold, walk_forward_folds
from dbahn_delay.models.evaluate import (
    classification_metrics,
    quantile_metrics,
    reliability_plot,
)

logger = logging.getLogger(__name__)

VAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
TRAIN_WINDOW_MONTHS = 12

CATEGORICAL = ["station_name", "train_type", "train_number"]
NUMERIC = [
    "train_line_station_num",
    "scheduled_hour",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "train_mean_delay_14d",
    "train_delayed_rate_14d",
    "train_mean_delay_30d",
    "train_delayed_rate_30d",
    "train_count_30d",
    "station_type_mean_delay_14d",
    "station_type_delayed_rate_14d",
    "station_type_mean_delay_30d",
    "station_type_delayed_rate_30d",
    "station_type_count_30d",
    "type_mean_delay_14d",
    "type_delayed_rate_14d",
    "type_mean_delay_30d",
    "type_delayed_rate_30d",
    "type_count_30d",
]
FEATURES = CATEGORICAL + NUMERIC

LGB_COMMON: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 127,
    "min_child_samples": 100,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def load_split(months: list[str]) -> pl.DataFrame:
    lf = (
        pl.scan_parquet(settings.processed_dir / "features.parquet")
        .filter(pl.col("fold_month").is_in(months))
        .select([*FEATURES, "target_delayed6", "target_delay_min"])
    )
    smoke_rows = int(os.environ.get("DBAHN_SMOKE_ROWS", "0"))
    if smoke_rows:
        lf = lf.head(smoke_rows)
    return lf.collect()


def to_lgb_frame(df: pl.DataFrame) -> Any:
    """polars -> pandas, memory-conscious.

    Strings are cast to polars Categorical first so the pandas conversion
    yields category dtype via arrow dictionaries — never Python-object
    columns (which explode to tens of GB at 20M+ rows). Numerics go to
    float32. LightGBM stores the training categories in the booster and
    re-maps prediction frames by value, so per-frame category codes are safe.
    """
    return (
        df.select(FEATURES)
        .with_columns(
            [pl.col(c).cast(pl.Categorical) for c in CATEGORICAL]
            + [pl.col(c).cast(pl.Float32) for c in NUMERIC]
        )
        .to_pandas()
    )


def fit_models(x: Any, y_cls: Any, y_reg: Any) -> dict[str, lgb.LGBMModel]:
    clf = lgb.LGBMClassifier(objective="binary", **LGB_COMMON)
    clf.fit(x, y_cls, categorical_feature=CATEGORICAL)

    models: dict[str, lgb.LGBMModel] = {"clf": clf}
    for alpha, name in ((0.5, "q50"), (0.9, "q90")):
        reg = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGB_COMMON)
        reg.fit(x, y_reg, categorical_feature=CATEGORICAL)
        models[name] = reg
    return models


def fit_models_from_split(months: list[str]) -> dict[str, lgb.LGBMModel]:
    """Load a train window, convert, and free the polars frame before fitting."""
    train = load_split(months)
    x = to_lgb_frame(train)
    y_cls = train["target_delayed6"].to_numpy()
    y_reg = train["target_delay_min"].to_numpy()
    del train
    return fit_models(x, y_cls, y_reg)


def evaluate_fold(fold: Fold, artifacts_dir: Path) -> dict[str, float]:
    logger.info("fold %s: loading data", fold.val_month)
    train = load_split(fold.train_months)
    val = load_split([fold.val_month])
    logger.info("fold %s: train=%d rows, val=%d rows", fold.val_month, len(train), len(val))

    tables = fit_baseline(train.lazy())
    val_bl = predict_baseline(tables, val.lazy())

    x_train = to_lgb_frame(train)
    y_cls_train = train["target_delayed6"].to_numpy()
    y_reg_train = train["target_delay_min"].to_numpy()
    del train  # free the polars frame before LightGBM allocates its datasets

    models = fit_models(x_train, y_cls_train, y_reg_train)
    del x_train, y_cls_train, y_reg_train
    x_val = to_lgb_frame(val)
    prob = np.asarray(models["clf"].predict_proba(x_val))[:, 1]
    p50 = np.asarray(models["q50"].predict(x_val), dtype=np.float64)
    p90 = np.asarray(models["q90"].predict(x_val), dtype=np.float64)

    y_cls = val["target_delayed6"].to_numpy().astype(np.float64)
    y_reg = val["target_delay_min"].to_numpy().astype(np.float64)

    metrics: dict[str, float] = {}
    for name, value in classification_metrics(y_cls, prob).items():
        metrics[f"model_{name}"] = value
    for name, value in classification_metrics(
        y_cls, val_bl["baseline_prob"].to_numpy().astype(np.float64)
    ).items():
        metrics[f"baseline_{name}"] = value
    for name, value in quantile_metrics(y_reg, p50, p90).items():
        metrics[f"model_{name}"] = value
    for name, value in quantile_metrics(
        y_reg,
        val_bl["baseline_p50"].to_numpy().astype(np.float64),
        val_bl["baseline_p90"].to_numpy().astype(np.float64),
    ).items():
        metrics[f"baseline_{name}"] = value

    reliability_plot(y_cls, prob, artifacts_dir / f"reliability_{fold.val_month}.png")
    return metrics


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def export_bundle(models: dict[str, lgb.LGBMModel], train_months: list[str]) -> Path:
    """Save boosters + metadata as the versioned serving artifact."""
    version = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    bundle_dir = Path("models") / version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.booster_.save_model(bundle_dir / f"{name}.txt")
    meta = {
        "version": version,
        "git_sha": git_sha(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "train_months": train_months,
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "lgb_params": {k: str(v) for k, v in LGB_COMMON.items()},
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Serving needs the latest rolling stats as lookup tables (feature snapshot).
    from dbahn_delay.features.snapshot import export_snapshot

    export_snapshot(bundle_dir)
    logger.info("Exported model bundle to %s", bundle_dir)
    return bundle_dir


def main() -> None:
    mlflow.set_experiment("dbahn-delay")
    folds = walk_forward_folds(VAL_MONTHS, TRAIN_WINDOW_MONTHS)
    artifacts_dir = Path("mlruns_artifacts_tmp")
    artifacts_dir.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=f"cv-{git_sha()}"):
        mlflow.log_params(
            {
                "val_months": ",".join(VAL_MONTHS),
                "train_window_months": TRAIN_WINDOW_MONTHS,
                "features": len(FEATURES),
                "git_sha": git_sha(),
                **{f"lgb_{k}": v for k, v in LGB_COMMON.items()},
            }
        )
        per_fold: list[dict[str, float]] = []
        for fold in folds:
            metrics = evaluate_fold(fold, artifacts_dir)
            per_fold.append(metrics)
            mlflow.log_metrics({f"{k}_{fold.val_month}": v for k, v in metrics.items()})
            logger.info("fold %s: %s", fold.val_month, json.dumps(metrics, indent=2))

        # Aggregate: mean over folds
        agg = {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]}
        mlflow.log_metrics({f"mean_{k}": v for k, v in agg.items()})
        mlflow.log_artifacts(str(artifacts_dir))
        logger.info("aggregate: %s", json.dumps(agg, indent=2))

        # Final model on the freshest window, exported for serving
        final_months = walk_forward_folds([VAL_MONTHS[-1]], TRAIN_WINDOW_MONTHS)[0]
        freshest = [*final_months.train_months[1:], final_months.val_month]
        logger.info("final fit on %s..%s", freshest[0], freshest[-1])
        final_models = fit_models_from_split(freshest)
        bundle = export_bundle(final_models, freshest)
        mlflow.log_param("bundle_path", str(bundle))

        (artifacts_dir / "cv_summary.json").write_text(json.dumps(agg, indent=2))
        mlflow.log_artifact(str(artifacts_dir / "cv_summary.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
