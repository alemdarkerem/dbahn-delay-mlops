"""Retraining orchestrator: new data -> challenger -> championship -> report.

End-to-end in one command, with a deliberate HUMAN promotion gate at the end
(training needs the big machine, and promoting a model to production is a
human-approved deployment step — by design, not by omission).

Flow:
1. Check the HF dataset for monthly files we don't have. None -> exit
   (``--force`` continues anyway, for plumbing tests).
2. Download new months, rebuild the canonical dataset and feature frame.
3. Championship on the newest complete month M: challenger (fresh fit on the
   12 months ending M-1) vs champion (current bundle) — both predict M cold.
4. Promotion needs ALL of: PR-AUC not worse than champion by >0.002,
   pinball_p90 within +2%, ECE within +0.01. If promoted: final refit on the
   12 months ending M, bundle export, upload runbook printed.

Usage: ``make retrain`` (or ``uv run python -m dbahn_delay.models.retrain [--force]``).
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from huggingface_hub import list_repo_files

from dbahn_delay.config import settings
from dbahn_delay.models.cv import month_range, walk_forward_folds
from dbahn_delay.models.evaluate import classification_metrics, quantile_metrics
from dbahn_delay.models.train import (
    export_bundle,
    fit_models_from_split,
    load_split,
    to_lgb_frame,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("models") / "retrain_reports"

# Promotion tolerances: the challenger must not be meaningfully worse on ANY
# of these while presumably being fresher.
PR_AUC_TOLERANCE = 0.002
PINBALL_P90_RATIO = 1.02
ECE_TOLERANCE = 0.01


def local_months() -> list[str]:
    files = sorted(settings.monthly_raw_dir.glob("data-*.parquet"))
    return [f.stem.removeprefix("data-") for f in files]


def remote_months() -> list[str]:
    files = list_repo_files(settings.hf_dataset_repo, repo_type="dataset")
    return sorted(
        f.removeprefix("monthly_processed_data/data-").removesuffix(".parquet")
        for f in files
        if f.startswith("monthly_processed_data/data-")
    )


def new_months() -> list[str]:
    have = set(local_months())
    return [m for m in remote_months() if m not in have]


def rebuild_data() -> None:
    """Download any new months and regenerate canonical + feature datasets."""
    from dbahn_delay.data.download import download_monthly_data
    from dbahn_delay.data.ingest import build_stops
    from dbahn_delay.features import build as feature_build

    download_monthly_data()
    build_stops()
    feature_build.main()


def score_boosters(clf: Any, q50: Any, q90: Any, val: pl.DataFrame) -> dict[str, float]:
    """Predict a validation month with raw boosters and compute all metrics."""
    x = to_lgb_frame(val)
    y_cls = val["target_delayed6"].to_numpy().astype(np.float64)
    y_reg = val["target_delay_min"].to_numpy().astype(np.float64)
    prob = np.asarray(clf.predict(x), dtype=np.float64)
    if prob.ndim == 2:  # sklearn predict_proba shape
        prob = prob[:, 1]
    p50 = np.asarray(q50.predict(x), dtype=np.float64)
    p90 = np.maximum(np.asarray(q90.predict(x), dtype=np.float64), p50)
    return {
        **classification_metrics(y_cls, prob),
        **quantile_metrics(y_reg, p50, p90),
    }


def champion_boosters() -> tuple[Any, Any, Any, str]:
    import os

    import lightgbm as lgb

    bundle_dir = Path(os.environ.get("DBAHN_MODEL_DIR", ""))
    if not bundle_dir.name:
        candidates = sorted(Path("models").glob("*/metadata.json"))
        if not candidates:
            raise RuntimeError("no champion bundle found - set DBAHN_MODEL_DIR")
        bundle_dir = candidates[-1].parent
    version = json.loads((bundle_dir / "metadata.json").read_text())["version"]
    return (
        lgb.Booster(model_file=str(bundle_dir / "clf.txt")),
        lgb.Booster(model_file=str(bundle_dir / "q50.txt")),
        lgb.Booster(model_file=str(bundle_dir / "q90.txt")),
        version,
    )


def promotion_verdict(
    champion: dict[str, float], challenger: dict[str, float]
) -> tuple[bool, list[str]]:
    """All rules must hold; returns (promote, reasons-for-failure)."""
    failures = []
    if challenger["pr_auc"] < champion["pr_auc"] - PR_AUC_TOLERANCE:
        failures.append(
            f"pr_auc {challenger['pr_auc']:.4f} < champion {champion['pr_auc']:.4f} - tol"
        )
    if challenger["pinball_p90"] > champion["pinball_p90"] * PINBALL_P90_RATIO:
        failures.append(
            f"pinball_p90 {challenger['pinball_p90']:.4f} > champion "
            f"{champion['pinball_p90']:.4f} x {PINBALL_P90_RATIO}"
        )
    if challenger["ece"] > champion["ece"] + ECE_TOLERANCE:
        failures.append(f"ece {challenger['ece']:.4f} > champion {champion['ece']:.4f} + tol")
    return (not failures, failures)


def write_report(
    eval_month: str,
    champion_version: str,
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    promoted: bool,
    failures: list[str],
    bundle_path: Path | None,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    lines = [
        f"# Retraining report — {ts}",
        "",
        f"- Evaluation month: **{eval_month}** (unseen by both models)",
        f"- Champion: `{champion_version}`",
        f"- Verdict: **{'PROMOTE' if promoted else 'KEEP CHAMPION'}**",
    ]
    if failures:
        lines += ["- Challenger failed:"] + [f"  - {f}" for f in failures]
    lines += ["", "| metric | champion | challenger |", "|---|---|---|"]
    for key in sorted(set(champion_metrics) | set(challenger_metrics)):
        lines.append(
            f"| {key} | {champion_metrics.get(key, float('nan')):.4f} "
            f"| {challenger_metrics.get(key, float('nan')):.4f} |"
        )
    if bundle_path:
        lines += [
            "",
            "## Upload runbook",
            "```bash",
            f"scp -r {bundle_path} root@91.98.76.106:/opt/dbahn/models/",
            f"ssh root@91.98.76.106 'ln -sfn /opt/dbahn/models/{bundle_path.name} "
            "/opt/dbahn/models/current'",
            "# then: Coolify -> Restart (loads the new bundle)",
            "```",
        ]
    path = REPORTS_DIR / f"{ts}.md"
    path.write_text("\n".join(lines) + "\n")
    (REPORTS_DIR / f"{ts}.json").write_text(
        json.dumps(
            {
                "eval_month": eval_month,
                "champion": champion_metrics,
                "challenger": challenger_metrics,
                "promoted": promoted,
                "failures": failures,
            },
            indent=2,
        )
    )
    return path


def main() -> None:
    force = "--force" in sys.argv
    fresh = new_months()
    if fresh:
        logger.info("new months on HF: %s - rebuilding data", ", ".join(fresh))
        rebuild_data()
    elif force:
        logger.warning("no new months; --force set, continuing with existing data")
    else:
        logger.info("no new months on HF - nothing to do (use --force to test the flow)")
        return

    eval_month = local_months()[-1]
    train_months = walk_forward_folds([eval_month], 12)[0].train_months
    logger.info(
        "championship: train %s..%s, evaluate on %s", train_months[0], train_months[-1], eval_month
    )

    val = load_split([eval_month])
    clf_c, q50_c, q90_c, champion_version = champion_boosters()
    champion_metrics = score_boosters(clf_c, q50_c, q90_c, val)
    logger.info("champion (%s): %s", champion_version, champion_metrics)

    challenger = fit_models_from_split(train_months)
    challenger_metrics = score_boosters(
        challenger["clf"], challenger["q50"], challenger["q90"], val
    )
    logger.info("challenger: %s", challenger_metrics)

    promoted, failures = promotion_verdict(champion_metrics, challenger_metrics)
    bundle_path = None
    if promoted:
        freshest = month_range(train_months[1], eval_month)
        logger.info("PROMOTED - final refit on %s..%s", freshest[0], freshest[-1])
        final_models = fit_models_from_split(freshest)
        bundle_path = export_bundle(final_models, freshest)
    report = write_report(
        eval_month,
        champion_version,
        champion_metrics,
        challenger_metrics,
        promoted,
        failures,
        bundle_path,
    )
    logger.info("report: %s | verdict: %s", report, "PROMOTE" if promoted else "KEEP CHAMPION")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
