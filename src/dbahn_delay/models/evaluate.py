"""Evaluation metrics: honest, baseline-relative, calibration-aware.

Classification: ROC-AUC, PR-AUC, Brier score, expected calibration error.
Quantile regression: pinball loss at 0.5/0.9, MAE of p50, empirical p90
coverage (share of actuals at or below the predicted p90 — target 0.90).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never require a display
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FloatArray = npt.NDArray[np.float64]


def pinball_loss(y_true: FloatArray, y_pred: FloatArray, alpha: float) -> float:
    """Quantile (pinball) loss — the metric quantile regression optimizes."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def coverage(y_true: FloatArray, y_pred_quantile: FloatArray) -> float:
    """Share of actuals <= predicted quantile (target: the quantile level)."""
    return float(np.mean(y_true <= y_pred_quantile))


def expected_calibration_error(y_true: FloatArray, y_prob: FloatArray, n_bins: int = 10) -> float:
    """Weighted mean |predicted probability - observed frequency| over bins."""
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def classification_metrics(y_true: FloatArray, y_prob: FloatArray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "base_rate": float(y_true.mean()),
    }


def quantile_metrics(y_true: FloatArray, p50: FloatArray, p90: FloatArray) -> dict[str, float]:
    return {
        "pinball_p50": pinball_loss(y_true, p50, 0.5),
        "pinball_p90": pinball_loss(y_true, p90, 0.9),
        "mae_p50": float(np.mean(np.abs(y_true - p50))),
        "coverage_p90": coverage(y_true, p90),
    }


def reliability_plot(y_true: FloatArray, y_prob: FloatArray, path: Path, n_bins: int = 10) -> None:
    """Calibration curve: does a predicted 70% mean an observed ~70%?"""
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    xs, ys, sizes = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            xs.append(y_prob[mask].mean())
            ys.append(y_true[mask].mean())
            sizes.append(mask.mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(xs, ys, marker="o", color="#1f77b4", label="model")
    ax.set_xlabel("predicted probability of delay >= 6 min")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
