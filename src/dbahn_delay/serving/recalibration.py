"""Serving-side recalibration: apply the daily drift correction to outputs.

Loads the JSON written by ``live/recalibrate.py`` (mtime reload, overlay
pattern). Missing or unreadable file => identity, so local dev, tests and
cold starts behave exactly like the uncorrected model.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RecalibrationStore:
    def __init__(self, path: Path, model_version: str | None = None) -> None:
        self._path = path
        self._model_version = model_version
        self._mtime: float | None = None
        self._calibration: dict[str, Any] | None = None

    def _reload_if_changed(self) -> None:
        if not self._path.exists():
            self._mtime = None
            self._calibration = None
            return
        mtime = self._path.stat().st_mtime
        if mtime == self._mtime:
            return
        try:
            calibration = json.loads(self._path.read_text())
            assert calibration["curve_x"] and len(calibration["curve_x"]) == len(
                calibration["curve_y"]
            )
            fitted_for = calibration.get("model_version")
            if self._model_version and fitted_for and fitted_for != self._model_version:
                # Curve belongs to another model (just-promoted bundle):
                # serve raw outputs until the nightly job refits.
                logger.warning(
                    "calibration fitted for %s, serving %s - ignoring",
                    fitted_for,
                    self._model_version,
                )
                self._calibration = None
                self._mtime = mtime
                return
            self._calibration = calibration
            logger.info(
                "recalibration loaded: created_at=%s n=%s p90_delta=%s",
                calibration.get("created_at"),
                calibration.get("n_samples"),
                calibration.get("p90_delta_min"),
            )
        except Exception:
            logger.exception("unreadable calibration file - serving uncorrected outputs")
            self._calibration = None
        self._mtime = mtime

    def apply(self, prob: float, p50: float, p90: float) -> tuple[float, float, float]:
        """Corrected (probability, p50, p90); identity without a calibration."""
        self._reload_if_changed()
        c = self._calibration
        if c is None:
            return prob, p50, p90
        prob = float(np.interp(prob, c["curve_x"], c["curve_y"]))
        p90 = max(p50, p90 + float(c["p90_delta_min"]))
        return prob, p50, p90

    def info(self) -> dict[str, Any] | None:
        """Metadata for /model-info (honesty: corrections are visible)."""
        self._reload_if_changed()
        if self._calibration is None:
            return None
        return {k: v for k, v in self._calibration.items() if not k.startswith("curve_")}
