"""Build the tiny committed test bundle used by API and golden tests.

Deterministic (seeded): trains micro LightGBM boosters on synthetic data with
the real feature schema and writes a complete bundle (boosters + metadata +
snapshot tables). Re-run only when the feature schema changes, then commit
the regenerated bundle AND update the golden expectations:

    uv run python tests/fixtures/make_fixture_bundle.py
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from dbahn_delay.features.build import GRANULARITIES
from dbahn_delay.models.train import CATEGORICAL, FEATURES, NUMERIC
from dbahn_delay.serving.loader import STAT_COLUMNS

BUNDLE_DIR = Path(__file__).parent / "bundle"
SEED = 42

STATIONS = ["Berlin Hbf", "München Hbf", "Köln Hbf"]
TYPES = ["ICE", "RE", "S"]
NUMBERS = ["1601", "5", "7"]
SNAPSHOT_DATE = dt.date(2026, 7, 1)


def synthetic_frame(n: int, rng: np.random.Generator) -> pl.DataFrame:
    """Synthetic rows with a learnable pattern: ICE + rush hour = delayed."""
    df = pl.DataFrame(
        {
            "station_name": rng.choice(STATIONS, n),
            "train_type": rng.choice(TYPES, n),
            "train_number": rng.choice(NUMBERS, n),
            "train_line_station_num": rng.integers(0, 20, n),
            "scheduled_hour": rng.integers(0, 24, n),
            "weekday": rng.integers(1, 8, n),
            "month": rng.integers(1, 13, n),
            "is_weekend": rng.random(n) < 0.3,
            "is_holiday": rng.random(n) < 0.05,
        }
    )
    for prefix in GRANULARITIES:
        df = df.with_columns(
            pl.Series(f"{prefix}_mean_delay_14d", rng.gamma(2, 2, n)),
            pl.Series(f"{prefix}_delayed_rate_14d", rng.random(n) * 0.5),
            pl.Series(f"{prefix}_mean_delay_30d", rng.gamma(2, 2, n)),
            pl.Series(f"{prefix}_delayed_rate_30d", rng.random(n) * 0.5),
            pl.Series(f"{prefix}_count_30d", rng.integers(10, 500, n).astype(np.float64)),
        )
    ice = (df["train_type"] == "ICE").to_numpy()
    rush = ((df["scheduled_hour"] > 15) & (df["scheduled_hour"] < 20)).to_numpy()
    rate = 0.08 + 0.3 * ice + 0.15 * rush + 0.3 * df["type_delayed_rate_30d"].to_numpy()
    delayed = rng.random(n) < rate
    delay_min = np.where(delayed, rng.gamma(2, 8, n) + 6, rng.gamma(1, 1.5, n)).round()
    return df.with_columns(
        pl.Series("target_delayed6", delayed),
        pl.Series("target_delay_min", delay_min),
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = synthetic_frame(5000, rng)

    pdf = (
        df.select(FEATURES)
        .with_columns(
            [pl.col(c).cast(pl.Categorical) for c in CATEGORICAL]
            + [pl.col(c).cast(pl.Float32) for c in NUMERIC]
        )
        .to_pandas()
    )
    params: dict[str, Any] = {
        "n_estimators": 20,
        "num_leaves": 7,
        "min_child_samples": 20,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": 1,
    }
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    clf = lgb.LGBMClassifier(objective="binary", **params)
    clf.fit(pdf, df["target_delayed6"].to_numpy(), categorical_feature=CATEGORICAL)
    clf.booster_.save_model(BUNDLE_DIR / "clf.txt")
    for alpha, name in ((0.5, "q50"), (0.9, "q90")):
        reg = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        reg.fit(pdf, df["target_delay_min"].to_numpy(), categorical_feature=CATEGORICAL)
        reg.booster_.save_model(BUNDLE_DIR / f"{name}.txt")

    # Snapshot tables: deterministic stats for every entity combination.
    for prefix, keys in GRANULARITIES.items():
        combos: list[tuple[str, ...]]
        if prefix == "train":
            combos = [(t, n) for t in TYPES for n in NUMBERS]
        elif prefix == "station_type":
            combos = [(s, t) for s in STATIONS for t in TYPES]
        else:
            combos = [(t,) for t in TYPES]
        rows = []
        for i, combo in enumerate(combos):
            row: dict[str, object] = dict(zip(keys, combo, strict=True))
            row["join_date"] = SNAPSHOT_DATE
            base = 2.0 + i * 0.5
            for col in STAT_COLUMNS[prefix]:
                row[col] = float(len(combos)) if col.endswith("count_30d") else base
            rows.append(row)
        pl.DataFrame(rows).write_parquet(BUNDLE_DIR / f"{prefix}_stats.parquet")

    meta = {
        "version": "fixture-0",
        "git_sha": "fixture",
        "created_at": "2026-07-01T00:00:00+00:00",
        "train_months": ["synthetic"],
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "lgb_params": {k: str(v) for k, v in params.items()},
    }
    (BUNDLE_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    size_kb = sum(f.stat().st_size for f in BUNDLE_DIR.iterdir()) / 1024
    print(f"fixture bundle written to {BUNDLE_DIR} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
