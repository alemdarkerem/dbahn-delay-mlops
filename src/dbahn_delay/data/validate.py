"""Data quality validation for the monthly DB delay files.

Hand-rolled checks over polars LazyFrames producing a structured report.
Acts as the gate between raw downloads and the ingest pipeline: garbage stops
here, not in the model.

Usage: ``python -m dbahn_delay.data.validate`` validates all raw monthly files.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from dbahn_delay.config import settings

logger = logging.getLogger(__name__)

# Schema of the raw monthly files (documented on the HF dataset card).
EXPECTED_SCHEMA: dict[str, pl.DataType] = {
    "station_name": pl.String(),
    "xml_station_name": pl.String(),
    "eva": pl.String(),
    "train_name": pl.String(),
    "final_destination_station": pl.String(),
    "delay_in_min": pl.Int64(),
    "time": pl.Datetime("us"),
    "is_canceled": pl.Boolean(),
    "train_type": pl.String(),
    "train_line_ride_id": pl.String(),
    "train_line_station_num": pl.Int64(),
    "arrival_planned_time": pl.Datetime("us"),
    "arrival_change_time": pl.Datetime("us"),
    "departure_planned_time": pl.Datetime("us"),
    "departure_change_time": pl.Datetime("us"),
    "id": pl.String(),
}

# Columns that must never be null for a row to be usable.
REQUIRED_COLUMNS = ("station_name", "train_type", "time", "id")

# Sanity bounds for delays in minutes. Delays outside this window are treated
# as data errors, not real operations (tuned after EDA).
DELAY_MIN = -60
DELAY_MAX = 1_440  # 24 hours


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class ValidationReport:
    source: str
    results: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = [f"Validation report for {self.source}:"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.name}: {r.details}")
        return "\n".join(lines)


def check_schema(lf: pl.LazyFrame) -> CheckResult:
    """All expected columns must exist with the expected dtypes."""
    actual = dict(lf.collect_schema())
    problems = []
    for col, dtype in EXPECTED_SCHEMA.items():
        if col not in actual:
            problems.append(f"missing column {col!r}")
        elif actual[col] != dtype:
            problems.append(f"{col}: expected {dtype}, got {actual[col]}")
    return CheckResult(
        name="schema",
        passed=not problems,
        details="; ".join(problems) or f"{len(EXPECTED_SCHEMA)} columns OK",
    )


def check_required_not_null(lf: pl.LazyFrame, max_null_rate: float = 0.0) -> CheckResult:
    """Key columns must be present in (almost) every row."""
    height = lf.select(pl.len()).collect().item()
    if height == 0:
        return CheckResult("required_not_null", False, "dataset is empty")
    null_rates = (
        lf.select([pl.col(c).null_count().alias(c) for c in REQUIRED_COLUMNS]).collect().row(0)
    )
    problems = [
        f"{col}: {count / height:.2%} null"
        for col, count in zip(REQUIRED_COLUMNS, null_rates, strict=True)
        if count / height > max_null_rate
    ]
    return CheckResult(
        name="required_not_null",
        passed=not problems,
        details="; ".join(problems) or f"{len(REQUIRED_COLUMNS)} required columns OK",
    )


def check_delay_range(lf: pl.LazyFrame) -> CheckResult:
    """Delays must fall inside plausible operational bounds."""
    out_of_range = (
        lf.filter(pl.col("delay_in_min").is_between(DELAY_MIN, DELAY_MAX).not_())
        .select(pl.len())
        .collect()
        .item()
    )
    return CheckResult(
        name="delay_range",
        passed=out_of_range == 0,
        details=f"{out_of_range} rows outside [{DELAY_MIN}, {DELAY_MAX}] min",
    )


def check_duplicate_ids(lf: pl.LazyFrame) -> CheckResult:
    """The stop id should uniquely identify a row."""
    counts = lf.select(
        total=pl.len(),
        unique=pl.col("id").n_unique(),
    ).collect()
    total, unique = counts.row(0)
    dupes = total - unique
    return CheckResult(
        name="duplicate_ids",
        passed=dupes == 0,
        details=f"{dupes} duplicated ids out of {total} rows",
    )


def check_time_range(lf: pl.LazyFrame, start: datetime, end: datetime) -> CheckResult:
    """Event times must fall inside the window the file claims to cover."""
    outside = (
        lf.filter(pl.col("time").is_between(start, end).not_()).select(pl.len()).collect().item()
    )
    return CheckResult(
        name="time_range",
        passed=outside == 0,
        details=f"{outside} rows outside [{start:%Y-%m-%d}, {end:%Y-%m-%d})",
    )


def validate(lf: pl.LazyFrame, source: str, *, start: datetime, end: datetime) -> ValidationReport:
    """Run all checks against one dataset and collect a report."""
    results = [
        check_schema(lf),
        check_required_not_null(lf),
        check_delay_range(lf),
        check_duplicate_ids(lf),
        check_time_range(lf, start, end),
    ]
    return ValidationReport(source=source, results=results)


def validate_raw_files() -> list[ValidationReport]:
    """Validate every downloaded monthly file; report per file."""
    reports = []
    for path in sorted(settings.monthly_raw_dir.glob("*.parquet")):
        # File name convention: data-YYYY-MM.parquet
        year, month = map(int, path.stem.removeprefix("data-").split("-"))
        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
        report = validate(pl.scan_parquet(path), path.name, start=start, end=end)
        logger.info("%s", report.summary())
        reports.append(report)
    return reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    all_reports = validate_raw_files()
    failed = [r for r in all_reports if not r.passed]
    print(f"\n{len(all_reports) - len(failed)}/{len(all_reports)} files passed validation")
