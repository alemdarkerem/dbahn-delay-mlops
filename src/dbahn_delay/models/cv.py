"""Walk-forward cross-validation over calendar months.

Never random splits: each fold trains on a rolling window of past months and
validates on the single month right after it — exactly how the production
model will be retrained and then face the future.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_months: list[str]  # e.g. ["2025-01", ..., "2025-12"]
    val_month: str  # e.g. "2026-01"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of YYYY-MM strings from start to end."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def walk_forward_folds(
    val_months: list[str],
    train_window_months: int = 12,
) -> list[Fold]:
    """One fold per validation month, trained on the preceding rolling window."""
    folds = []
    for val in val_months:
        y, m = map(int, val.split("-"))
        # last train month = month before val
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        end = f"{y:04d}-{m:02d}"
        # first train month = window start
        m -= train_window_months - 1
        while m <= 0:
            y, m = y - 1, m + 12
        start = f"{y:04d}-{m:02d}"
        folds.append(Fold(train_months=month_range(start, end), val_month=val))
    return folds
