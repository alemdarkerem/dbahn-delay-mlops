"""Regenerate golden predictions from the fixture bundle.

Run ONLY after an intentional feature/model-schema change, review the diff,
then commit: ``uv run python -m tests.regenerate_golden``
"""

import json
import os
from pathlib import Path

FIXTURE_BUNDLE = Path(__file__).parent / "fixtures" / "bundle"
GOLDEN_PATH = Path(__file__).parent / "golden_predictions.json"

CASES: list[dict[str, object]] = [
    {
        "name": "ice_rush_hour_known_train",
        "request": {
            "station_name": "Berlin Hbf",
            "train_type": "ICE",
            "train_number": "1601",
            "scheduled_time": "2026-07-02T17:30:00",
            "train_line_station_num": 3,
        },
    },
    {
        "name": "sbahn_morning",
        "request": {
            "station_name": "München Hbf",
            "train_type": "S",
            "train_number": "7",
            "scheduled_time": "2026-07-03T08:00:00",
        },
    },
    {
        "name": "unknown_station_fallback",
        "request": {
            "station_name": "Nirgendwo Hbf",
            "train_type": "RE",
            "train_number": "5",
            "scheduled_time": "2026-07-04T12:15:00",
            "train_line_station_num": 10,
        },
    },
]


def main() -> None:
    os.environ["DBAHN_MODEL_DIR"] = str(FIXTURE_BUNDLE)
    import importlib

    from fastapi.testclient import TestClient

    import dbahn_delay.serving.app as app_module

    importlib.reload(app_module)
    client = TestClient(app_module.app)
    for case in CASES:
        response = client.post("/predict", json=case["request"])
        response.raise_for_status()
        case["expected"] = response.json()
    GOLDEN_PATH.write_text(json.dumps(CASES, indent=2) + "\n")
    print(f"wrote {GOLDEN_PATH} with {len(CASES)} cases")


if __name__ == "__main__":
    main()
