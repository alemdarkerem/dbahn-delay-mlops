"""API tests against the committed fixture bundle (CI-safe, no real data)."""

import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE_BUNDLE = Path(__file__).parent / "fixtures" / "bundle"
GOLDEN_PATH = Path(__file__).parent / "golden_predictions.json"


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["DBAHN_MODEL_DIR"] = str(FIXTURE_BUNDLE)
    import dbahn_delay.serving.app as app_module

    importlib.reload(app_module)  # re-run bundle loading with the fixture env
    return TestClient(app_module.app)


def base_request() -> dict[str, object]:
    return {
        "station_name": "Berlin Hbf",
        "train_type": "ICE",
        "train_number": "1601",
        "scheduled_time": "2026-07-02T17:30:00",
        "train_line_station_num": 3,
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "fixture-0"


def test_model_info(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["metadata"]["version"] == "fixture-0"
    assert body["snapshot_entities"]["train"] == 9


def test_predict_happy_path(client: TestClient) -> None:
    response = client.post("/predict", json=base_request())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["delay_p90_min"] >= body["delay_p50_min"] >= 0.0
    assert body["coverage"] == "train"
    assert body["model_version"] == "fixture-0"


def test_predict_validation_errors(client: TestClient) -> None:
    bad = base_request() | {"station_name": ""}
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in base_request().items() if k != "train_type"}
    assert client.post("/predict", json=missing).status_code == 422


def test_coverage_fallback_chain(client: TestClient) -> None:
    # Unknown train number -> falls back to station_type stats
    r1 = client.post("/predict", json=base_request() | {"train_number": "9999"}).json()
    assert r1["coverage"] == "station_type"
    # Unknown station too -> type stats
    r2 = client.post(
        "/predict", json=base_request() | {"train_number": "9999", "station_name": "Nirgendwo"}
    ).json()
    assert r2["coverage"] == "type"
    # Unknown train type as well -> cold (still answers, never crashes)
    r3 = client.post(
        "/predict",
        json=base_request()
        | {"train_number": "9999", "station_name": "Nirgendwo", "train_type": "XX"},
    ).json()
    assert r3["coverage"] == "cold"


def test_stale_snapshot_downgrades_coverage(client: TestClient) -> None:
    # Fixture snapshot is dated 2026-07-01; querying far in the future makes
    # every granularity stale -> cold.
    late = base_request() | {"scheduled_time": "2026-12-24T09:00:00"}
    assert client.post("/predict", json=late).json()["coverage"] == "cold"


def test_golden_predictions(client: TestClient) -> None:
    """Pin exact outputs: any silent change in feature assembly must fail here.

    Regenerate (only after intentional changes) with:
    uv run python tests/fixtures/make_fixture_bundle.py && \
    uv run python -m tests.regenerate_golden
    """
    golden = json.loads(GOLDEN_PATH.read_text())
    for case in golden:
        response = client.post("/predict", json=case["request"])
        assert response.status_code == 200
        assert response.json() == case["expected"], f"golden mismatch for {case['name']}"
