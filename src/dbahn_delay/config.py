"""Central configuration — the single source of truth for paths and data scope.

Every module reads settings from here. Values can be overridden via environment
variables or a local .env file (e.g. ``DBAHN_DATA_DIR=/mnt/data``).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project-wide settings, overridable via environment or .env."""

    model_config = SettingsConfigDict(env_prefix="DBAHN_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    # DB API Marketplace (Timetables) — credentials come from .env / environment
    # (aliased: the env var names have no DBAHN_ prefix, see .env.example)
    db_api_client_id: str = Field(default="", validation_alias="DB_API_CLIENT_ID")
    db_api_client_secret: str = Field(default="", validation_alias="DB_API_CLIENT_SECRET")
    db_api_base_url: str = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"

    # Live-loop storage (predictions, observed changes, daily metrics)
    live_dir: Path = PROJECT_ROOT / "data" / "live"

    # Where the live fetcher asks for predictions. Points at the API process
    # that ALREADY holds the model in memory — the fetcher must never load a
    # second copy of the bundle (that OOM-killed the 4GB VPS on day one).
    predict_url: str = "http://127.0.0.1:8000/predict"

    # Historical dataset (Hugging Face) — data by Deutsche Bahn, CC BY 4.0
    hf_dataset_repo: str = "piebro/deutsche-bahn-data"
    hf_monthly_pattern: str = "monthly_processed_data/*.parquet"
    first_month: str = "2024-07"
    last_month: str = "2026-06"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def monthly_raw_dir(self) -> Path:
        return self.raw_dir / "monthly_processed_data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def stops_path(self) -> Path:
        """Canonical cleaned dataset produced by the ingest pipeline."""
        return self.processed_dir / "stops.parquet"


settings = Settings()
