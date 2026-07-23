"""Download monthly parquet files from the Hugging Face dataset.

Idempotent: already-downloaded files are skipped, interrupted downloads resume.
Usage: ``python -m dbahn_delay.data.download`` (or ``make data``).

Data by Deutsche Bahn via ``piebro/deutsche-bahn-data``, licensed CC BY 4.0.
"""

import logging

from huggingface_hub import snapshot_download

from dbahn_delay.config import settings

logger = logging.getLogger(__name__)


def download_monthly_data() -> None:
    """Fetch all monthly processed parquet files into the raw data directory."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading %s (%s) into %s",
        settings.hf_dataset_repo,
        settings.hf_monthly_pattern,
        settings.raw_dir,
    )
    snapshot_download(
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        allow_patterns=settings.hf_monthly_pattern,
        local_dir=settings.raw_dir,
    )
    files = sorted(settings.monthly_raw_dir.glob("*.parquet"))
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    logger.info("Done: %d monthly files, %.1f GB total", len(files), total_gb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_monthly_data()
