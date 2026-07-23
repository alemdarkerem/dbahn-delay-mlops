"""Thin client for the DB Timetables API.

Free tier allows 60 requests/minute, so every call is paced. Individual
request failures raise; callers decide whether to skip a station or abort —
the hourly loop must survive one flaky station.
"""

import logging
import time

import httpx

from dbahn_delay.config import settings

logger = logging.getLogger(__name__)

# Stay safely under the 60 req/min free-tier limit.
MIN_SECONDS_BETWEEN_REQUESTS = 1.1
TIMEOUT_SECONDS = 15.0


class TimetablesClient:
    def __init__(self) -> None:
        if not settings.db_api_client_id or not settings.db_api_client_secret:
            raise RuntimeError(
                "DB API credentials missing - set DB_API_CLIENT_ID and "
                "DB_API_CLIENT_SECRET (see .env.example)"
            )
        self._http = httpx.Client(
            base_url=settings.db_api_base_url,
            headers={
                "DB-Client-Id": settings.db_api_client_id,
                "DB-Api-Key": settings.db_api_client_secret,
                "accept": "application/xml",
            },
            timeout=TIMEOUT_SECONDS,
        )
        self._last_request_at = 0.0

    def _get(self, path: str) -> str:
        """Rate-limited GET with one retry on transient failures."""
        for attempt in (1, 2):
            wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()
            try:
                response = self._http.get(path)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.text
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if attempt == 2 or (
                    isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500
                ):
                    raise
                logger.warning("retrying %s after: %s", path, exc)
        raise AssertionError("unreachable")

    def search_station(self, name: str) -> str:
        # quote everything incl. "/" (Köln Messe/Deutz would break the path)
        from urllib.parse import quote

        return self._get(f"/station/{quote(name, safe='')}")

    def fetch_plan(self, eva: str, date_yymmdd: str, hour_hh: str) -> str:
        return self._get(f"/plan/{eva}/{date_yymmdd}/{hour_hh}")

    def fetch_changes(self, eva: str) -> str:
        return self._get(f"/fchg/{eva}")

    def close(self) -> None:
        self._http.close()
