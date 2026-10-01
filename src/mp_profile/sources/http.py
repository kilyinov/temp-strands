"""Polite HTTP client with an on-disk raw cache so re-ingestion and parser changes don't re-fetch."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import httpx


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
        min_interval: float = 0.5,
        retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self._client = httpx.Client(headers={"User-Agent": user_agent}, timeout=timeout, follow_redirects=True)
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / hashlib.sha256(url.encode()).hexdigest()[:32]

    def get_bytes(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> bytes:
        full = str(httpx.URL(url, params=params)) if params else url
        path = self._cache_path(full)
        if use_cache and path and path.exists():
            return path.read_bytes()
        last_error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                response = self._client.get(full)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
                response.raise_for_status()
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(response.content)
                return response.content
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def get_text(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> str:
        return self.get_bytes(url, params, use_cache).decode("utf-8", errors="replace")

    def close(self) -> None:
        self._client.close()
