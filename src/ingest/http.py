"""HTTP client: browser-like headers, per-host rate limits, and errors mapped to tracker error types."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.ingest.errors import IngestError

NSE_HOME = "https://www.nseindia.com/"
NSE_HEADERS = {"Referer": NSE_HOME, "Accept": "application/json, text/plain, */*"}


class Http:
    def __init__(self, cfg):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": cfg["http"]["user_agent"],
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.timeout = cfg["http"]["timeout"]
        self.rate_limits: dict = cfg["rate_limits"]
        self._last_request: dict[str, float] = {}
        self._nse_warmed = False

    def _throttle(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        gap = float(self.rate_limits.get(host, self.rate_limits.get("default", 1.0)))
        wait = self._last_request.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def get(self, url: str, headers: dict | None = None, stream: bool = False,
            timeout: float | None = None) -> requests.Response:
        self._throttle(url)
        if headers is None and "nseindia.com" in url:
            headers = {"Referer": NSE_HOME}
        try:
            r = self.session.get(url, headers=headers, stream=stream, timeout=timeout or self.timeout)
        except requests.Timeout as e:
            raise IngestError("timeout", f"{url}: {e}") from e
        except requests.ConnectionError as e:
            raise IngestError("connection_error", f"{url}: {e}") from e
        if r.status_code == 200:
            return r
        code = r.status_code
        error_type = {403: "http_403", 404: "http_404", 429: "http_429"}.get(code)
        if error_type is None:
            error_type = "http_5xx" if code >= 500 else "http_other"
        raise IngestError(error_type, f"{url} returned HTTP {code}", code)

    def nse_json(self, url: str):
        """JSON from an NSE API endpoint. If NSE answers with its anti-bot HTML page, fetch the
        homepage once to get cookies and try again."""
        for attempt in range(2):
            r = self.get(url, headers=NSE_HEADERS)
            text = r.text.strip()
            if not text:
                raise IngestError("empty_response", url)
            if text.startswith("<"):
                if attempt == 0 and not self._nse_warmed:
                    self._warm_nse()
                    continue
                raise IngestError("blocked_html", f"{url} returned an HTML page instead of JSON")
            try:
                return r.json()
            except ValueError as e:
                raise IngestError("bad_response", f"{url}: invalid JSON") from e
        raise IngestError("blocked_html", url)

    def _warm_nse(self) -> None:
        self._nse_warmed = True
        try:
            self.get(NSE_HOME, headers={"Accept": "text/html"})
        except IngestError:
            pass  # the homepage itself is sometimes blocked while the API still works

    def json(self, url: str, params: dict | None = None):
        if params:
            url = requests.Request("GET", url, params=params).prepare().url
        r = self.get(url, headers={"Accept": "application/json"})
        try:
            return r.json()
        except ValueError as e:
            raise IngestError("bad_response", f"{url}: invalid JSON") from e

    def get_bytes(self, url: str, kind: str | None = None) -> bytes:
        """Raw bytes. kind='zip' or 'csv' adds a check that we didn't get an HTML page instead."""
        data = self.get(url).content
        if not data:
            raise IngestError("empty_response", url)
        looks_html = data.lstrip()[:1] == b"<"
        if kind == "zip" and not data.startswith(b"PK"):
            raise IngestError("blocked_html" if looks_html else "bad_response", f"{url}: not a zip file")
        if kind == "csv" and looks_html:
            raise IngestError("blocked_html", f"{url}: got an HTML page instead of CSV")
        return data

    def download_to(self, url: str, path: Path, expected_bytes: int | None = None,
                    expected_md5: str | None = None) -> tuple[int, str]:
        """Stream a file to disk, checking size and MD5 when they are known."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        md5, size = hashlib.md5(), 0
        r = self.get(url, stream=True, timeout=max(self.timeout, 120))
        try:
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    md5.update(chunk)
                    size += len(chunk)
        except requests.RequestException as e:
            tmp.unlink(missing_ok=True)
            raise IngestError("connection_error", f"{url}: download interrupted: {e}") from e
        digest = md5.hexdigest()
        if (expected_bytes and size != expected_bytes) or (expected_md5 and digest != expected_md5):
            tmp.unlink(missing_ok=True)
            raise IngestError("checksum_mismatch",
                              f"{url}: got {size} bytes / md5 {digest}, expected {expected_bytes} / {expected_md5}")
        os.replace(tmp, path)
        return size, digest


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
