"""Error categories. Each one decides whether the tracker retries automatically or asks a person."""
from __future__ import annotations

# error_type -> (retryable, description)
ERROR_TYPES: dict[str, tuple[bool, str]] = {
    "timeout": (True, "Request timed out"),
    "connection_error": (True, "Network connection failed"),
    "http_403": (True, "Forbidden: often a temporary block"),
    "http_429": (True, "Rate limited"),
    "http_5xx": (True, "Server error"),
    "blocked_html": (True, "Got an HTML page instead of data (anti-bot page)"),
    "bad_response": (True, "Response was not valid JSON / zip / CSV"),
    "empty_response": (True, "Response body was empty"),
    "checksum_mismatch": (True, "Downloaded file does not match the expected size or MD5"),
    "not_yet_available": (True, "File not published yet (recent date)"),
    "http_404": (False, "Not found on a day it should exist"),
    "http_other": (False, "Unexpected HTTP status"),
    "unsupported_file": (False, "Attachment is not a PDF or zip"),
    "parse_error": (False, "Our parser failed on this file"),
    "schema_changed": (False, "Columns or fields are not what the parser expects"),
    "validation_failed": (False, "A blocking data-quality check failed"),
    "config_missing": (False, "A required setting or API key is missing"),
    "unexpected_error": (False, "Unexpected error in collector code"),
}


class IngestError(Exception):
    def __init__(self, error_type: str, message: str, http_status: int | None = None):
        if error_type not in ERROR_TYPES:
            raise ValueError(f"Unknown error type: {error_type}")
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message
        self.http_status = http_status

    @property
    def retryable(self) -> bool:
        return ERROR_TYPES[self.error_type][0]
