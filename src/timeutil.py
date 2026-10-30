"""Time helpers. Everything is stored in UTC and shown in IST.

IST has no daylight saving, so a fixed +05:30 offset is exact (and avoids needing
the tz database on Windows).
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

_DATE_FORMATS = ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d/%m/%Y")
_DATETIME_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                     "%d %b %Y %H:%M:%S")


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def today_ist() -> date:
    return datetime.now(IST).date()


def iso(dt: datetime | None) -> str | None:
    """UTC ISO string used in the tracker database, e.g. 2026-10-30T11:00:00Z."""
    if dt is None:
        return None
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def at_ist(d: date, hhmm: str) -> datetime:
    """A wall-clock time in IST on date d, returned in UTC."""
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(d, time(h, m), IST).astimezone(UTC)


def parse_date(value) -> date | None:
    """Parse the date formats NSE uses: 09-Jun-2016, 04-JAN-2016, 04-01-2016, 2025-10-01."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s in ("-", "NA", "None", "nan"):
        return None
    s = s.title() if s[3:6].isalpha() else s  # 04-JAN-2016 -> 04-Jan-2016
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_ist_datetime(value) -> datetime | None:
    """Parse an NSE IST timestamp (e.g. '23-Jan-2018 14:30:00') and return it in UTC.

    A date without a time is treated as end of day (23:59 IST), the conservative choice:
    the record is then usable only from the next day.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("-", "NA", "None", "nan"):
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=IST).astimezone(UTC)
        except ValueError:
            continue
    d = parse_date(s)
    return at_ist(d, "23:59") if d else None


def days(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
