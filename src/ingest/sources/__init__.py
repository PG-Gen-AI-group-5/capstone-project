"""Registry of collectors. Each one knows how to plan, download and parse one data source."""
from __future__ import annotations

from src.ingest.sources.base import Source
from src.ingest.sources.fred import Fred
from src.ingest.sources.gdelt import GdeltRaw
from src.ingest.sources.nse_announcements import NseAnnouncements, NsePdfs
from src.ingest.sources.nse_corporate import (
    NseBoardMeetings,
    NseCorporateActions,
    NseInsider,
    NsePledges,
    NseResults,
    NseShareholding,
)
from src.ingest.sources.nse_daily import NseBhavcopy, NseFiiDii, NseIndices
from src.ingest.sources.rbi import RbiPolicy

ALL: list[Source] = [
    NseBhavcopy(),
    NseIndices(),
    NseCorporateActions(),
    NseResults(),
    NseBoardMeetings(),
    NseAnnouncements(),
    NsePdfs(),
    NseInsider(),
    NsePledges(),
    NseShareholding(),
    NseFiiDii(),
    Fred(),
    RbiPolicy(),
    GdeltRaw(),
]
SOURCES: dict[str, Source] = {s.name: s for s in ALL}


def get(name: str) -> Source:
    if name not in SOURCES:
        raise SystemExit(f"Unknown source {name!r}. Known: {', '.join(SOURCES)}")
    return SOURCES[name]
