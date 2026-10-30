"""NSE corporate announcements (list, per stock per year) and the PDFs of the important ones."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from src.ingest.errors import IngestError
from src.ingest.http import write_atomic
from src.ingest.sources.base import DownloadResult, Frame, Issue, ParseResult, raw_path
from src.ingest.sources.nse_corporate import API, NseApiSource, require, row_hash
from src.ingest.tracker import TaskSpec
from src.timeutil import IST, parse_ist_datetime

ANNOUNCEMENT_SCHEMA = {
    "doc_id": "str", "symbol": "str", "category": "str", "summary": "str", "attachment_url": "str",
    "attachment_ext": "str", "published_at": "ts", "has_xbrl": "str", "company": "str", "isin": "str",
    "industry": "str", "is_important": "bool", "source": "str",
}


def is_important(category: str | None, summary: str | None, cfg: dict) -> bool:
    """Important if the category is on the list; for generic categories ('Updates'), if the summary
    contains one of the keywords."""
    cat = (category or "").strip().lower()
    if any(k in cat for k in cfg["important_categories"]):
        return True
    if cat in cfg["generic_categories"]:
        text = (summary or "").lower()
        return any(re.search(rf"\b{re.escape(k)}(s|es|ed|ing)?\b", text) for k in cfg["important_keywords"])
    return False


class NseAnnouncements(NseApiSource):
    name = "nse_announcements"
    task_unit = "symbol_year"
    description = "NSE corporate announcements list, per stock per year"
    folder = "announcements"
    per_year = True

    def url(self, symbol, start, end):
        return (f"{API}/corporate-announcements?index=equities&symbol={quote(symbol, safe='')}"
                f"&from_date={start:%d-%m-%Y}&to_date={end:%d-%m-%Y}")

    def parse(self, ctx, task, raw):
        df = self.records(raw)
        require(df, {"desc", "attchmntText"}, "announcements")
        published = df.get("exchdisstime", pd.Series(index=df.index, dtype=object)).map(parse_ist_datetime)
        for fallback in ("an_dt", "sort_date", "dt"):
            if fallback in df:
                published = published.fillna(df[fallback].map(parse_ist_datetime))
        records = df.to_dict("records")
        doc_ids = [f"nse:{r['seq_id']}" if r.get("seq_id") else f"nse:h{row_hash(r)}" for r in records]
        urls = df.get("attchmntFile", pd.Series(index=df.index, dtype=object))
        cfg = ctx.cfg["announcements"]
        out = pd.DataFrame({
            "doc_id": doc_ids,
            "symbol": task.symbol,
            "category": df["desc"],
            "summary": df["attchmntText"],
            "attachment_url": urls,
            "attachment_ext": urls.map(lambda u: Path(str(u)).suffix.lower().lstrip(".") if u else None),
            "published_at": published,
            "has_xbrl": df.get("hasXbrl"),
            "company": df.get("sm_name"),
            "isin": df.get("sm_isin"),
            "industry": df.get("smIndustry"),
            "is_important": [is_important(c, s, cfg) for c, s in zip(df["desc"], df["attchmntText"])],
            "source": "nse_announcements",
        })
        issues = []
        no_time = int(out["published_at"].isna().sum())
        if no_time:
            issues.append(Issue("missing_timestamp", "warn", {"rows": no_time}))
        return ParseResult([Frame("announcements", out, ["doc_id"], "symbol", ANNOUNCEMENT_SCHEMA)], issues)


# ---------------------------------------------------------------- PDFs

TEXT_SCHEMA = {"doc_id": "str", "symbol": "str", "n_pages": "int", "n_chars": "int", "scanned": "bool",
               "text": "str", "source": "str", "part": "str"}


def _safe_name(doc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)


class NsePdfs(NseApiSource):
    name = "nse_pdfs"
    task_unit = "document"
    description = "PDF attachments of important announcements (agent window only), with extracted text"
    start_key = "pdfs"

    def plan(self, ctx, start, end, symbols):
        anns = ctx.lake.read("announcements", ["doc_id", "symbol", "attachment_url", "attachment_ext",
                                               "published_at", "is_important"])
        if anns.empty:
            print("  No announcements in the lake yet: run nse_announcements first.")
            return []
        when = anns["published_at"].dt.tz_convert(IST).dt.date
        keep = (anns["is_important"].fillna(False).astype(bool)
                & anns["attachment_ext"].isin(["pdf", "zip"])
                & (when >= start) & (when <= end))
        if symbols:
            keep &= anns["symbol"].isin(symbols)
        rows = anns[keep].assign(day=when[keep])
        return [TaskSpec(task_key=r.doc_id, symbol=r.symbol, period_start=r.day, period_end=r.day,
                         url=r.attachment_url) for r in rows.itertuples()]

    def download(self, ctx, task):
        path = raw_path(ctx, "nse_pdfs", task.symbol, f"{_safe_name(task.task_key)}.pdf")
        if path.exists() and path.stat().st_size > 0:
            return DownloadResult(path, nbytes=path.stat().st_size)
        data = ctx.http.get_bytes(task.url)
        if data[:4] != b"%PDF":
            if data[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    pdfs = [n for n in z.namelist() if n.lower().endswith(".pdf")]
                    if not pdfs:
                        raise IngestError("unsupported_file", f"Zip without a PDF: {z.namelist()[:5]}")
                    data = z.read(pdfs[0])
            elif data.lstrip()[:1] == b"<":
                raise IngestError("blocked_html", f"{task.url}: got an HTML page instead of a PDF")
            else:
                raise IngestError("unsupported_file", f"{task.url}: not a PDF (starts with {data[:8]!r})")
        write_atomic(path, data)
        return DownloadResult(path, nbytes=len(data))

    def parse(self, ctx, task, raw):
        import pdfplumber

        with pdfplumber.open(raw) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        n_pages, n_chars = len(pages), len(text)
        scanned = n_chars < ctx.cfg["pdfs"]["scanned_chars_per_page"] * max(n_pages, 1)
        out = pd.DataFrame([{
            "doc_id": task.task_key, "symbol": task.symbol, "n_pages": n_pages, "n_chars": n_chars,
            "scanned": scanned, "text": text, "source": "nse_pdfs",
            "part": f"{task.symbol}_{task.start_date.year}",
        }])
        issues = [Issue("scanned_pdf", "info", {"pages": n_pages, "chars": n_chars})] if scanned else []
        return ParseResult([Frame("announcement_text", out, ["doc_id"], "part", TEXT_SCHEMA)], issues)
