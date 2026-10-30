"""Per-stock NSE data: corporate actions, results filings, board meetings, insider trades,
promoter pledging and shareholding."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd

from src.ingest.errors import IngestError
from src.ingest.sources.base import (
    DownloadResult,
    Frame,
    Issue,
    ParseResult,
    Source,
    as_records,
    load_json,
    nse_date,
    raw_path,
    save_json,
    years,
)
from src.ingest.tracker import TaskSpec
from src.timeutil import parse_date, parse_ist_datetime

API = "https://www.nseindia.com/api"


def row_hash(record: dict) -> str:
    return hashlib.sha1(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:16]


class NseApiSource(Source):
    """Common pattern: one NSE API call per stock (or per stock and year), saved as raw JSON."""
    folder = ""                # raw/nse_api/<folder>/
    purpose = "agents"         # which universe: 'agents' or 'prices'
    per_year = False           # one task per stock per year instead of one per stock
    start_key = "announcements"

    def default_start(self, ctx):
        return ctx.cfg.start_date(self.start_key)

    def url(self, symbol: str, start: date, end: date) -> str:
        raise NotImplementedError

    def plan(self, ctx, start, end, symbols):
        specs = []
        for symbol in symbols or ctx.symbols(self.purpose):
            periods = years(start, end) if self.per_year else [(start, end)]
            for s, e in periods:
                key = f"{symbol}|{s.year}" if self.per_year else symbol
                specs.append(TaskSpec(task_key=key, symbol=symbol, period_start=s, period_end=e,
                                      url=self.url(symbol, s, e)))
        return specs

    def download(self, ctx, task):
        records = as_records(ctx.http.nse_json(task.url))
        if not records:
            return DownloadResult(no_data=True)
        name = f"{task.symbol}_{task.start_date.year}.json" if self.per_year else f"{task.symbol}.json"
        path = raw_path(ctx, "nse_api", self.folder, name)
        return DownloadResult(path, nbytes=save_json(path, records), rows=len(records))

    def records(self, raw) -> pd.DataFrame:
        return pd.DataFrame(load_json(raw))


def require(df: pd.DataFrame, columns: set[str], what: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise IngestError("schema_changed", f"{what}: missing fields {sorted(missing)}; got {list(df.columns)[:20]}")


# ---------------------------------------------------------------- corporate actions

_NUM = r"(\d+(?:\.\d+)?)"
_SPLIT = re.compile(rf"from\s*(?:rs\.?|re\.?|inr)?\s*{_NUM}.*?to\s*(?:rs\.?|re\.?|inr)?\s*{_NUM}", re.IGNORECASE)
_BONUS = re.compile(r"bonus\s*(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_AMOUNT = re.compile(rf"(?:rs\.?|re\.?|inr)\s*{_NUM}", re.IGNORECASE)


def parse_subject(subject: str) -> tuple[str, float | None, float | None]:
    """NSE's free-text subject -> (type, price adjustment factor, dividend amount per share).

    Factor = multiply prices before the ex-date by this, e.g. split Rs 10 -> Rs 2: 0.2; bonus 1:1: 0.5.
    """
    s = (subject or "").strip()
    low = s.lower()
    if "split" in low or "sub-division" in low or "subdivision" in low:
        m = _SPLIT.search(s)
        if m and float(m[1]) > 0 and float(m[2]) > 0:
            return "split", float(m[2]) / float(m[1]), None
        return "split", None, None
    if "bonus" in low:
        m = _BONUS.search(s)
        if m and int(m[2]) > 0:
            new, held = int(m[1]), int(m[2])
            return "bonus", held / (new + held), None
        return "bonus", None, None
    if "demerger" in low or "scheme of arrangement" in low:
        return "demerger", None, None
    if "rights" in low:
        return "rights", None, None
    if "buy back" in low or "buyback" in low:
        return "buyback", None, None
    if "dividend" in low:
        amounts = [float(x) for x in _AMOUNT.findall(s)]
        return "dividend", None, (sum(amounts) or None)
    return "other", None, None


CA_SCHEMA = {
    "symbol": "str", "ex_date": "date", "record_date": "date", "subject": "str", "ca_type": "str",
    "adj_factor": "float", "amount": "float", "face_value": "float", "isin": "str", "bc_start": "date",
    "bc_end": "date", "published_at": "ts", "source": "str",
}


class NseCorporateActions(NseApiSource):
    name = "nse_corporate_actions"
    task_unit = "symbol"
    description = "NSE corporate actions: splits, bonuses, dividends, demergers (per stock)"
    folder = "corporate_actions"
    purpose = "prices"
    start_key = "corporate_actions"

    def url(self, symbol, start, end):
        return (f"{API}/corporates-corporateActions?index=equities&symbol={quote(symbol, safe='')}"
                f"&from_date={nse_date(start)}&to_date={nse_date(end)}")

    def parse(self, ctx, task, raw):
        df = self.records(raw)
        require(df, {"subject", "exDate"}, "corporate actions")
        parsed = df["subject"].map(parse_subject)
        out = pd.DataFrame({
            "symbol": task.symbol,
            "ex_date": df["exDate"].map(parse_date),
            "record_date": df.get("recDate", pd.Series(dtype=object)).map(parse_date),
            "subject": df["subject"].str.strip(),
            "ca_type": parsed.map(lambda p: p[0]),
            "adj_factor": parsed.map(lambda p: p[1]),
            "amount": parsed.map(lambda p: p[2]),
            "face_value": df.get("faceVal"),
            "isin": df.get("isin"),
            "bc_start": df.get("bcStartDate", pd.Series(dtype=object)).map(parse_date),
            "bc_end": df.get("bcEndDate", pd.Series(dtype=object)).map(parse_date),
            "published_at": df.get("caBroadcastDate", pd.Series(dtype=object)).map(parse_ist_datetime),
            "source": "nse_corporate_actions",
        })
        issues = []
        for _, r in out.iterrows():
            if r["ca_type"] in ("split", "bonus") and pd.isna(r["adj_factor"]):
                issues.append(Issue("unparsed_corporate_action", "warn",
                                    {"subject": r["subject"], "ex_date": str(r["ex_date"])}))
            elif r["ca_type"] in ("demerger", "rights"):
                issues.append(Issue("manual_adjustment_needed", "warn",
                                    {"type": r["ca_type"], "subject": r["subject"], "ex_date": str(r["ex_date"])}))
        return ParseResult([Frame("corporate_actions", out, ["symbol", "ex_date", "subject"], "symbol", CA_SCHEMA)],
                           issues)


# ---------------------------------------------------------------- quarterly results filings

RESULTS_SCHEMA = {
    "symbol": "str", "seq_number": "str", "period_from": "date", "period_to": "date", "consolidated": "str",
    "audited": "str", "cumulative": "str", "period": "str", "financial_year": "str", "ind_as": "str",
    "params": "str", "filing_date": "date", "broadcast_at": "ts", "published_at": "ts",
    "detail_link": "str", "source": "str",
}


class NseResults(NseApiSource):
    name = "nse_results"
    task_unit = "symbol"
    description = "NSE quarterly results filings: dates and public times (numbers are an open item)"
    folder = "financial_results"
    start_key = "results"

    def url(self, symbol, start, end):
        return f"{API}/corporates-financial-results?index=equities&symbol={quote(symbol, safe='')}&period=Quarterly"

    def parse(self, ctx, task, raw):
        df = self.records(raw)
        require(df, {"toDate", "consolidated"}, "financial results")
        published = df.get("exchdisstime", pd.Series(dtype=object)).map(parse_ist_datetime)
        broadcast = df.get("broadCastDate", pd.Series(dtype=object)).map(parse_ist_datetime)
        filing = df.get("filingDate", pd.Series(dtype=object)).map(parse_ist_datetime)
        seq = df["seqNumber"] if "seqNumber" in df else df.get("params", "") + "|" + df["consolidated"]
        out = pd.DataFrame({
            "symbol": task.symbol,
            "seq_number": seq.astype(str),
            "period_from": df.get("fromDate", pd.Series(dtype=object)).map(parse_date),
            "period_to": df["toDate"].map(parse_date),
            "consolidated": df["consolidated"],
            "audited": df.get("audited"),
            "cumulative": df.get("cumulative"),
            "period": df.get("period"),
            "financial_year": df.get("financialYear"),
            "ind_as": df.get("indAs"),
            "params": df.get("params"),
            "filing_date": df.get("filingDate", pd.Series(dtype=object)).map(parse_date),
            "broadcast_at": broadcast,
            "published_at": published.fillna(broadcast).fillna(filing),
            "detail_link": df.get("resultDetailedDataLink"),
            "source": "nse_results",
        })
        issues = []
        latest = out["period_to"].dropna().max() if out["period_to"].notna().any() else None
        if latest and latest < task.end_date - timedelta(days=200):
            issues.append(Issue("results_list_stale", "info", {"latest_quarter": str(latest)}))
        return ParseResult([Frame("financial_results", out, ["symbol", "seq_number"], "symbol", RESULTS_SCHEMA)],
                           issues)


# ---------------------------------------------------------------- board meetings

BOARD_SCHEMA = {
    "symbol": "str", "meeting_date": "date", "purpose": "str", "description": "str", "meeting_type": "str",
    "attachment_url": "str", "announced_at": "ts", "published_at": "ts", "source": "str",
}


class NseBoardMeetings(NseApiSource):
    name = "nse_board_meetings"
    task_unit = "symbol"
    description = "NSE board meetings (results dates known in advance, for the catalyst calendar)"
    folder = "board_meetings"

    def url(self, symbol, start, end):
        return f"{API}/corporate-board-meetings?index=equities&symbol={quote(symbol, safe='')}"

    def parse(self, ctx, task, raw):
        df = self.records(raw)
        require(df, {"bm_date", "bm_purpose"}, "board meetings")
        announced = df.get("bm_timestamp", pd.Series(dtype=object)).map(parse_ist_datetime)
        out = pd.DataFrame({
            "symbol": task.symbol,
            "meeting_date": df["bm_date"].map(parse_date),
            "purpose": df["bm_purpose"],
            "description": df.get("bm_desc"),
            "meeting_type": df.get("meetingType"),
            "attachment_url": df.get("attachment"),
            "announced_at": announced,
            "published_at": announced,
            "source": "nse_board_meetings",
        })
        return ParseResult([Frame("board_meetings", out, ["symbol", "meeting_date", "purpose"], "symbol",
                                  BOARD_SCHEMA)])


# ---------------------------------------------------------------- insider trades (SEBI PIT disclosures)

INSIDER_SCHEMA = {
    "row_id": "str", "symbol": "str", "person": "str", "person_category": "str", "acq_mode": "str",
    "security_type": "str", "quantity": "float", "value": "float", "buy_value": "float", "sell_value": "float",
    "before_pct": "float", "after_pct": "float", "acq_from": "date", "acq_to": "date",
    "intimation_date": "date", "published_at": "ts", "exchange": "str", "remarks": "str", "source": "str",
}


class NseInsider(NseApiSource):
    name = "nse_insider"
    task_unit = "symbol_year"
    description = "NSE insider trading disclosures (promoters, directors), per stock per year"
    folder = "insider"
    per_year = True
    start_key = "insider"

    def url(self, symbol, start, end):
        return (f"{API}/corporates-pit?index=equities&symbol={quote(symbol, safe='')}"
                f"&from_date={nse_date(start)}&to_date={nse_date(end)}")

    def parse(self, ctx, task, raw):
        records = load_json(raw)
        df = pd.DataFrame(records)
        require(df, {"acqName"}, "insider trades")
        intimation = df.get("intimDt", pd.Series(dtype=object))
        published = df.get("date", pd.Series(dtype=object)).map(parse_ist_datetime)
        out = pd.DataFrame({
            "row_id": [row_hash(r) for r in records],
            "symbol": task.symbol,
            "person": df["acqName"],
            "person_category": df.get("personCategory"),
            "acq_mode": df.get("acqMode"),
            "security_type": df.get("secType"),
            "quantity": df.get("secAcq"),
            "value": df.get("secVal"),
            "buy_value": df.get("buyValue"),
            "sell_value": df.get("sellValue"),
            "before_pct": df.get("befAcqSharesPer"),
            "after_pct": df.get("afterAcqSharesPer"),
            "acq_from": df.get("acqfromDt", pd.Series(dtype=object)).map(parse_date),
            "acq_to": df.get("acqtoDt", pd.Series(dtype=object)).map(parse_date),
            "intimation_date": intimation.map(parse_date),
            "published_at": published.fillna(intimation.map(parse_ist_datetime)),
            "exchange": df.get("exchange"),
            "remarks": df.get("remarks"),
            "source": "nse_insider",
        })
        return ParseResult([Frame("insider_trades", out, ["row_id"], "symbol", INSIDER_SCHEMA)])


# ---------------------------------------------------------------- snapshots kept as JSON records

SNAPSHOT_SCHEMA = {"row_id": "str", "symbol": "str", "record_json": "str", "source": "str"}


class NseSnapshot(NseApiSource):
    """Endpoints whose fields we haven't modelled yet: each record is kept as JSON text."""
    endpoint = ""
    table = ""

    def url(self, symbol, start, end):
        return f"{API}/{self.endpoint}?index=equities&symbol={quote(symbol, safe='')}"

    def parse(self, ctx, task, raw):
        records = load_json(raw)
        out = pd.DataFrame({
            "row_id": [row_hash(r) for r in records],
            "symbol": task.symbol,
            "record_json": [json.dumps(r, ensure_ascii=False, default=str) for r in records],
            "source": self.name,
        })
        return ParseResult([Frame(self.table, out, ["row_id"], "symbol", SNAPSHOT_SCHEMA)])


class NsePledges(NseSnapshot):
    name = "nse_pledges"
    task_unit = "symbol"
    description = "NSE promoter pledging data (kept as JSON records)"
    folder = "pledges"
    endpoint = "corporate-pledgedata"
    table = "pledges"


class NseShareholding(NseSnapshot):
    name = "nse_shareholding"
    task_unit = "symbol"
    description = "NSE shareholding pattern (kept as JSON records)"
    folder = "shareholding"
    endpoint = "corporate-share-holdings-master"
    table = "shareholding"
