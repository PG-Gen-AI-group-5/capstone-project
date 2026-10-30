"""Daily NSE files: bhavcopy (prices), index closes (Nifty 50, India VIX), FII/DII flows."""
from __future__ import annotations

import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.ingest.errors import IngestError
from src.ingest.http import write_atomic
from src.ingest.sources.base import (
    Context,
    DownloadResult,
    Frame,
    Issue,
    ParseResult,
    Source,
    as_records,
    daily_404,
    load_json,
    raw_path,
    read_csv_from_zip,
    save_json,
    weekday_tasks,
)
from src.ingest.tracker import TaskSpec
from src.timeutil import at_ist, parse_date, today_ist

# ---------------------------------------------------------------- bhavcopy

OLD_FORMAT_LAST_DAY = date(2024, 7, 5)   # NSE switched to the new (UDiFF) file on 8 Jul 2024

OLD_COLUMNS = {"SYMBOL": "raw_symbol", "SERIES": "series", "TIMESTAMP": "trade_date", "OPEN": "open",
               "HIGH": "high", "LOW": "low", "CLOSE": "close", "LAST": "last", "PREVCLOSE": "prev_close",
               "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover", "TOTALTRADES": "trades", "ISIN": "isin"}
NEW_COLUMNS = {"TckrSymb": "raw_symbol", "SctySrs": "series", "TradDt": "trade_date", "OpnPric": "open",
               "HghPric": "high", "LwPric": "low", "ClsPric": "close", "LastPric": "last",
               "PrvsClsgPric": "prev_close", "TtlTradgVol": "volume", "TtlTrfVal": "turnover",
               "TtlNbOfTxsExctd": "trades", "ISIN": "isin"}

PRICE_SCHEMA = {
    "symbol": "str", "raw_symbol": "str", "trade_date": "date", "open": "float", "high": "float",
    "low": "float", "close": "float", "last": "float", "prev_close": "float", "volume": "int",
    "turnover": "float", "trades": "int", "isin": "str", "series": "str", "published_at": "ts",
    "source": "str", "year": "str",
}


def bhavcopy_url(d: date) -> str:
    if d <= OLD_FORMAT_LAST_DAY:
        mon = d.strftime("%b").upper()
        return (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
                f"{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip")
    return f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"


def parse_bhavcopy(df: pd.DataFrame) -> pd.DataFrame:
    """Map either file format to our column names and keep equity (EQ series) rows."""
    df = df.rename(columns=lambda c: str(c).strip())
    if set(OLD_COLUMNS) <= set(df.columns):
        mapping = OLD_COLUMNS
    elif set(NEW_COLUMNS) <= set(df.columns):
        mapping = NEW_COLUMNS
    else:
        raise IngestError("schema_changed", f"Unknown bhavcopy columns: {list(df.columns)[:15]}")
    out = df[list(mapping)].rename(columns=mapping)
    for col in ("raw_symbol", "series", "isin", "trade_date"):
        out[col] = out[col].astype(str).str.strip()
    return out[out["series"] == "EQ"].copy()


def first_seen(ctx: Context) -> dict[str, date]:
    """First trading date each stock appears in our price data (cached for the run)."""
    if "first_seen" not in ctx.state:
        prices = ctx.lake.read("prices_daily", ["symbol", "trade_date"])
        ctx.state["first_seen"] = (prices.groupby("symbol")["trade_date"].min().to_dict()
                                   if not prices.empty else {})
    return ctx.state["first_seen"]


def expected_on(ctx: Context, symbol: str, d: date, seen: dict[str, date]) -> bool:
    """Should this stock have a price on day d? Uses data_start from universe.csv, else first appearance."""
    start = ctx.universe.data_start(symbol)
    if start:
        return d >= start
    fs = seen.get(symbol)
    return fs is not None and fs < d


class NseBhavcopy(Source):
    name = "nse_bhavcopy"
    task_unit = "day"
    description = "NSE daily bhavcopy: OHLCV for every stock (one zip per trading day)"
    fallback_name = "yfinance"

    def default_start(self, ctx):
        return ctx.cfg.start_date("prices")

    def plan(self, ctx, start, end, symbols):
        return weekday_tasks(ctx, start, end, bhavcopy_url)

    def download(self, ctx, task):
        d = task.start_date
        path = raw_path(ctx, "nse_bhavcopy", str(d.year), Path(task.url).name)
        if path.exists() and zipfile.is_zipfile(path):
            return DownloadResult(path, nbytes=path.stat().st_size)
        try:
            data = ctx.http.get_bytes(task.url, kind="zip")
        except IngestError as e:
            if e.error_type == "http_404":
                return daily_404(ctx, d)
            raise
        write_atomic(path, data)
        return DownloadResult(path, nbytes=len(data))

    def parse(self, ctx, task, raw):
        d = task.start_date
        eq = parse_bhavcopy(read_csv_from_zip(raw))
        issues: list[Issue] = []
        file_dates = {parse_date(x) for x in eq["trade_date"].unique()}
        if file_dates and file_dates != {d}:
            issues.append(Issue("date_mismatch", "error", {"file_dates": sorted(map(str, file_dates))}, True))

        universe = set(ctx.symbols("prices"))
        old_map = ctx.universe.old_symbol_map()
        eq["symbol"] = eq["raw_symbol"].map(lambda s: old_map.get(s, s))
        rows = eq[eq["symbol"].isin(universe)].copy()
        rows["trade_date"] = d
        rows["published_at"] = at_ist(d, ctx.cfg["prices"]["publish_time_ist"])
        rows["source"] = "nse_bhavcopy"
        rows["year"] = str(d.year)

        if len(eq) < 500:
            issues.append(Issue("too_few_rows", "error", {"eq_rows": len(eq)}, blocking=True))
        if rows.empty:
            issues.append(Issue("no_universe_rows", "error", None, blocking=True))
        dupes = sorted(rows.loc[rows["symbol"].duplicated(), "symbol"])
        if dupes:
            issues.append(Issue("duplicate_symbol", "warn", {"symbols": dupes}))
            rows = rows.drop_duplicates("symbol", keep="last")

        seen = first_seen(ctx)
        missing = sorted(s for s in universe if expected_on(ctx, s, d, seen) and s not in set(rows["symbol"]))
        if missing:
            issues.append(Issue("missing_symbol", "warn", {"symbols": missing}))
        for s in rows["symbol"]:
            if s not in seen or d < seen[s]:
                seen[s] = d
        return ParseResult([Frame("prices_daily", rows, ["symbol", "trade_date"], "year", PRICE_SCHEMA)], issues)

    def fallback(self, ctx, task):
        """Fill a missing day from yfinance (used when a review decides 'use_fallback')."""
        import yfinance as yf

        d = task.start_date
        symbols = ctx.symbols("prices")
        data = yf.download([f"{s}.NS" for s in symbols], start=d.isoformat(),
                           end=(d + timedelta(days=1)).isoformat(), auto_adjust=False, progress=False,
                           group_by="ticker", threads=True)
        rows = []
        for s in symbols:
            ticker = f"{s}.NS"
            if data.empty or ticker not in data.columns.get_level_values(0):
                continue
            sub = data[ticker].dropna(how="all")
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({"symbol": s, "raw_symbol": s, "trade_date": d, "open": r["Open"], "high": r["High"],
                         "low": r["Low"], "close": r["Close"], "volume": r["Volume"], "series": "EQ"})
        if not rows:
            raise IngestError("empty_response", f"yfinance returned no prices for {d}")
        df = pd.DataFrame(rows)
        df["published_at"] = at_ist(d, ctx.cfg["prices"]["publish_time_ist"])
        df["source"] = "yfinance"
        df["year"] = str(d.year)
        return ParseResult([Frame("prices_daily", df, ["symbol", "trade_date"], "year", PRICE_SCHEMA)])


# ---------------------------------------------------------------- index closes

INDEX_COLUMNS = {"Index Name": "index_name", "Index Date": "trade_date", "Open Index Value": "open",
                 "High Index Value": "high", "Low Index Value": "low", "Closing Index Value": "close",
                 "Points Change": "points_change", "Change(%)": "change_pct", "Volume": "volume",
                 "Turnover (Rs. Cr.)": "turnover_cr", "P/E": "pe", "P/B": "pb", "Div Yield": "div_yield"}
KEEP_INDICES = {"nifty 50": "Nifty 50", "india vix": "India VIX"}

INDEX_SCHEMA = {
    "index_name": "str", "trade_date": "date", "open": "float", "high": "float", "low": "float",
    "close": "float", "points_change": "float", "change_pct": "float", "volume": "float",
    "turnover_cr": "float", "pe": "float", "pb": "float", "div_yield": "float", "published_at": "ts",
    "source": "str", "year": "str",
}


def index_url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv"


class NseIndices(Source):
    name = "nse_indices"
    task_unit = "day"
    description = "NSE daily index closes: Nifty 50 and India VIX (one CSV per trading day)"

    def default_start(self, ctx):
        return ctx.cfg.start_date("prices")

    def plan(self, ctx, start, end, symbols):
        return weekday_tasks(ctx, start, end, index_url)

    def download(self, ctx, task):
        d = task.start_date
        path = raw_path(ctx, "nse_indices", str(d.year), Path(task.url).name)
        if path.exists() and path.stat().st_size > 0:
            return DownloadResult(path, nbytes=path.stat().st_size)
        try:
            data = ctx.http.get_bytes(task.url, kind="csv")
        except IngestError as e:
            if e.error_type == "http_404":
                return daily_404(ctx, d)
            raise
        write_atomic(path, data)
        return DownloadResult(path, nbytes=len(data))

    def parse(self, ctx, task, raw):
        d = task.start_date
        df = pd.read_csv(raw, dtype=str).rename(columns=lambda c: str(c).strip())
        if not {"Index Name", "Index Date", "Closing Index Value"} <= set(df.columns):
            raise IngestError("schema_changed", f"Unknown index file columns: {list(df.columns)[:15]}")
        df = df.rename(columns=INDEX_COLUMNS)
        key = df["index_name"].str.strip().str.lower()
        df = df[key.isin(KEEP_INDICES)].copy()
        df["index_name"] = df["index_name"].str.strip().str.lower().map(KEEP_INDICES)
        df["trade_date"] = df["trade_date"].map(parse_date)   # "-" values become NaN when the lake conforms types
        df["published_at"] = at_ist(d, ctx.cfg["prices"]["publish_time_ist"])
        df["source"] = "nse_indices"
        df["year"] = str(d.year)
        issues = []
        if "Nifty 50" not in set(df["index_name"]):
            issues.append(Issue("missing_index", "error", {"index": "Nifty 50"}, blocking=True))
        if "India VIX" not in set(df["index_name"]):
            issues.append(Issue("missing_index", "warn", {"index": "India VIX"}))
        return ParseResult([Frame("index_daily", df, ["index_name", "trade_date"], "year", INDEX_SCHEMA)], issues)


# ---------------------------------------------------------------- FII / DII flows

FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
FII_DII_SCHEMA = {"trade_date": "date", "category": "str", "buy_value_cr": "float", "sell_value_cr": "float",
                  "net_value_cr": "float", "published_at": "ts", "source": "str", "year": "str"}


class NseFiiDii(Source):
    name = "nse_fii_dii"
    task_unit = "day"
    description = "NSE provisional FII/DII cash-market flows (only the latest day is available)"

    def plan(self, ctx, start, end, symbols):
        today = today_ist()
        return [TaskSpec(task_key=today.isoformat(), period_start=today, period_end=today, url=FII_DII_URL)]

    def download(self, ctx, task):
        records = as_records(ctx.http.nse_json(task.url))
        if not records:
            return DownloadResult(no_data=True)
        path = raw_path(ctx, "nse_api", "fii_dii", f"{task.task_key}.json")
        return DownloadResult(path, nbytes=save_json(path, records), rows=len(records))

    def parse(self, ctx, task, raw):
        df = pd.DataFrame(load_json(raw))
        if not {"category", "date", "buyValue", "sellValue", "netValue"} <= set(df.columns):
            raise IngestError("schema_changed", f"Unknown FII/DII fields: {list(df.columns)}")
        df["trade_date"] = df["date"].map(parse_date)
        df = df.rename(columns={"buyValue": "buy_value_cr", "sellValue": "sell_value_cr",
                                "netValue": "net_value_cr"})
        df["published_at"] = df["trade_date"].map(lambda d: at_ist(d, ctx.cfg["fii_dii"]["publish_time_ist"]))
        df["source"] = "nse_fii_dii"
        df["year"] = df["trade_date"].map(lambda d: str(d.year))
        issues = []
        latest = df["trade_date"].max()
        if latest and latest < task.start_date - timedelta(days=5):
            issues.append(Issue("stale_data", "warn", {"latest_date": str(latest)}))
        return ParseResult([Frame("fii_dii", df, ["trade_date", "category"], "year", FII_DII_SCHEMA)], issues)
