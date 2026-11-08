from __future__ import annotations

import io
import zipfile
from datetime import date, datetime

import pandas as pd
import pytest

from src.ingest.errors import IngestError
from src.ingest.lake import Lake
from src.ingest.sources.gdelt import build_matchers, filter_rows
from src.ingest.sources.nse_announcements import is_important
from src.ingest.sources.nse_corporate import parse_subject
from src.ingest.sources.nse_daily import NseBhavcopy, NseIndices, bhavcopy_url, parse_bhavcopy, yf_index_rows
from src.ingest.tracker import Task
from src.ingest.universe import default_aliases
from src.timeutil import UTC, at_ist, parse_date, parse_ist_datetime

OLD_HEADER = "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,"
NEW_HEADER = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,"
              "StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,"
              "SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,"
              "Rsvd1,Rsvd2,Rsvd3,Rsvd4")


def old_row(symbol, series="EQ", day="04-JAN-2016"):
    return f"{symbol},{series},100,110,95,105,105,101,5000,525000,{day},300,INE{symbol},"


# ---------------------------------------------------------------- time

def test_time_parsing():
    assert parse_ist_datetime("23-Jan-2018 14:30:00") == datetime(2018, 1, 23, 9, 0, tzinfo=UTC)
    assert parse_date("04-JAN-2016") == date(2016, 1, 4)
    assert parse_date("04-01-2016") == date(2016, 1, 4)
    assert parse_date("2025-10-01") == date(2025, 10, 1)
    assert parse_date("13-Jul-20") == date(2020, 7, 13)  # the 13 Jul 2020 bhavcopy uses a 2-digit year
    assert parse_date("-") is None
    assert at_ist(date(2024, 6, 3), "18:00") == datetime(2024, 6, 3, 12, 30, tzinfo=UTC)
    # a date without a time is treated as end of day IST (usable only from the next day)
    assert parse_ist_datetime("09-Jun-2016") == datetime(2016, 6, 9, 18, 29, tzinfo=UTC)


# ---------------------------------------------------------------- bhavcopy

def test_bhavcopy_url_switches_format_on_8_july_2024():
    assert bhavcopy_url(date(2024, 7, 5)).endswith("/EQUITIES/2024/JUL/cm05JUL2024bhav.csv.zip")
    assert bhavcopy_url(date(2024, 7, 8)).endswith("/BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip")


def test_parse_both_formats():
    old = pd.read_csv(io.StringIO("\n".join([OLD_HEADER, old_row("AAA"), old_row("AAA", "BE")])), dtype=str)
    assert list(parse_bhavcopy(old)["raw_symbol"]) == ["AAA"]

    new_row = ("2025-10-01,2025-10-01,CM,NSE,STK,1,INE009A01021,INFY,EQ,,,,,INFOSYS LTD,1500,1520,1490,1510,1511,"
               "1495,,1510,,,1000000,1510000000,25000,F1,1,,,,,")
    new = pd.read_csv(io.StringIO(NEW_HEADER + "\n" + new_row), dtype=str)
    parsed = parse_bhavcopy(new)
    assert parsed.iloc[0]["raw_symbol"] == "INFY" and parsed.iloc[0]["close"] == "1510"

    with pytest.raises(IngestError) as e:
        parse_bhavcopy(pd.DataFrame({"foo": [1]}))
    assert e.value.error_type == "schema_changed"


def test_bhavcopy_parse_maps_old_symbols_and_flags_missing_stocks(ctx, tmp_path):
    rows = [OLD_HEADER, old_row("AAA"), old_row("ZOMATO")] + [old_row(f"X{i}") for i in range(600)]
    path = tmp_path / "cm04JAN2016bhav.csv.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("cm04JAN2016bhav.csv", "\n".join(rows))
    ctx.state["first_seen"] = {"BBB": date(2015, 12, 1)}   # BBB traded before, so it's expected today
    task = Task(1, "nse_bhavcopy", "2016-01-04", "*", "2016-01-04", "2016-01-04", None, None, None,
                "parse", "running", 0, str(path))

    result = NseBhavcopy().parse(ctx, task, path)
    frame = result.frames[0]
    assert sorted(frame.df["symbol"]) == ["AAA", "ETERNAL"]
    assert frame.df.loc[frame.df["symbol"] == "ETERNAL", "raw_symbol"].iloc[0] == "ZOMATO"
    issues = {i.check_name: i for i in result.issues}
    assert issues["missing_symbol"].details == {"symbols": ["BBB"]}
    assert not any(i.blocking for i in result.issues)


INDEX_HEADER = ("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,"
                "Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield")


def parse_index_file(ctx, tmp_path, day: str, file_date: str):
    path = tmp_path / "ind_close_all.csv"
    path.write_text("\n".join([INDEX_HEADER,
                               f"Nifty 50,{file_date},17500,17650,17450,17600,100,0.57,1,1,21,4,1.3",
                               f"India VIX,{file_date},12,13,11,12.5,0.5,4,-,-,-,-,-"]))
    task = Task(1, "nse_indices", day, "*", day, day, None, None, None, "parse", "running", 0, str(path))
    return NseIndices().parse(ctx, task, path)


def test_index_parse_uses_the_task_date_even_when_the_file_writes_month_first(ctx, tmp_path):
    # NSE's files for 6, 10 and 11 Apr 2023 write the date as MM-DD-YYYY ("04-10-2023" = 10 Apr)
    result = parse_index_file(ctx, tmp_path, "2023-04-10", "04-10-2023")
    assert set(result.frames[0].df["trade_date"]) == {date(2023, 4, 10)}
    assert not any(i.blocking for i in result.issues)


def test_index_parse_blocks_a_file_for_another_day(ctx, tmp_path):
    result = parse_index_file(ctx, tmp_path, "2023-04-10", "11-04-2023")
    assert [i.check_name for i in result.issues if i.blocking] == ["date_mismatch"]


def test_index_rows_from_yfinance_keep_only_the_requested_day():
    # NSE has no index file for 20 Jun 2016; the fallback reads ^NSEI and ^INDIAVIX from yfinance
    days = pd.to_datetime(["2016-06-17", "2016-06-20"])
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([["^NSEI", "^INDIAVIX"], fields])
    data = pd.DataFrame([[8140, 8180, 8130, 8170.2, 0, 17.0, 17.6, 16.9, 17.35, 0],
                         [8200, 8250, 8190, 8238.5, 0, 17.4, 17.9, 17.1, 17.52, 0]], index=days, columns=cols)
    df = yf_index_rows(data, date(2016, 6, 20)).set_index("index_name")
    assert df.loc["Nifty 50", "close"] == 8238.5
    assert df.loc["India VIX", "close"] == 17.52
    assert set(df["trade_date"]) == {date(2016, 6, 20)}
    assert yf_index_rows(pd.DataFrame(), date(2016, 6, 20)).empty


# ---------------------------------------------------------------- corporate actions

@pytest.mark.parametrize("subject, expected", [
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", ("split", 0.2, None)),
    ("Bonus 1:1", ("bonus", 0.5, None)),
    ("Bonus 3:2", ("bonus", 0.4, None)),
    ("Annual General Meeting/Dividend - Rs 14.25 Per Share", ("dividend", None, 14.25)),
    ("Interim Dividend - Rs 21 Per Share / Special Dividend - Rs 8 Per Share", ("dividend", None, 29.0)),
    ("Demerger", ("demerger", None, None)),
    ("Rights 1:10 @ Premium Rs 500/-", ("rights", None, None)),
])
def test_parse_subject(subject, expected):
    kind, factor, amount = parse_subject(subject)
    assert kind == expected[0]
    assert factor == pytest.approx(expected[1]) if expected[1] else factor is None
    assert amount == pytest.approx(expected[2]) if expected[2] else amount is None


# ---------------------------------------------------------------- announcements

def test_is_important(cfg):
    rules = cfg["announcements"]
    assert is_important("Press Release", "", rules)
    assert is_important("Updates", "The company has received an order worth Rs 500 crore", rules)
    assert not is_important("Updates", "Closure of trading window", rules)
    assert not is_important("Copy of Newspaper Publication", "", rules)


# ---------------------------------------------------------------- GDELT

def gkg_row(orgs: str, title: str, tone="2.5,4.0,1.5,5.5,20,0,350") -> str:
    cols = [""] * 27
    cols[0], cols[1], cols[3], cols[4] = "20261030110000-1", "20261030110000", "livemint.com", "https://x.com/a"
    cols[8] = "ECON_STOCKMARKET,10;TAX_FNCACT_CEO,20;ECON_STOCKMARKET,30"
    cols[13], cols[15] = orgs, tone
    cols[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(cols)


def test_gdelt_filter_matches_whole_names_and_scores_relevance():
    any_rx, per_symbol = build_matchers({"INFY": ["infosys"], "AAA": ["aaa industries"]})
    lines = [
        gkg_row("infosys;reserve bank of india", "Infosys wins large deal"),
        gkg_row("infosystems ltd", "Unrelated company"),
        gkg_row("a;b;c;d;e;f;aaa industries", "Market wrap"),
    ]
    records = filter_rows(io.StringIO("\n".join(lines)), any_rx, per_symbol)
    assert [(r["symbol"], r["in_title"], r["relevance"]) for r in records] == [("INFY", True, 1.0),
                                                                                ("AAA", False, 0.3)]
    assert records[0]["tone"] == "2.5" and records[0]["word_count"] == "350"
    assert records[0]["themes"] == "ECON_STOCKMARKET;TAX_FNCACT_CEO"


# ---------------------------------------------------------------- lake

def test_lake_upsert_replaces_rows_with_the_same_key(tmp_path):
    lake = Lake(tmp_path)
    schema = {"symbol": "str", "trade_date": "date", "close": "float", "year": "str"}
    df = pd.DataFrame({"symbol": ["A", "B"], "trade_date": [date(2024, 1, 2)] * 2, "close": [1.0, 2.0],
                       "year": ["2024"] * 2})
    lake.upsert("prices", df, ["symbol", "trade_date"], "year", schema)
    lake.upsert("prices", df, ["symbol", "trade_date"], "year", schema)
    assert len(lake.read("prices")) == 2
    lake.upsert("prices", df.assign(close=[9.0, 2.0]).iloc[:1], ["symbol", "trade_date"], "year", schema)
    out = lake.read("prices").set_index("symbol")
    assert out.loc["A", "close"] == 9.0 and len(out) == 2


# ---------------------------------------------------------------- universe

def test_default_aliases():
    assert default_aliases("Larsen & Toubro Ltd.") == "larsen & toubro; larsen and toubro"
    assert default_aliases("Infosys Ltd.") == "infosys"
    assert default_aliases("HDFC Bank Ltd.") == "hdfc bank"
