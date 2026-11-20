from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.analysis import inventory as inv


def _days(*ds):
    return pd.DatetimeIndex(pd.to_datetime(list(ds)))


def _ist(*ts):
    return pd.Series(pd.to_datetime(list(ts), format="ISO8601")).dt.tz_localize(inv.IST).dt.tz_convert("UTC")


# Mon 3 Jun - Fri 7 Jun 2024, then Mon 10 Jun; 8-9 Jun is a weekend
DAYS = _days("2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-10")


def test_trading_days_uses_is_trading():
    cal = pd.DataFrame({"trade_date": ["2024-06-07", "2024-06-08", "2024-06-10"], "is_trading": [1, 0, 1]})
    assert list(inv.trading_days(cal)) == list(_days("2024-06-07", "2024-06-10"))


def test_inventory_row_counts_span_nulls_and_source():
    spec = inv.TableSpec("t", "published_at", ["symbol", "published_at", "category"])
    df = pd.DataFrame({
        "symbol": ["A", "A", "B", "C"],
        "published_at": _ist("2024-01-01 10:00", "2024-03-01 10:00", "2025-01-01 00:30", None),
        "category": ["x", None, "y", "z"],
        "source": ["nse", "nse", "nse", "nse"],
    })
    row = inv.inventory_row(df, spec)
    assert row["rows"] == 4 and row["stocks"] == 3
    # 00:30 IST on 1 Jan 2025 is still 1 Jan in IST (it is 31 Dec in UTC)
    assert str(row["first"]) == "2024-01-01" and str(row["last"]) == "2025-01-01"
    assert row["null_share"] == "published_at=25%, category=25%"
    assert row["source"] == "nse"


def test_stock_year_counts():
    df = pd.DataFrame({"symbol": ["A", "A", "B"], "d": pd.to_datetime(["2016-05-01", "2017-01-01", "2017-02-01"])})
    m = inv.stock_year_counts(df, "d")
    assert m.loc["A", 2016] == 1 and m.loc["A", 2017] == 1 and m.loc["B", 2016] == 0


def test_price_coverage_counts_from_first_day_only():
    prices = pd.DataFrame({
        "symbol": ["A"] * 5 + ["B"] * 2,
        "trade_date": ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-07", "2024-06-10",  # A misses 6 Jun
                       "2024-06-07", "2024-06-10"],                                             # B lists 7 Jun
    })
    cov = inv.price_coverage(prices, DAYS)
    assert cov.loc["A", 2024] == pytest.approx(5 / 6)
    assert cov.loc["B", 2024] == 1.0


def test_series_day_coverage():
    s = inv.series_day_coverage(pd.Series(["2024-06-03", "2024-06-04", "2024-06-08"]), DAYS)
    # range 3 Jun - 8 Jun has 5 trading days; 2 of them have a value (8 Jun is a weekend)
    assert s.loc[2024] == pytest.approx(2 / 5)


def test_calendar_mismatch_finds_missing_and_extra_days():
    missing, extra = inv.calendar_mismatch(pd.Series(["2024-06-03", "2024-06-05", "2024-06-09", "2024-06-10"]), DAYS)
    assert [str(d) for d in missing] == ["2024-06-04", "2024-06-06", "2024-06-07"]
    assert [str(d) for d in extra] == ["2024-06-09"]


def test_has_time_and_ist_session():
    ts = _ist("2024-06-03 00:00:00", "2024-06-03 08:29:59", "2024-06-03 08:30", "2024-06-03 15:30", None)
    assert inv.has_time(ts).tolist() == [False, True, True, True, False]
    sess = inv.ist_session(ts)
    assert sess.tolist()[:4] == ["before 08:30", "before 08:30", "08:30-15:30", "after 15:30"]
    assert pd.isna(sess.iloc[4])


def test_first_informed_day_follows_the_time_rule():
    ts = _ist(
        "2024-06-03 08:29",   # before the forecast on a trading day -> same day
        "2024-06-03 08:30",   # exactly at 08:30 is not before as_of -> next day
        "2024-06-07 16:00",   # Friday evening -> Monday
        "2024-06-08 07:00",   # Saturday morning -> Monday
        "2024-06-10 20:00",   # after the last calendar day -> unknown
        "2024-05-31 20:00",   # before the first calendar day -> unknown, not 3 Jun
    )
    out = inv.first_informed_day(ts, DAYS)
    assert [None if pd.isna(x) else str(x.date()) for x in out] == [
        "2024-06-03", "2024-06-04", "2024-06-10", "2024-06-10", None, None]


def test_large_moves_marks_corporate_action_days():
    prices = pd.DataFrame({
        "symbol": ["A", "A", "B"], "trade_date": ["2024-06-03", "2024-06-04", "2024-06-04"],
        "close": [50.0, 52.0, 70.0], "prev_close": [100.0, 50.0, 100.0],
    })
    ca = pd.DataFrame({"symbol": ["A", "B"], "ex_date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
                       "ca_type": ["split", "dividend"]})
    out = inv.large_moves(prices, ca)
    assert len(out) == 2  # A on 4 Jun moved only 4%
    a = out[out.symbol == "A"].iloc[0]
    b = out[out.symbol == "B"].iloc[0]
    assert a.explained and a.ca_type == "split"
    assert not b.explained  # a dividend does not explain a 30% fall


def test_late_starts():
    prices = pd.DataFrame({"symbol": ["A", "B"], "trade_date": ["2016-01-01", "2020-02-27"]})
    uni = pd.DataFrame({"symbol": ["A", "B"], "data_start": [None, None], "old_symbols": [None, "OLDB"],
                        "notes": [None, None]})
    out = inv.late_starts(prices, uni, "2016-01-01")
    assert out.symbol.tolist() == ["B"] and out.old_symbols.iloc[0] == "OLDB"


def test_json_specs_parse_dates():
    con = duckdb.connect()
    con.execute("CREATE TABLE shareholding AS SELECT 'A' symbol, "
                "'{\"date\": \"30-SEP-2026\", \"broadcastDate\": \"12-OCT-2026 18:48:24\"}' record_json, 'nse' source")
    df = inv.load_table(con, inv.TABLES["shareholding"])
    assert str(df.period_end.iloc[0])[:10] == "2026-09-30"
    assert str(df.broadcast_ist.iloc[0]) == "2026-10-12 18:48:24"
    assert inv.inventory_row(df, inv.TABLES["shareholding"])["stocks"] == 1
