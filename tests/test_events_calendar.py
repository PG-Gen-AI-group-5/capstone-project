import numpy as np
import pandas as pd
import pytest

from src.features.build import calendar_table, expiry_days, rolling_beta
from src.features.events import event_features, results_records, results_releases
from src.store.db import skipped_sessions
from src.store.pit import first_usable_day
from tests import synthetic

DAYS = pd.bdate_range("2024-06-03", "2024-06-28")            # Mon 3 Jun .. Fri 28 Jun


def ist(*stamps):
    return pd.Series(pd.to_datetime(list(stamps), format="ISO8601")).dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")


def test_first_usable_day_applies_published_before_as_of():
    out = first_usable_day(ist("2024-06-03 08:29:59", "2024-06-03 08:30:00", "2024-06-07 16:00"), DAYS)
    assert [str(d.date()) for d in out] == ["2024-06-03", "2024-06-04", "2024-06-10"]


def announcements(*rows):
    """rows: (symbol, IST time, category, important, summary)"""
    return pd.DataFrame({"symbol": [r[0] for r in rows], "published_at": ist(*[r[1] for r in rows]),
                         "category": [r[2] for r in rows], "is_important": [r[3] for r in rows],
                         "summary": [r[4] for r in rows]})


NO_RESULTS = pd.DataFrame({"symbol": pd.Series(dtype=str), "broadcast_at": pd.Series(dtype="datetime64[ns, UTC]")})


def keys(symbol="AAA"):
    return pd.DataFrame({"symbol": symbol, "trade_date": DAYS})


def test_an_announcement_at_the_forecast_time_counts_from_the_next_day():
    """Leakage test: a record published at 08:30 IST on d (not before as_of) must not appear in d's row."""
    ann = announcements(("AAA", "2024-06-05 08:30", "Press Release", True, ""),
                        ("AAA", "2024-06-05 08:29", "Press Release", True, ""))
    out = event_features(keys(), pd.DataFrame(columns=["symbol", "day0"]), ann, DAYS).set_index("trade_date")
    assert out.loc["2024-06-05", "ann_important_day0"] == pytest.approx(np.log1p(1))
    assert out.loc["2024-06-06", "ann_important_day0"] == pytest.approx(np.log1p(1))
    assert out.loc["2024-06-04", "ann_important_day0"] == 0.0


def test_results_records_merge_into_one_release_on_the_earliest_day0():
    ann = announcements(
        ("AAA", "2024-06-05 17:00", "Financial Result Updates", True, ""),        # day 0 = 6 Jun
        ("AAA", "2024-06-06 18:00", "Outcome of Board Meeting", True, "Financial results for Q4"),
        ("AAA", "2024-06-20 18:00", "Outcome of Board Meeting", True, "Fund raising"),  # not results
    )
    results = pd.DataFrame({"symbol": ["AAA"], "broadcast_at": ist("2024-06-07 10:00")})
    recs = results_records(results, ann, ["Financial Result Updates"])
    assert len(recs) == 3
    rel = results_releases(recs, DAYS, merge_days=10)
    assert rel["day0"].tolist() == [pd.Timestamp("2024-06-06")]

    out = event_features(keys(), rel, ann, DAYS, cap=90).set_index("trade_date")
    assert out.loc["2024-06-06", "results_day0"] == 1.0 and out["results_day0"].sum() == 1.0
    assert out.loc["2024-06-05", "days_since_results"] == 90          # before the first release: the cap
    assert out.loc["2024-06-06", "days_since_results"] == 0
    assert out.loc["2024-06-10", "days_since_results"] == 2


def test_a_record_after_the_merge_window_is_a_new_release():
    recs = pd.DataFrame({"symbol": "AAA", "published_at": ist("2024-06-03 18:00", "2024-06-25 18:00")})
    assert len(results_releases(recs, DAYS, merge_days=10)) == 2
    assert len(results_releases(recs, DAYS, merge_days=20)) == 1


def test_monthly_expiry_is_the_last_thursday_or_the_trading_day_before():
    days = pd.Series(pd.bdate_range("2024-10-01", "2024-11-30"))
    days = days[days != pd.Timestamp("2024-11-28")]                    # pretend the last Thursday is a holiday
    exp = expiry_days(days, {"2016-01-01": 3})
    assert [str(d.date()) for d in exp] == ["2024-10-31", "2024-11-27"]
    exp = expiry_days(pd.Series(pd.bdate_range("2025-08-01", "2025-09-30")), {"2016-01-01": 3, "2025-09-01": 1})
    assert [str(d.date()) for d in exp] == ["2025-08-28", "2025-09-30"]


def test_days_to_expiry_counts_trading_days():
    cal = calendar_table(pd.Series(pd.bdate_range("2024-10-21", "2024-10-31")), extra_days=5).set_index("trade_date")
    assert cal.loc["2024-10-31", "days_to_expiry"] == 0
    assert cal.loc["2024-10-28", "days_to_expiry"] == 3
    assert cal["days_to_expiry"].notna().all()


def test_rolling_beta_recovers_the_slope():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 400))
    r = 1.5 * m + pd.Series(rng.normal(0, 0.002, 400))
    b = rolling_beta(r, m, 60)
    assert b.iloc[:59].isna().all()
    assert b.iloc[-1] == pytest.approx(1.5, abs=0.1)


def test_skipped_sessions_match_the_override_notes():
    cfg = synthetic.config()
    skip = skipped_sessions(cfg)
    assert pd.Timestamp("2017-10-19") in skip                          # weekday Muhurat
    assert pd.Timestamp("2020-02-01") not in skip                      # Budget Saturday: a full session
