from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import event_study as E

DAYS = pd.bdate_range("2024-01-01", periods=30)


def _ar():
    ar = pd.DataFrame(0.0, index=DAYS, columns=["A", "B"])
    ar.loc[DAYS[10], "A"] = 0.05        # the event reaction, on A's day 0
    ar.loc[DAYS[12], "A"] = 0.01
    ar.loc[DAYS[20], "B"] = np.nan      # a dropped ex-date
    return ar


def test_event_days_follow_the_time_rule():
    ts = pd.Series(pd.to_datetime(["2024-01-15 08:00", "2024-01-15 16:00", "2024-01-13 10:00"]))
    ts = ts.dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    d = E.event_days(ts, DAYS)
    assert [str(x.date()) for x in d] == ["2024-01-15", "2024-01-16", "2024-01-15"]


def test_abnormal_returns_subtract_market():
    r = pd.DataFrame({"A": [0.02, 0.01]}, index=DAYS[:2])
    assert E.abnormal_returns(r, pd.Series([0.01, 0.03], index=DAYS[:2]))["A"].tolist() == pytest.approx([0.01, -0.02])


def test_event_windows_put_day_zero_at_offset_zero():
    ev = pd.DataFrame({"symbol": ["A", "A", "B", "Z"], "event_day": [DAYS[10], DAYS[2], DAYS[18], DAYS[10]]})
    w = E.event_windows(_ar(), ev, pre=5, post=3)
    assert list(w.columns) == list(range(-5, 4))
    assert w.loc[0, 0] == 0.05 and w.loc[0, 2] == 0.01 and w.loc[0, -1] == 0.0
    assert w.loc[1, -5:-3].isna().all() and w.loc[1, -2:].notna().all()   # starts before the data
    assert np.isnan(w.loc[2, 2])                                            # B's dropped day
    assert w.loc[3].isna().all()                                            # unknown symbol
    assert list(E.complete(w).index) == [0]


def test_car_and_car_path():
    ev = pd.DataFrame({"symbol": ["A"], "event_day": [DAYS[10]]})
    w = E.event_windows(_ar(), ev, pre=2, post=3)
    assert E.car(w, 0, 2).iloc[0] == pytest.approx(0.06)
    assert E.car_path(w).iloc[0].tolist() == pytest.approx([0, 0, 0.05, 0.05, 0.06, 0.06])
    assert E.car_path(w, start=1).columns.tolist() == [1, 2, 3]


def test_bootstrap_means_resample_clusters():
    vals = pd.DataFrame({"x": [1.0, 1.0, 3.0]})
    clusters = pd.Series(["w1", "w1", "w2"])
    b = E.bootstrap_means(vals, clusters, n_boot=500, seed=1)
    # only three possible resampled means: both w1 (1.0), both w2 (3.0), one of each (5/3)
    assert set(np.round(b[:, 0], 6)) <= {1.0, 3.0, round(5 / 3, 6)}
    s = E.summarize(vals, clusters, n_boot=500, seed=1)
    assert s.loc["x", "n"] == 3 and s.loc["x", "mean"] == pytest.approx(5 / 3) and s.loc["x", "share_pos"] == 1.0
    assert s.loc["x", "lo"] <= s.loc["x", "mean"] <= s.loc["x", "hi"]


def test_difference_interval_excludes_zero_for_a_clear_effect():
    rng = np.random.default_rng(0)
    ev = pd.DataFrame({"x": rng.normal(0.02, 0.01, 400)})
    ctl = pd.DataFrame({"x": rng.normal(0.0, 0.01, 400)})
    cl = pd.Series(np.arange(400) // 4)
    d = E.difference(ev, cl, ctl, cl, n_boot=500)
    assert d.loc["x", "lo"] > 0


def test_control_sample_matches_counts_and_has_full_windows():
    ar = _ar()
    ev = pd.DataFrame({"symbol": ["A", "A", "B"], "event_day": [DAYS[10], DAYS[12], DAYS[10]]})
    ctl = E.control_sample(ar, ev, pre=2, post=2, seed=3)
    assert ctl.symbol.value_counts().to_dict() == {"A": 2, "B": 1}
    w = E.event_windows(ar, ctl, pre=2, post=2)
    assert w.notna().all().all()
    assert E.control_sample(ar, ev, pre=2, post=2, seed=3).equals(ctl)   # reproducible


def test_week_cluster():
    c = E.week_cluster(pd.Series(pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-08"])))
    assert c.iloc[0] == c.iloc[1] != c.iloc[2]
