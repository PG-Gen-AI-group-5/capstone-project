from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import macro as M

DAYS = pd.DatetimeIndex(pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"]))


def _known(obs, hhmm="05:30"):
    """known_from = the day after obs_date at hhmm IST (the FRED rule), as UTC."""
    t = pd.to_datetime(obs) + pd.Timedelta(days=1) + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    return t.tz_localize("Asia/Kolkata").tz_convert("UTC")


def _macro():
    obs = ["2024-05-31", "2024-06-03", "2024-06-04", "2024-06-05"]
    return pd.DataFrame({"series_id": "X", "obs_date": pd.to_datetime(obs), "value": [1.0, 2.0, np.nan, 4.0],
                         "known_from": [_known(o) for o in obs]})


def test_as_of_times_are_0830_ist_in_utc():
    t = M.as_of_times(DAYS[:1])
    assert str(t.iloc[0]) == "2024-06-03 03:00:00+00:00"


def test_macro_asof_uses_previous_us_close_and_skips_missing_values():
    vals, obs = M.macro_asof(_macro(), DAYS.rename("trade_date"))   # a named index must work too
    # 3 Jun uses Fri 31 May; 4 Jun uses 3 Jun; 5 Jun: the 4 Jun value is missing -> still 3 Jun; 6 Jun uses 5 Jun
    assert vals["X"].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert [str(d.date()) for d in obs["X"]] == ["2024-05-31", "2024-06-03", "2024-06-03", "2024-06-05"]


def test_macro_asof_never_uses_a_value_published_after_the_forecast():
    """Leakage check: a value whose known_from is 08:31 IST on day d must not appear on day d."""
    m = _macro()
    late = pd.DataFrame({"series_id": ["X"], "obs_date": [pd.Timestamp("2024-06-05")], "value": [999.0],
                         "known_from": [pd.Timestamp("2024-06-06 08:31", tz="Asia/Kolkata").tz_convert("UTC")]})
    m = pd.concat([m[m.obs_date != "2024-06-05"], late], ignore_index=True)
    vals, _ = M.macro_asof(m, DAYS)
    assert 999.0 not in vals["X"].tolist()
    exact = late.assign(known_from=pd.Timestamp("2024-06-06 08:30", tz="Asia/Kolkata").tz_convert("UTC"))
    vals2, _ = M.macro_asof(pd.concat([m[m.value != 999.0], exact]), DAYS)
    assert vals2["X"].iloc[-1] == 999.0      # known_from <= as_of is allowed


def test_overnight_change_is_nan_without_a_new_observation():
    vals, obs = M.macro_asof(_macro(), DAYS)
    ch = M.overnight_change(vals, obs)["X"]
    assert np.isnan(ch.iloc[0]) and ch.iloc[1] == 1.0 and np.isnan(ch.iloc[2]) and ch.iloc[3] == 2.0
    lch = M.overnight_change(vals, obs, log_series=("X",))["X"]
    assert lch.iloc[1] == pytest.approx(np.log(2.0))


def test_block_bootstrap_corr():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=1000))
    y = 0.5 * x + rng.normal(size=1000)
    out = M.block_bootstrap_corr(x, y, block=20, n_boot=500)
    assert out["lo"] < out["corr"] < out["hi"] and out["lo"] > 0.3
    assert np.isnan(M.block_bootstrap_corr(x[:10], y[:10])["corr"])
