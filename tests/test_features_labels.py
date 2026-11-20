import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES, build_features, macro_features
from src.labels.labels import make_labels
from src.labels.splits import early_stop_split, make_folds, train_mask
from tests import synthetic


def test_features_for_day_d_ignore_day_d_and_later():
    """The time rule: changing prices on day d and after leaves row d's features unchanged."""
    prices, index, macro = synthetic.market(300)
    base, _ = build_features(prices, index, macro, 60)
    d = prices["trade_date"].unique()[250]
    changed = prices.copy()
    later = changed["trade_date"] >= d
    for c in ("open", "high", "low", "close"):
        changed.loc[later, c] *= 1.3
    changed.loc[later, "volume"] *= 5
    idx_changed = index.copy()
    idx_changed.loc[idx_changed["trade_date"] >= d, "close"] *= 1.3
    after, _ = build_features(changed, idx_changed, macro, 60)
    row = (base["trade_date"] == d)
    pd.testing.assert_frame_equal(base.loc[row, FEATURES].reset_index(drop=True),
                                  after.loc[row, FEATURES].reset_index(drop=True))
    nxt = base["trade_date"] == pd.Timestamp(d) + pd.offsets.BDay(1)
    assert not np.allclose(base.loc[nxt, "ret_1"], after.loc[nxt, "ret_1"])


def test_macro_value_is_used_only_from_known_from():
    days = pd.Series(pd.bdate_range("2024-01-01", periods=3))
    macro = pd.DataFrame({"series_id": "DGS10", "obs_date": days,
                          "value": [1.0, 2.0, 3.0],
                          # the 2nd value becomes known at 09:00 IST on day 3: after the 08:30 forecast
                          "known_from": pd.to_datetime(["2024-01-01 20:00", "2024-01-03 09:00",
                                                        "2024-01-04 00:00"]).tz_localize("Asia/Kolkata")})
    out = macro_features(macro, days, "08:30").set_index("trade_date")
    assert out.loc[pd.Timestamp("2024-01-02"), "us10y"] == 1.0
    assert out.loc[pd.Timestamp("2024-01-03"), "us10y"] == 1.0


def test_labels_open_entry_and_rms_volatility():
    days = pd.bdate_range("2024-01-01", periods=4)
    p = pd.DataFrame({"symbol": "AAA", "trade_date": days, "open": [10.0, 11.0, 12.0, 13.0],
                      "close": [10.0, 12.0, 12.0, 15.0]})
    lab = make_labels(p, [1, 2], "open")
    assert lab.loc[1, "fwd_ret_1"] == pytest.approx(np.log(12 / 11))
    assert lab.loc[1, "fwd_ret_2"] == pytest.approx(np.log(12 / 11))
    r1, r2 = np.log(12 / 10), np.log(12 / 12)
    assert lab.loc[1, "fwd_vol_2"] == pytest.approx(np.sqrt((r1 ** 2 + r2 ** 2) / 2))
    assert lab.loc[1, "end_2"] == days[2]
    assert np.isnan(lab.loc[3, "fwd_ret_2"]) and np.isnan(lab.loc[3, "up_2"])
    close_lab = make_labels(p, [1], "close")
    assert close_lab.loc[1, "fwd_ret_1"] == pytest.approx(np.log(12 / 10))


def test_walk_forward_gap_rule():
    panel = synthetic.panel()
    folds = make_folds(panel.dates, "2020-03-01", "2020-06-30", "monthly")
    assert [f.start.month for f in folds] == [3, 4, 5, 6]
    for f in folds:
        m = train_mask(panel.df, f.start, panel.max_h)
        assert m.any()
        assert (pd.to_datetime(panel.df.loc[m, "end_10"]) < f.start).all()
    m = train_mask(panel.df, folds[0].start, panel.max_h)
    inner_train, inner_val = early_stop_split(panel.df, m, 20, panel.max_h)
    cut = panel.df.loc[inner_val, "trade_date"].min()
    assert (pd.to_datetime(panel.df.loc[inner_train, "end_10"]) < cut).all()
    assert not (inner_train & inner_val).any()
