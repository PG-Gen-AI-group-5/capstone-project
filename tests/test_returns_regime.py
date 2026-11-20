from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import returns as R
from src.config import ROOT, Config
from src.features import regime as G

DAYS = pd.bdate_range("2024-01-01", periods=6)


def _frame(values, cols=("A",)):
    return pd.DataFrame(np.array(values, dtype=float).reshape(len(DAYS), len(cols)), index=DAYS, columns=list(cols))


CLOSE = _frame([100, 110, 121, 60.5, 66.55, 73.205])   # +10% a day, except a 1:2 split on day 3
OPEN = _frame([99, 105, 115, 58, 63, 70])


def _mask():
    ca = pd.DataFrame({"symbol": ["A", "Z"], "ex_date": [DAYS[3], DAYS[1]], "ca_type": ["split", "split"]})
    return R.ex_date_mask(DAYS, ["A"], ca)


def test_to_wide():
    long = pd.DataFrame({"symbol": ["A", "B", "A"], "trade_date": ["2024-01-02", "2024-01-02", "2024-01-01"],
                         "close": [2.0, 3.0, 1.0]})
    w = R.to_wide(long, "close")
    assert list(w.columns) == ["A", "B"] and w.loc["2024-01-01", "A"] == 1.0 and np.isnan(w.loc["2024-01-01", "B"])


def test_ex_date_mask_ignores_unknown_symbols():
    m = _mask()
    assert m["A"].tolist() == [False, False, False, True, False, False]


def test_log_returns_mask_ex_dates_and_first_day():
    r = R.log_returns(CLOSE, _mask())["A"]
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[3])
    assert r.iloc[1] == pytest.approx(np.log(1.1)) and r.iloc[4] == pytest.approx(np.log(1.1))


def test_forward_sum_never_spans_a_masked_day():
    r = R.log_returns(CLOSE, _mask())
    f2 = R.forward_sum(r, 2)["A"]
    # day1: r1 + r2 = 2*log(1.1); day2: r2 + r3 where r3 is masked -> NaN; day3: masked -> NaN
    assert f2.iloc[1] == pytest.approx(2 * np.log(1.1))
    assert np.isnan(f2.iloc[2]) and np.isnan(f2.iloc[3])
    assert f2.iloc[4] == pytest.approx(2 * np.log(1.1))
    assert np.isnan(f2.iloc[5])  # window runs past the data


def test_label_returns_close_and_open_definitions():
    m = _mask()
    close_kind = R.label_returns(CLOSE, OPEN, m, 2, "close")["A"]
    open_kind = R.label_returns(CLOSE, OPEN, m, 2, "open")["A"]
    # forecast day 1: close(0)=100 -> close(2)=121 ; open(1)=105 -> close(2)=121
    assert close_kind.iloc[1] == pytest.approx(np.log(121 / 100))
    assert open_kind.iloc[1] == pytest.approx(np.log(121 / 105))
    # open-kind h=2 from day 3 (ex-date): open(3) -> close(4) does not cross the split, so it is valid
    assert open_kind.iloc[3] == pytest.approx(np.log(66.55 / 58))
    assert R.label_returns(CLOSE, OPEN, m, 1, "open")["A"].iloc[3] == pytest.approx(np.log(60.5 / 58))
    with pytest.raises(ValueError):
        R.label_returns(CLOSE, OPEN, m, 2, "mid")


def test_gap_intraday_add_up_to_daily_return():
    gap, intra = R.gap_intraday(OPEN, CLOSE, _mask())
    r = R.log_returns(CLOSE, _mask())
    assert (gap + intra)["A"].iloc[1] == pytest.approx(r["A"].iloc[1])
    assert np.isnan(gap["A"].iloc[3]) and not np.isnan(intra["A"].iloc[3])


def test_realised_and_forward_vol_alignment():
    r = _frame([0.0, 0.01, -0.01, 0.02, -0.02, 0.0])
    back = R.realised_vol(r, 3)["A"]
    fwd = R.forward_vol(r, 3)["A"]
    assert back.iloc[3] == pytest.approx(np.std([0.01, -0.01, 0.02], ddof=1))
    assert fwd.iloc[1] == pytest.approx(back.iloc[3])  # same three days, attached to the first one
    assert np.isnan(fwd.iloc[4])


def test_every_nth_and_moments():
    assert R.every_nth(_frame(range(6)), 2)["A"].tolist() == [0, 2, 4]
    m = R.moments(np.random.default_rng(0).normal(0, 1, 50_000))
    assert abs(m["skew"]) < 0.05 and abs(m["excess_kurt"]) < 0.1 and m["normal_beyond_3sd"] == pytest.approx(0.0027, abs=1e-4)


def test_beta_corr():
    rng = np.random.default_rng(1)
    mkt = pd.Series(rng.normal(0, 0.01, 1000))
    r = pd.DataFrame({"X": 1.5 * mkt + rng.normal(0, 0.001, 1000), "Short": mkt.where(mkt.index < 10)})
    out = R.beta_corr(r, mkt)
    assert out.loc["X", "beta"] == pytest.approx(1.5, abs=0.02) and "Short" not in out.index


# ---- regimes --------------------------------------------------------------------------------------

def test_trend_regime_is_lagged_one_day():
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series([10, 10, 10, 20, 20, 1, 1, 1, 1, 1], index=idx, dtype=float)
    lab = G.trend_regime(close, span=3)
    assert lab.iloc[:3].isna().all()                    # EMA needs 3 values, then one more day of lag
    assert lab.iloc[4] == "above_ema"                   # day 3 (20 > EMA) is known on day 4
    assert lab.iloc[6] == "below_ema"                   # day 5 (1 < EMA) is known on day 6


def test_regime_labels_never_use_the_same_days_value():
    """Leakage check: changing day d's close or VIX must not change day d's label."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    vix = pd.Series(rng.uniform(10, 30, 300), index=idx)
    d = idx[250]
    base_t, base_v = G.trend_regime(close, 200), G.vix_regime(vix, [15, 20])
    c2, v2 = close.copy(), vix.copy()
    c2[d] *= 10
    v2[d] = 99
    assert G.trend_regime(c2, 200)[d] == base_t[d]
    assert G.vix_regime(v2, [15, 20])[d] == base_v[d]
    assert G.vix_regime(v2, [15, 20])[idx[251]] == "high"   # it shows up the next day


def test_vix_bands_use_only_the_fitting_period():
    idx = pd.bdate_range("2022-12-26", periods=10)
    vix = pd.Series([10, 20, 30, 10, 20, 999, 999, 999, 999, 999], index=idx, dtype=float)
    bands = G.vix_bands(vix, "2022-12-01", "2022-12-31")
    assert max(bands) < 30
    with pytest.raises(ValueError):
        G.vix_bands(vix, "2030-01-01", "2030-12-31")


def test_vix_regime_buckets():
    idx = pd.bdate_range("2020-01-01", periods=5)
    lab = G.vix_regime(pd.Series([12, 15, 19.9, 20, np.nan], index=idx), [15, 20])
    assert lab.tolist()[1:] == ["low", "mid", "mid", "high"]
    with pytest.raises(ValueError):
        G.vix_regime(pd.Series([1.0]), [15, 20], labels=("a", "b"))


def test_regime_table_reads_config():
    cfg = Config.load(ROOT / "config.yaml")
    idx = pd.bdate_range("2016-01-01", "2023-12-29")
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    vix = pd.Series(rng.uniform(10, 30, len(idx)), index=idx)
    tab = G.regime_table(close, vix, cfg["regime"])
    assert set(tab["vix"].dropna()) == {"low", "mid", "high"} and len(tab.attrs["vix_bands"]) == 2
    train = tab.loc[:"2022-12-31", "vix"].dropna()
    assert train.value_counts(normalize=True).min() > 0.3     # terciles on train
