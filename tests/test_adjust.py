import numpy as np
import pandas as pd
import pytest

from src.store.adjust import (
    adjust_prices,
    build_events,
    jump_report,
    part_factor,
    rights_terms,
    split_bonus_parts,
)


def prices(symbol="AAA", n=6, jump_at=3, ratio=0.5):
    days = pd.bdate_range("2024-01-01", periods=n)
    close = np.array([100.0] * n)
    close[jump_at:] *= ratio                      # what an unadjusted bonus 1:1 looks like
    return pd.DataFrame({"symbol": symbol, "trade_date": days, "open": close, "high": close, "low": close,
                         "close": close, "volume": 1000.0, "turnover": 1.0})


def actions(subject, ca_type, ex="2024-01-04", face=1.0):
    return pd.DataFrame({"symbol": ["AAA"], "ex_date": [pd.Timestamp(ex)], "subject": [subject],
                         "ca_type": [ca_type], "face_value": [face]})


NO_MANUAL = pd.DataFrame(columns=["symbol", "ex_date", "factor", "kind", "note"])


def test_combined_bonus_and_split_subject_multiplies_both():
    parts = split_bonus_parts("Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/-")
    assert np.prod([part_factor(p) for p in parts]) == pytest.approx(0.1)


def test_rights_terms():
    assert rights_terms("Rights 1:15 @ Premium Rs 1247") == [(1, 15, 1247.0)]
    assert rights_terms("Rights - 4:25 Fully Paid Up Shares @ Premium Rs 500/- Per Share / "
                        "2:25 Partly Paid Up Shares @ Premium Rs 605/- Per Share") == [(4, 25, 500.0), (2, 25, 605.0)]
    assert rights_terms("Dividend Rs 5") == []


def test_bonus_removes_the_jump_and_scales_volume():
    p = prices()
    ev = build_events(p, actions("Bonus 1:1", "bonus"), NO_MANUAL)
    adj = adjust_prices(p, ev)
    assert adj["close"].pct_change().abs().max() == pytest.approx(0.0)
    assert adj.loc[0, "volume"] == pytest.approx(2000.0)
    assert adj.loc[5, "raw_close"] == pytest.approx(50.0)


def test_rights_factor_is_terp_over_close_before():
    p = prices(ratio=1.0)
    ev = build_events(p, actions("Rights 1:4 @ Premium Rs 49", "rights", face=1.0), NO_MANUAL)
    terp = (4 * 100 + 1 * 50) / 5
    assert ev.loc[0, "factor"] == pytest.approx(terp / 100)


def test_rights_with_two_classes_counts_both():
    p = prices(ratio=1.0)
    subject = ("Rights - 1:4 Fully Paid Up Shares @ Premium Rs 49/- Per Share / "
               "1:4 Partly Paid Up Shares @ Premium Rs 59/- Per Share")
    ev = build_events(p, actions(subject, "rights", face=1.0), NO_MANUAL)
    assert ev.loc[0, "factor"] == pytest.approx((4 * 100 + 50 + 60) / 6 / 100)


def test_rights_issue_price_uses_the_face_value_before_a_later_split():
    """NSE's face_value is today's: a Rs 10 share split to Re 1 later shows face 1 on the old rights row."""
    p = prices(ratio=1.0)
    acts = pd.concat([actions("Rights 1:4 @ Premium Rs 40", "rights", face=1.0),
                      actions("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share",
                              "split", ex="2024-01-05", face=1.0)], ignore_index=True)
    ev = build_events(p, acts, NO_MANUAL)
    rights = ev[ev["kind"] == "rights"].iloc[0]
    assert rights["factor"] == pytest.approx((4 * 100 + 1 * 50) / 5 / 100)       # issue price 10 + 40


def test_manual_row_replaces_computed_factor_and_adds_missing_events():
    p = prices()
    manual = pd.DataFrame({"symbol": ["AAA"], "ex_date": ["2024-01-04"], "factor": [0.5], "kind": ["split"],
                           "note": ["hand-checked"]})
    ev = build_events(p, actions("Bonus 1:1", "bonus"), manual)
    assert len(ev) == 1 and ev.loc[0, "method"] == "manual"
    ev2 = build_events(p, actions("Dividend Rs 5", "dividend"), manual)
    assert len(ev2) == 1 and ev2.loc[0, "factor"] == 0.5


def test_jump_report_flags_an_unadjusted_event_day():
    p = prices()
    ev = pd.DataFrame({"symbol": ["AAA"], "ex_date": [pd.Timestamp("2024-01-04")],
                       "trade_date": [pd.Timestamp("2024-01-04")], "kind": ["bonus"], "factor": [np.nan],
                       "method": ["unparsed"], "detail": [""]})
    adj = adjust_prices(p, ev)
    index_close = pd.Series(1000.0, index=p["trade_date"])
    rep = jump_report(adj, ev, index_close, 0.2)
    assert len(rep) == 1 and bool(rep.loc[0, "on_event_day"])
