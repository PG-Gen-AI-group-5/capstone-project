import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agents.context import RunData, with_ref_close
from src.api.app import create_app
from src.catalysts.calendar import CatalystCalendar
from src.config import ROOT, Config
from src.desk import live, views
from src.desk.data import DeskData
from src.desk.service import DeskError
from src.desk.store import Store
from src.features.build import FEATURES, add_placeholder_day, build_features
from src.features.setups import detect
from src.store.pit import AnnouncementIndex
from tests import synthetic

# ---------------------------------------------------------------- the feature row for a day without prices

def test_the_row_built_before_a_days_prices_exist_equals_the_row_built_after():
    prices, index, macro = synthetic.market(320)
    day = prices["trade_date"].max()
    full, _ = build_features(prices, index, macro, 60)
    early_p, early_i = add_placeholder_day(prices[prices["trade_date"] < day], index[index["trade_date"] < day], day)
    early, _ = build_features(early_p, early_i, macro, 60)
    a = full[full["trade_date"] == day].set_index("symbol")[FEATURES + ["history", "ready"]]
    b = early[early["trade_date"] == day].set_index("symbol")[FEATURES + ["history", "ready"]]
    assert len(a) == 3
    pd.testing.assert_frame_equal(a, b)


def test_a_placeholder_day_must_come_after_the_last_prices():
    prices, index, _ = synthetic.market(80)
    with pytest.raises(ValueError):
        add_placeholder_day(prices, index, prices["trade_date"].max())


# ---------------------------------------------------------------- a live desk on a synthetic market

@pytest.fixture()
def live_world(tmp_path):
    raw = copy.deepcopy(Config.load().raw)
    raw["paths"]["data"] = str(tmp_path / "data")
    raw["llm"] = {"provider": "offline"}
    raw["agents"] = {"min_planner_score": 0, "max_stocks_per_day": 3}
    raw["live"] = {**raw.get("live", {}), "stale_ok": True}          # synthetic dates are in the past
    cfg = Config(raw, ROOT)
    prices, index, macro = synthetic.market(320)
    day = prices["trade_date"].max()                                   # the "next trading day": no prices yet
    seen_p, seen_i = prices[prices["trade_date"] < day], index[index["trade_date"] < day]
    feats, cal = build_features(*add_placeholder_day(seen_p, seen_i, day), macro, 60)
    feats = with_ref_close(feats, seen_p.rename(columns={"close": "raw_close"}))
    feats = feats[feats["ready"]].assign(setup_type=lambda d: detect(d)).reset_index(drop=True)
    ann = pd.DataFrame({"doc_id": ["1"], "symbol": ["AAA"], "category": ["Press Release"], "summary": ["new order"],
                        "published_at": [pd.Timestamp(day).tz_localize("UTC") - pd.Timedelta(days=3)],
                        "is_important": [True], "attachment_url": [None]})
    releases = pd.DataFrame({"symbol": pd.Series(dtype=str), "day0": pd.Series(dtype="datetime64[ns]")})
    fc = pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "p_up", "exp_return", "vol_forecast", "model_version"])
    run = RunData(cfg, feats, cal, AnnouncementIndex(ann), CatalystCalendar(cal, releases), fc, "tft")
    chart = seen_p.assign(adj_factor=1.0)[["symbol", "trade_date", "open", "high", "low", "close", "volume", "adj_factor"]]
    vol = pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "vol_forecast"])
    data = DeskData(cfg, run, seen_p[["symbol", "trade_date", "open", "high", "low", "close"]], chart, vol,
                    {s: f"{s} Ltd" for s in synthetic.SYMBOLS}, (run.days[0].date(), pd.Timestamp(day).date()))
    return cfg, data, Store(tmp_path / "desk.sqlite"), seen_p


def test_a_live_desk_suggests_for_the_next_day_and_follows_a_holding_to_its_exit(live_world):
    cfg, data, store, seen = live_world
    did = live.create_live_desk(store, cfg, "live", 1_000_000, "offline")
    out = live.run_live_morning(store, data, cfg, did)
    desk = live.live_desk(store, did)
    assert out["trade_date"] == str(data.window[1]) and desk["phase"] == "morning"
    rows = views.watchlist(store, data, desk)["rows"]
    assert rows and all(r["plan"] is not None for r in rows)
    detail = views.stock_detail(store, data, desk, rows[0]["symbol"])
    assert detail["actions"]["bought"] and not detail["actions"]["sold"]
    assert detail["chart"]["bars"][-1]["d"] < desk["current_date"]          # the live day's own prices don't exist

    sym = rows[0]["symbol"]
    last = seen[seen["symbol"] == sym].sort_values("trade_date").iloc[-3]    # bought three sessions ago
    hid = live.add_holding(store, data, did, sym, last.trade_date, float(last.close), 100,
                           stop=float(last.close) * 0.5, target=float(last.close) * 2, max_hold_days=2)
    with pytest.raises(DeskError):
        live.add_holding(store, data, did, sym, last.trade_date, float(last.close), 10)   # one open holding a stock
    h = live.holding_rows(store, data, desk)[0]
    assert h["checks"]["days_held"] == 3 and h["checks"]["time_up"]          # 3 sessions held, planned 2
    assert h["advice"]["action"] == "time_up" and h["advice"]["urgent"]

    live.run_live_morning(store, data, cfg, did)                             # the Exit Monitor sees the holding
    assert store.scalar("SELECT action FROM exit_decisions WHERE desk_id = ? AND symbol = ?", (did, sym)) == "exit"
    summary = views.desk_summary(store, data, live.live_desk(store, did))
    assert summary["mode"] == "live" and summary["open_positions"] == 1 and summary["urgent"] == 1

    tid = live.close_holding(store, data, did, hid, data.window[1], float(last.close) * 1.02, "exit_monitor")
    trade = store.row("SELECT * FROM trades WHERE id = ?", (tid,))
    assert trade["exit_reason"] == "exit_monitor" and trade["costs"] > 0
    assert trade["net_pnl"] == pytest.approx(float(last.close) * 0.02 * 100 - trade["costs"], abs=0.01)
    assert not live.holding_rows(store, data, live.live_desk(store, did))


def test_stop_and_target_checks_find_the_first_day_touched(live_world):
    _, data, _, seen = live_world
    g = seen[seen["symbol"] == "AAA"].sort_values("trade_date").tail(5)
    entry = g.iloc[0]
    h = {"symbol": "AAA", "entry_date": str(entry.trade_date.date()), "max_hold_days": 10,
         "stop": float(g["low"].iloc[2]) + 0.001, "target": 1e9}
    c = live.price_checks(data, h, g["trade_date"].iloc[-1])
    first = g[g["low"] <= h["stop"]].iloc[0]
    assert c["stop_hit"]["date"] == str(first.trade_date.date()) and c["days_held"] == 5
    assert live.advice({**h, "entry_price": 1.0}, c, None)["action"] == "stop_hit"


def test_the_api_records_a_bought_and_sold_trade_on_a_live_desk(live_world):
    cfg, data, store, _ = live_world
    did = live.create_live_desk(store, cfg, "live", 1_000_000, "offline")
    live.run_live_morning(store, data, cfg, did)
    app = create_app(cfg, store, data, run_worker=False, live_data=data)
    with TestClient(app) as client:
        summary = client.get(f"/api/desks/{did}").json()
        assert summary["mode"] == "live" and summary["phase"] == "morning"
        sym = client.get(f"/api/desks/{did}/watchlist").json()["rows"][0]["symbol"]
        detail = client.get(f"/api/desks/{did}/stocks/{sym}").json()
        assert detail["suggested_quantity"] is None or detail["suggested_quantity"] >= 0
        r = client.post(f"/api/desks/{did}/holdings", json={"symbol": sym, "entry_date": str(data.window[1]),
                                                          "entry_price": 100.0, "quantity": 5})
        assert r.status_code == 201, r.text
        hid = r.json()["holding_id"]
        held = client.get(f"/api/desks/{did}/holdings").json()
        assert held[0]["advice"]["text"] and held[0]["symbol"] == sym
        assert client.patch(f"/api/desks/{did}/holdings/{hid}", json={"stop": 1.0}).status_code == 200
        r = client.post(f"/api/desks/{did}/holdings/{hid}/close",
                        json={"exit_date": str(data.window[1]), "exit_price": 101.0, "reason": "manual"})
        assert r.status_code == 200, r.text
        assert len(client.get(f"/api/desks/{did}/trades").json()) == 1
        assert client.post(f"/api/desks/{did}/advance", json={"days": 1}).status_code == 400   # no sessions on live
