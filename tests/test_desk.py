import copy
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agents.context import RunData
from src.api.app import create_app
from src.backtest.engine import Book, simulate
from src.catalysts.calendar import CatalystCalendar
from src.config import ROOT, Config
from src.contracts import Signal
from src.desk import service, views
from src.desk.data import DeskData, forecast_band
from src.desk.store import Store
from src.features.build import build_features
from src.features.setups import detect
from src.store.pit import AnnouncementIndex
from tests import synthetic

# ---------------------------------------------------------------- the engine, one day at a time

def test_a_book_stepped_day_by_day_with_saved_state_matches_simulate():
    prices = synthetic.market(140)[0]
    p = prices.sort_values(["symbol", "trade_date"]).assign(prev=lambda d: d.groupby("symbol")["close"].shift(1))
    days = sorted(prices["trade_date"].unique())
    signals = []
    for i in range(30, 120, 4):
        for k, sym in enumerate(synthetic.SYMBOLS):
            prev = float(p[(p["symbol"] == sym) & (p["trade_date"] == days[i])]["prev"].iloc[0])
            signals.append(Signal(symbol=sym, signal_date=pd.Timestamp(days[i]).date(), stop=prev * 0.97,
                                  target=prev * 1.05, max_hold_days=3 + k, source="t", confidence=0.5 + 0.01 * k))
    journal, equity = simulate(signals, prices)

    by_day = {d: g.set_index("symbol") for d, g in prices.groupby("trade_date")}
    todays = {}
    for s in signals:
        todays.setdefault(pd.Timestamp(s.signal_date), []).append(s)
    state, trades, curve = Book(1_000_000).state(), [], []
    for d in [x for x in days if x >= days[30]]:
        book = Book.from_state(json.loads(json.dumps(state)))          # saved and reloaded every day, like the desk
        res = book.session(d, by_day[d], todays.get(pd.Timestamp(d), []))
        trades += res.trades
        curve.append(res.equity)
        state = book.state()
    trades += Book.from_state(state).mark_open(by_day[days[-1]])
    stepped = pd.DataFrame([t.model_dump() for t in trades])
    pd.testing.assert_frame_equal(stepped, journal)
    pd.testing.assert_frame_equal(pd.DataFrame(curve), equity)


def test_the_forecast_band_widens_with_the_horizon():
    band = forecast_band(100.0, 5, {1: 0.01, 5: 0.012}, {5: 0.01}, None)
    assert [b["t"] for b in band] == [0, 1, 2, 3, 4, 5]
    assert band[0]["lo80"] == band[0]["hi80"] == 100.0
    widths = [b["hi80"] - b["lo80"] for b in band]
    assert widths == sorted(widths) and band[5]["mid"] == pytest.approx(101.0)


# ---------------------------------------------------------------- the desk on a synthetic market

@pytest.fixture()
def desk_world(tmp_path):
    raw = copy.deepcopy(Config.load().raw)
    raw["paths"]["data"] = str(tmp_path / "data")
    raw["llm"] = {"provider": "offline"}
    raw["agents"] = {"min_planner_score": 0, "max_stocks_per_day": 3}
    cfg = Config(raw, ROOT)
    prices, index, macro = synthetic.market(320)
    feats, cal = build_features(prices, index, macro, 60)
    p = prices.sort_values(["symbol", "trade_date"])
    feats = feats.merge(p.assign(ref_close=p.groupby("symbol")["close"].shift(1))[["symbol", "trade_date", "ref_close"]],
                        on=["symbol", "trade_date"])
    feats = feats[feats["ready"]].assign(setup_type=lambda d: detect(d)).reset_index(drop=True)
    ann = pd.DataFrame({"doc_id": ["1"], "symbol": ["AAA"], "category": ["Press Release"], "summary": ["new order"],
                        "published_at": [pd.Timestamp(feats["trade_date"].iloc[-30]).tz_localize("UTC")],
                        "is_important": [True], "attachment_url": [None]})
    releases = pd.DataFrame({"symbol": pd.Series(dtype=str), "day0": pd.Series(dtype="datetime64[ns]")})
    fc = pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "p_up", "exp_return", "vol_forecast", "model_version"])
    run = RunData(cfg, feats, cal, AnnouncementIndex(ann), CatalystCalendar(cal, releases), fc, "tft")
    window = (run.days[-40].date(), run.days[-1].date())
    chart = prices.assign(adj_factor=1.0)[["symbol", "trade_date", "open", "high", "low", "close", "volume", "adj_factor"]]
    vol = pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "vol_forecast"])
    data = DeskData(cfg, run, prices[["symbol", "trade_date", "open", "high", "low", "close"]], chart, vol,
                    {s: f"{s} Ltd" for s in synthetic.SYMBOLS}, window)
    return cfg, data, Store(tmp_path / "desk.sqlite")


def force_levels(store, desk_id, symbol, accepted):
    """Make a plan fill for sure at the next open (wide stop and target), accepted by the agents or not."""
    with store.tx() as con:
        con.execute("UPDATE plans SET stop = 0.01, target = 1e9, accepted = ?, direction = 'long', verdict = 'accept' "
                    "WHERE desk_id = ? AND symbol = ?", (int(accepted), desk_id, symbol))


def cash_from_orders(store, desk_id, capital):
    cash = capital
    for o in store.rows("SELECT * FROM orders WHERE desk_id = ? AND status = 'filled'", (desk_id,)):
        value = o["quantity"] * o["price"]
        cash += -(value + o["costs"]) if o["side"] == "buy" else value - o["costs"]
    return cash


def test_the_desk_plans_each_morning_and_shows_nothing_after_the_last_close(desk_world):
    cfg, data, store = desk_world
    did = service.create_desk(store, data, cfg, "t", data.window[0])
    service.run_morning(store, data, cfg, did)
    desk = service.get_desk(store, did)
    assert desk["phase"] == "morning" and desk["current_date"] == str(data.window[0])
    rows = views.watchlist(store, data, desk)["rows"]
    assert rows and all(r["plan"] is not None for r in rows)
    detail = views.stock_detail(store, data, desk, rows[0]["symbol"])
    assert detail["chart"]["bars"][-1]["d"] < desk["current_date"]            # D's own bar is not out yet
    assert detail["visible_through"] < desk["current_date"]
    assert store.scalar("SELECT COUNT(*) FROM trace_steps WHERE desk_id = ?", (did,)) > 0
    service.run_session(store, data, cfg, did)
    with pytest.raises(service.DeskError):
        service.run_session(store, data, cfg, did)                            # one session per morning
    after = views.stock_detail(store, data, service.get_desk(store, did), rows[0]["symbol"])
    assert after["chart"]["bars"][-1]["d"] == desk["current_date"]            # after the session, D's bar is in


def test_removed_stocks_are_not_traded_and_people_can_buy_and_exit(desk_world):
    cfg, data, store = desk_world
    did = service.create_desk(store, data, cfg, "t", data.window[0])
    service.run_morning(store, data, cfg, did)
    syms = [r["symbol"] for r in views.watchlist(store, data, service.get_desk(store, did))["rows"]]
    assert len(syms) == 3
    keep, drop, mine = syms
    force_levels(store, did, keep, accepted=True)
    force_levels(store, did, drop, accepted=True)
    force_levels(store, did, mine, accepted=False)
    service.remove_from_watchlist(store, did, drop)
    service.place_order(store, did, mine, "buy")
    with pytest.raises(service.DeskError):
        service.place_order(store, did, keep, "buy")                         # the agents' plan already buys it
    service.run_session(store, data, cfg, did)
    entries = {o["symbol"]: o["source"] for o in store.rows(
        "SELECT symbol, source FROM orders WHERE desk_id = ? AND kind = 'entry'", (did,))}
    assert entries == {keep: "agents", mine: "manual"}

    service.run_morning(store, data, cfg, did)                                # next day
    desk = service.get_desk(store, did)
    assert mine in {p["signal"]["symbol"] for p in desk["book"]["positions"]}
    service.place_order(store, did, mine, "exit")
    service.advance(store, data, cfg, did, 1)
    exits = store.rows("SELECT exit_reason, source FROM trades WHERE desk_id = ? AND symbol = ?", (did, mine))
    assert [(e["exit_reason"], e["source"]) for e in exits] == [("manual", "manual")]


def test_advancing_keeps_cash_equal_to_the_orders_and_stops_at_the_end(desk_world):
    cfg, data, store = desk_world
    start = data.days[-6].date()
    did = service.create_desk(store, data, cfg, "t", start)
    service.run_morning(store, data, cfg, did)
    for r in views.watchlist(store, data, service.get_desk(store, did))["rows"]:
        force_levels(store, did, r["symbol"], accepted=True)
    n = service.advance(store, data, cfg, did, 20)
    desk = service.get_desk(store, did)
    assert n == 6 and desk["phase"] == "finished"
    assert store.scalar("SELECT COUNT(*) FROM equity WHERE desk_id = ?", (did,)) == 6
    assert desk["book"]["cash"] == pytest.approx(cash_from_orders(store, did, desk["capital"]), abs=0.01)
    summary = views.desk_summary(store, data, desk)
    assert summary["days_done"] == summary["days_total"] == 6


# ---------------------------------------------------------------- the API

def run_job(app, job_id):
    app.state.worker.execute(app.state.store.row("SELECT * FROM jobs WHERE id = ?", (job_id,)))
    job = app.state.store.row("SELECT * FROM jobs WHERE id = ?", (job_id,))
    assert job["status"] == "done", job["error"]


def test_the_api_runs_a_desk_end_to_end(desk_world):
    cfg, data, store = desk_world
    app = create_app(cfg, store, data, run_worker=False)
    with TestClient(app) as client:
        meta = client.get("/api/meta").json()
        assert meta["providers"][0] == {"provider": "offline", "model": None, "ready": True,
                                        "label": "Rule-based agents (no LLM)"}
        r = client.post("/api/desks", json={"name": "api", "start_date": str(data.window[0])})
        assert r.status_code == 201
        desk_id = r.json()["desk"]["id"]
        run_job(app, r.json()["job_id"])
        assert client.get(f"/api/desks/{desk_id}").json()["phase"] == "morning"
        assert client.post("/api/desks", json={"name": "api", "start_date": str(data.window[0])}).status_code == 400
        rows = client.get(f"/api/desks/{desk_id}/watchlist").json()["rows"]
        sym = rows[0]["symbol"]
        detail = client.get(f"/api/desks/{desk_id}/stocks/{sym}").json()
        assert detail["plan"]["symbol"] == sym and detail["checks"]["pullback_uptrend"]
        day = detail["trade_date"]
        assert client.get(f"/api/desks/{desk_id}/trace", params={"date": day, "symbol": sym}).json()
        assert client.delete(f"/api/desks/{desk_id}/watchlist/{sym}").json()["removed"]
        assert client.post(f"/api/desks/{desk_id}/watchlist", json={"symbol": sym}).json()["restored"]

        job = client.post(f"/api/desks/{desk_id}/advance", json={"days": 2}).json()["job_id"]
        assert client.post(f"/api/desks/{desk_id}/advance", json={"days": 1}).status_code == 400   # busy
        run_job(app, job)
        summary = client.get(f"/api/desks/{desk_id}").json()
        assert summary["days_done"] == 2 and summary["phase"] == "morning"
        for path in ("book", "trades", "equity", "days", "events"):
            assert client.get(f"/api/desks/{desk_id}/{path}").status_code == 200
        assert len(client.get(f"/api/desks/{desk_id}/equity").json()) == 2
        assert client.get("/api/desks/999").status_code == 404
        page = client.get("/")
        assert page.status_code == 200 and "Trading with AI" in page.text
