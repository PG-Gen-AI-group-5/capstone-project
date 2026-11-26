"""Live desks: each trading day, suggestions for the next session and advice on the trades you took.

Every weekday at 08:00 IST (and whenever you press "Update and suggest"), `daily`:
1. downloads the new prices and filings and rebuilds the live tables (src/live/update.py),
2. predicts the next trading day D with the live TFT and LSTM (src/live/models.py),
3. for each live desk runs the agents for D (src/graph/build.py): the Exit Monitor on your holdings, then the
   Planner and the debate, forecast and critique per stock. Nothing after D-1's close is visible.

You report a trade you took ("I bought": your date, price and quantity, with the plan's stop, target and holding
period). Until you report the exit, every morning checks it: was the stop or target touched since the last
close you saw, is the holding period over, and what does the Exit Monitor say (hold, raise the stop, exit at
the open). Closing it writes a trade with estimated Indian delivery costs, for the history and analysis pages.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.base import AgentContext
from src.agents.context import RunData
from src.backtest.costs import order_cost
from src.backtest.engine import PORTFOLIO_DEFAULTS, Book
from src.backtest.run import raw_bars
from src.config import Config
from src.desk.data import DeskData, desk_settings, load_vol
from src.desk.service import (
    PROVIDERS,
    DbTracer,
    DeskError,
    agent_config,
    event,
    get_desk,
    llm_usage,
    provider_status,
    save_result,
)
from src.desk.store import Store, insert, now, update
from src.graph.build import day_graph
from src.live import update as live_update
from src.live.env import live_config, live_settings
from src.llm.client import LLMClient
from src.store.db import read_table, universe
from src.timeutil import today_ist

Progress = Callable[[str], None]


# ---------------------------------------------------------------- the live data, loaded when it changes

_cache: dict = {"key": None, "data": None}
_lock = threading.Lock()


def live_data(cfg: Config) -> DeskData:
    """DeskData over the live tables, reloaded after each live build."""
    info = live_update.built(cfg)
    if not info:
        raise DeskError("No live data yet. Press 'Update and suggest' (or run: python -m src.live daily).")
    lcfg = live_config(cfg)
    forecasts = sorted((lcfg.derived_dir / "forecasts").glob("*/live.parquet"))
    key = (info["built_at"], tuple((p.name, p.parent.name, p.stat().st_mtime_ns) for p in forecasts))
    with _lock:                       # reload after a rebuild and after new forecasts (written after the build)
        if _cache["key"] == key:
            return _cache["data"]
        day = pd.Timestamp(info["day"])
        s = desk_settings(cfg)
        run = RunData.load(lcfg, day - pd.Timedelta(days=int(s["history_days"])), day)
        chart = read_table(lcfg, "prices_adj", layer="derived",
                           columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "adj_factor"])
        names = dict(zip(*[universe(cfg)[c] for c in ("symbol", "company_name")]))
        data = DeskData(lcfg, run, raw_bars(lcfg), chart, load_vol(lcfg, s["vol_model"], ["live"]), names,
                        (run.days[0].date(), day.date()))
        _cache.update(key=key, data=data)
        return data


def live_day(data: DeskData) -> date:
    return data.window[1]


# ---------------------------------------------------------------- desks

def create_live_desk(store: Store, cfg: Config, name: str, capital: float | None = None, provider: str = "offline") -> int:
    name = (name or "").strip()
    if not name:
        raise DeskError("Give the desk a name.")
    if store.scalar("SELECT 1 FROM desks WHERE name = ?", (name,)):
        raise DeskError(f"A desk called {name!r} already exists.")
    if provider not in PROVIDERS:
        raise DeskError(f"Unknown LLM provider {provider!r}.")
    status = {p["provider"]: p for p in provider_status(cfg)}[provider]
    if not status["ready"]:
        raise DeskError(f"The {provider} provider is not configured (see .env.example).")
    capital = float(capital or desk_settings(cfg)["capital"])
    portfolio = {**PORTFOLIO_DEFAULTS, **((cfg.get("backtest") or {}).get("portfolio") or {}), "capital": capital}
    settings = {"portfolio": portfolio, "costs": cfg.get("costs") or {}, "agents": cfg.get("agents") or {},
                "setups": cfg.get("setups") or {}}
    stamp, today = now(), str(today_ist())
    with store.tx() as con:
        return insert(con, "desks", {
            "name": name, "mode": "live", "provider": provider, "model": status["model"], "start_date": today,
            "end_date": "9999-12-31", "capital": capital, "settings": settings, "current_date": today, "phase": "new",
            "book": Book(capital, portfolio).state(), "created_at": stamp, "updated_at": stamp})


def live_desk(store: Store, desk_id: int) -> dict:
    desk = get_desk(store, desk_id)
    if desk.get("mode") != "live":
        raise DeskError("That is a replay desk; holdings and live suggestions belong to live desks.")
    return desk


# ---------------------------------------------------------------- holdings and their daily checks

def open_holdings(store: Store, desk_id: int) -> list[dict]:
    return store.rows("SELECT * FROM holdings WHERE desk_id = ? AND status = 'open' ORDER BY entry_date, id", (desk_id,))


def price_checks(data: DeskData, h: dict, through) -> dict:
    """What the prices since the entry say about a holding, up to the close of `through`: the first day the stop
    or target was touched, and the trading days held."""
    out = {"stop_hit": None, "target_hit": None, "days_held": 0, "time_up": False, "through": None}
    if through is None:
        return out
    b = data.bars[(data.bars["symbol"] == h["symbol"]) & (data.bars["trade_date"] >= pd.Timestamp(h["entry_date"]))
                  & (data.bars["trade_date"] <= pd.Timestamp(through))].sort_values("trade_date")
    out["through"] = str(pd.Timestamp(through).date())
    out["days_held"] = len(b)
    out["time_up"] = len(b) >= int(h["max_hold_days"])
    stop = b[b["low"] <= h["stop"]]
    target = b[b["high"] >= h["target"]]
    if len(stop):
        r = stop.iloc[0]
        out["stop_hit"] = {"date": str(r.trade_date.date()), "low": round(float(r.low), 2),
                           "open": round(float(r.open), 2)}
    if len(target):
        r = target.iloc[0]
        out["target_hit"] = {"date": str(r.trade_date.date()), "high": round(float(r.high), 2)}
    return out


def advice(h: dict, checks: dict, decision: dict | None) -> dict:
    """One line of advice per holding, most urgent first."""
    s, t = checks["stop_hit"], checks["target_hit"]
    if s and (not t or s["date"] <= t["date"]):
        gap = s["open"] <= h["stop"]
        return {"action": "stop_hit", "urgent": True, "text":
                f"Stop {h['stop']:,.2f} was hit on {pd.Timestamp(s['date']):%d %b} "
                + (f"(opened at {s['open']:,.2f}, below the stop). " if gap else f"(low {s['low']:,.2f}). ")
                + "If your stop order was in, you are out; if not, sell at the open."}
    if t:
        return {"action": "target_hit", "urgent": True, "text":
                f"Target {h['target']:,.2f} was reached on {pd.Timestamp(t['date']):%d %b} (high {t['high']:,.2f}). "
                "If your target order was in, you are out; if not, sell at the open."}
    if decision and decision["action"] == "exit":
        return {"action": "exit", "urgent": True, "text": f"Exit Monitor: sell at the open. {decision['reason']}"}
    if checks["time_up"]:
        return {"action": "time_up", "urgent": True, "text":
                f"The planned holding period ({h['max_hold_days']} days) is over: sell at the open."}
    if decision and decision["action"] == "tighten_stop" and decision["new_stop"] and decision["new_stop"] > h["stop"]:
        return {"action": "tighten_stop", "urgent": False, "new_stop": decision["new_stop"], "text":
                f"Exit Monitor: raise the stop to {decision['new_stop']:,.2f}. {decision['reason']}"}
    left = max(0, int(h["max_hold_days"]) - checks["days_held"])
    reason = f" {decision['reason']}" if decision else ""
    return {"action": "hold", "urgent": False, "text": f"Hold: day {checks['days_held']} of {h['max_hold_days']}, "
            f"{left} left. Stop {h['stop']:,.2f}, target {h['target']:,.2f}.{reason}"}


def holding_rows(store: Store, data: DeskData | None, desk: dict, status: str = "open") -> list[dict]:
    """Holdings with their last close, P&L, checks and today's advice."""
    rows = store.rows("SELECT * FROM holdings WHERE desk_id = ? AND status = ? ORDER BY entry_date DESC, id DESC",
                      (desk["id"], status))
    if data is None or status != "open":
        return rows
    d = desk["current_date"]
    seen = data.prev_close_day(d) if desk["phase"] == "morning" else None
    decisions = {r["symbol"]: r for r in store.rows("SELECT * FROM exit_decisions WHERE desk_id = ? AND trade_date = ?",
                                                    (desk["id"], d))}
    costs = desk["settings"].get("costs")
    for h in rows:
        close = data.raw_close(h["symbol"], seen) if seen else None
        buy_cost = order_cost(h["entry_price"] * h["quantity"], "buy", costs)
        h["name"] = data.names.get(h["symbol"], h["symbol"])
        h["last_close"] = close
        h["pnl"] = (close - h["entry_price"]) * h["quantity"] - buy_cost if close is not None else None
        h["return_pct"] = h["pnl"] / (h["entry_price"] * h["quantity"]) if h["pnl"] is not None else None
        h["checks"] = price_checks(data, h, seen)
        h["exit_decision"] = decisions.get(h["symbol"])
        h["advice"] = advice(h, h["checks"], h["exit_decision"])
    return rows


def suggested_quantity(desk: dict, entry: float, stop: float) -> int:
    p = {**PORTFOLIO_DEFAULTS, **desk["settings"]["portfolio"]}
    if entry <= stop or entry <= 0:
        return 0
    cap = float(p["capital"])
    return max(0, int(min(cap * p["risk_per_trade"] / (entry - stop), cap * p["max_position_pct"] / entry)))


def add_holding(store: Store, data: DeskData, desk_id: int, symbol: str, entry_date, entry_price: float, quantity: int,
                stop: float | None = None, target: float | None = None, max_hold_days: int | None = None) -> int:
    desk = live_desk(store, desk_id)
    symbol = symbol.strip().upper()
    if symbol not in data.names:
        raise DeskError(f"{symbol} isn't in the desk's universe (the Nifty 50 stocks).")
    if store.scalar("SELECT 1 FROM holdings WHERE desk_id = ? AND symbol = ? AND status = 'open'", (desk_id, symbol)):
        raise DeskError(f"You already hold {symbol}. Report that exit first, or edit it.")
    entry_date = pd.Timestamp(entry_date).date()
    if entry_date > today_ist():
        raise DeskError("The buy date can't be in the future.")
    plan = store.row("SELECT * FROM plans WHERE desk_id = ? AND symbol = ? AND trade_date <= ? ORDER BY trade_date DESC "
                     "LIMIT 1", (desk_id, symbol, str(entry_date)))
    stop = float(stop if stop is not None else (plan or {}).get("stop") or 0)
    target = float(target if target is not None else (plan or {}).get("target") or 0)
    hold = int(max_hold_days or (plan or {}).get("max_hold_days") or 5)
    entry_price, quantity = float(entry_price), int(quantity)
    if entry_price <= 0 or quantity <= 0:
        raise DeskError("Enter the price you paid and the number of shares.")
    if not (0 < stop < target):
        raise DeskError("Set a stop below the target (the plan's levels are filled in for you).")
    if not 1 <= hold <= 30:
        raise DeskError("The holding period is 1 to 30 trading days.")
    stamp = now()
    with store.tx() as con:
        hid = insert(con, "holdings", {
            "desk_id": desk_id, "symbol": symbol, "plan_id": plan["id"] if plan else None,
            "source": "agents" if plan and plan["accepted"] else "manual",
            "setup_type": plan["setup_type"] if plan else "none", "entry_date": str(entry_date),
            "entry_price": entry_price, "quantity": quantity, "stop": stop, "target": target, "max_hold_days": hold,
            "status": "open", "created_at": stamp, "updated_at": stamp})
        event(con, desk_id, desk["current_date"], "you", "trade", symbol,
              f"You bought {quantity:,} at {entry_price:,.2f} on {entry_date:%d %b}. Stop {stop:,.2f}, target "
              f"{target:,.2f}, up to {hold} days. The desk checks it every morning.")
    return hid


def update_holding(store: Store, desk_id: int, holding_id: int, stop: float | None = None,
                   target: float | None = None) -> None:
    live_desk(store, desk_id)
    h = store.row("SELECT * FROM holdings WHERE id = ? AND desk_id = ? AND status = 'open'", (holding_id, desk_id))
    if h is None:
        raise DeskError("No such open holding.")
    stop = float(stop if stop is not None else h["stop"])
    target = float(target if target is not None else h["target"])
    if not 0 < stop < target:
        raise DeskError("The stop must be below the target.")
    with store.tx() as con:
        update(con, "holdings", {"stop": stop, "target": target, "updated_at": now()}, "id = ?", (holding_id,))


EXIT_REASONS = ("stop", "target", "time", "exit_monitor", "manual")


def close_holding(store: Store, data: DeskData, desk_id: int, holding_id: int, exit_date, exit_price: float,
                  reason: str = "manual") -> int:
    """Report the sale: writes the trade (with estimated delivery costs) and closes the holding."""
    desk = live_desk(store, desk_id)
    h = store.row("SELECT * FROM holdings WHERE id = ? AND desk_id = ? AND status = 'open'", (holding_id, desk_id))
    if h is None:
        raise DeskError("No such open holding.")
    exit_date = pd.Timestamp(exit_date).date()
    exit_price = float(exit_price)
    if exit_price <= 0:
        raise DeskError("Enter the price you sold at.")
    if exit_date < date.fromisoformat(h["entry_date"]) or exit_date > today_ist():
        raise DeskError("The sale date must be between the buy date and today.")
    if reason not in EXIT_REASONS:
        raise DeskError(f"Exit reason is one of {', '.join(EXIT_REASONS)}.")
    costs = desk["settings"].get("costs")
    buy, sell = h["entry_price"] * h["quantity"], exit_price * h["quantity"]
    cost = order_cost(buy, "buy", costs) + order_cost(sell, "sell", costs)
    gross = sell - buy
    stamp = now()
    with store.tx() as con:
        tid = insert(con, "trades", {
            "desk_id": desk_id, "symbol": h["symbol"], "source": h["source"], "plan_id": h["plan_id"],
            "setup_type": h["setup_type"], "signal_date": h["entry_date"], "entry_date": h["entry_date"],
            "entry_price": h["entry_price"], "quantity": h["quantity"], "exit_date": str(exit_date),
            "exit_price": exit_price, "exit_reason": reason, "gross_pnl": round(gross, 2), "costs": round(cost, 2),
            "net_pnl": round(gross - cost, 2), "return_pct": round((gross - cost) / buy, 5),
            "held_days": max(1, data.held_days(h["entry_date"], exit_date))})
        update(con, "holdings", {"status": "closed", "exit_date": str(exit_date), "exit_price": exit_price,
                                 "trade_id": tid, "updated_at": stamp}, "id = ?", (holding_id,))
        event(con, desk_id, desk["current_date"], "you", "trade", h["symbol"],
              f"You sold {h['quantity']:,} at {exit_price:,.2f} on {exit_date:%d %b}: "
              f"{'+' if gross - cost >= 0 else '−'}₹{abs(gross - cost):,.0f} after estimated costs.")
    return tid


def delete_holding(store: Store, desk_id: int, holding_id: int) -> None:
    live_desk(store, desk_id)
    with store.tx() as con:
        n = con.execute("DELETE FROM holdings WHERE id = ? AND desk_id = ? AND status = 'open'",
                        (holding_id, desk_id)).rowcount
    if not n:
        raise DeskError("No such open holding.")


# ---------------------------------------------------------------- the morning for a live desk

def run_live_morning(store: Store, data: DeskData, cfg: Config, desk_id: int, progress: Progress | None = None) -> dict:
    desk = live_desk(store, desk_id)
    d = live_day(data)
    if d < today_ist() and not live_settings(cfg)["stale_ok"]:
        raise DeskError(f"Prices stop at {data.prev_close_day(d)}, so the next trading day ({d}) has already passed. "
                        "The data update didn't bring newer prices (check the network, or whether NSE has published "
                        "the bhavcopy).")
    view = data.run.day(d)
    seen = data.prev_close_day(d)
    prev_seen = data.prev_close_day(desk["current_date"]) if desk["phase"] == "morning" else None
    holdings = open_holdings(store, desk_id)
    positions = [{"symbol": h["symbol"], "entry_date": h["entry_date"], "entry": h["entry_price"], "stop": h["stop"],
                  "target": h["target"], "max_hold_days": h["max_hold_days"]} for h in holdings]
    dcfg = agent_config(cfg, desk)
    tracer = DbTracer(desk_id, d, progress)
    ctx = AgentContext(dcfg, f"desk-{desk_id}", LLMClient(dcfg, tracer), tracer, view)
    out = day_graph(ctx).invoke({"positions": positions})
    planner, results = out["planner"], out.get("results", [])
    stamp, ds = now(), str(d)
    at = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M")
    with store.tx() as con:
        manual = [r[0] for r in con.execute("SELECT symbol FROM watchlist WHERE desk_id = ? AND trade_date = ? AND "
                                            "origin = 'manual'", (desk_id, ds))]
        keep = f"({','.join('?' * len(manual))})" if manual else "('')"
        for table in ("plans", "evidence", "trace_steps"):           # a rerun of the same day replaces it
            con.execute(f"DELETE FROM {table} WHERE desk_id = ? AND trade_date = ? AND (symbol IS NULL OR symbol NOT IN "
                        f"{keep})", (desk_id, ds, *manual))
        con.execute("DELETE FROM events WHERE desk_id = ? AND trade_date = ? AND kind != 'trade'", (desk_id, ds))
        for table in ("watchlist", "exit_decisions"):
            extra = " AND origin = 'planner'" if table == "watchlist" else ""
            con.execute(f"DELETE FROM {table} WHERE desk_id = ? AND trade_date = ?{extra}", (desk_id, ds))
        insert(con, "days", {"desk_id": desk_id, "trade_date": ds, "as_of": view.as_of.isoformat(),
                             "planner": planner.model_dump(mode="json"), **llm_usage(tracer, desk["provider"]),
                             "planned_at": stamp}, replace=True)
        for t in planner.selected:
            con.execute("INSERT OR IGNORE INTO watchlist (desk_id, trade_date, symbol, origin, reasons, added_at) "
                        "VALUES (?, ?, ?, 'planner', ?, ?)", (desk_id, ds, t.symbol, json.dumps(t.reasons), stamp))
        accepted = sum(save_result(con, desk_id, d, r, stamp) for r in results)
        for dec in out.get("exits", []):
            insert(con, "exit_decisions", {"desk_id": desk_id, "trade_date": ds, "symbol": dec.symbol,
                                           "action": dec.action, "new_stop": dec.new_stop, "reason": dec.reason},
                   replace=True)
        for h in holdings:                                            # what happened since the last morning
            c = price_checks(data, h, seen)
            for kind in ("stop_hit", "target_hit"):
                hit = c[kind]
                if hit and (prev_seen is None or hit["date"] > str(prev_seen)):
                    what = "touched the stop" if kind == "stop_hit" else "reached the target"
                    event(con, desk_id, d, at, "agent", h["symbol"],
                          f"Your holding {what} on {pd.Timestamp(hit['date']):%d %b}. Check the advice in the trade book.")
            dec = next((x for x in out.get("exits", []) if x.symbol == h["symbol"]), None)
            if dec is not None and dec.action != "hold":
                what = "sell at the open" if dec.action == "exit" else f"raise the stop to {dec.new_stop:,.2f}"
                event(con, desk_id, d, at, "agent", h["symbol"], f"Exit Monitor: {what}. {dec.reason}")
        event(con, desk_id, d, at, "system", None,
              f"Suggestions for {d:%a %d %b}: the Planner picked {len(planner.selected)} of {len(view.symbols)} stocks "
              f"with data to the close of {seen:%d %b}; {accepted} long {'plan' if accepted == 1 else 'plans'} accepted; "
              f"{len(holdings)} of your holdings checked.")
        con.execute("UPDATE events SET at = ? WHERE desk_id = ? AND trade_date = ? AND at = '08:30'", (at, desk_id, ds))
        tracer.write(con)
        update(con, "desks", {"current_date": ds, "phase": "morning", "updated_at": stamp}, "id = ?", (desk_id,))
    return {"trade_date": ds, "analysed": len(results), "accepted": accepted, "holdings": len(holdings)}


# ---------------------------------------------------------------- the whole daily run

def daily(cfg: Config, store: Store, desk_ids: list[int] | None = None, *, update_data: bool = True,
          refit_if: str = "missing", progress: Progress | None = None) -> dict:
    """Download, rebuild, forecast, then plan for every live desk (or the given ones)."""
    from src.live import models as live_models

    say = progress or (lambda msg: None)
    out: dict = {}
    if update_data:
        say("Downloading new prices and filings")
        out["ingest"] = live_update.ingest(cfg, log=say)
        say("Building the live tables")
        out["build"] = live_update.build(cfg, log=say)
    info = live_update.built(cfg)
    if not info:
        raise DeskError("No live tables yet; run the update first.")
    day = date.fromisoformat(str(info["day"]))
    say(f"Forecasting {day:%a %d %b} with the live models")
    out["models"] = live_models.forecast_all(cfg, day, refit_if, log=say)
    data = live_data(cfg)
    ids = desk_ids or [r["id"] for r in store.rows("SELECT id FROM desks WHERE mode = 'live' ORDER BY id")]
    out["desks"] = {}
    for desk_id in ids:
        say(f"Agents planning {day:%a %d %b} for desk {desk_id}")
        out["desks"][desk_id] = run_live_morning(store, data, cfg, desk_id, say)
    return out
