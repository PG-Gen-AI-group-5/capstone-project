"""The paper-trading desk: the agents' daily inference and a book that trades their plans, one day at a time.

A desk replays history as if live. For trading day D:

    morning(D)   08:30 IST. The Exit Monitor reviews the open positions, the Planner picks the stocks and each
                 one goes through retrieve -> debate -> forecast -> critique (src/graph/build.py). Nothing after
                 D-1's close is visible. A person can now review the plans, remove a stock (its plan is then not
                 traded), buy an analysed stock themselves, ask to exit a position, or add a stock for analysis.
    session(D)   The market day, on the same engine as the backtester (src/backtest/engine.py): exits and new
                 entries at the open, stops and targets during the day, time exits at the close.

`advance` runs session(D) and then morning(next day). Each step is one database transaction.
"""
from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd

from src.agents.base import AgentContext
from src.backtest.engine import PORTFOLIO_DEFAULTS, Book
from src.config import Config
from src.contracts import Signal, TraceStep
from src.desk.data import DeskData, desk_settings
from src.desk.store import Store, insert, now, update
from src.graph.build import day_graph, stock_graph
from src.llm.client import LLMClient

PROVIDERS = ("offline", "openai_compatible", "anthropic")
Progress = Callable[[str], None]


class DeskError(ValueError):
    """A request the desk can't carry out (shown to the person as is)."""


# ---------------------------------------------------------------- LLM providers

def provider_status(cfg: Config) -> list[dict]:
    """Which LLM providers a desk can use, and the model each would call."""
    import os

    from src.llm.openai_compatible import configuration_ready, endpoint_settings

    llm = cfg.get("llm") or {}
    out = [{"provider": "offline", "model": None, "ready": True, "label": "Rule-based agents (no LLM)"}]
    compat = endpoint_settings(llm)
    out.append({"provider": "openai_compatible", "model": compat.get("model") or None,
                "ready": bool(configuration_ready(llm)),
                "label": f"LLM agents · {compat.get('model') or 'endpoint not set'}"})
    out.append({"provider": "anthropic", "model": llm.get("model"), "ready": bool(os.getenv("ANTHROPIC_API_KEY")),
                "label": f"LLM agents · {llm.get('model') or 'Claude'}"})
    return out


def desk_config(cfg: Config, provider: str) -> Config:
    raw = copy.deepcopy(cfg.raw)
    raw["llm"] = {**(raw.get("llm") or {}), "provider": provider}
    return Config(raw, cfg.root)


# ---------------------------------------------------------------- trace steps go to the database

class DbTracer:
    """Collects the agents' trace steps for one morning; they are written with the morning's transaction."""

    def __init__(self, desk_id: int, trade_date: date, progress: Progress | None = None):
        self.run_id = f"desk-{desk_id}"
        self.desk_id, self.trade_date, self.progress = desk_id, trade_date, progress
        self.steps: list[TraceStep] = []
        self._lock = threading.Lock()

    def step(self, agent: str, kind: str, **fields) -> TraceStep:
        with self._lock:
            row = TraceStep(run_id=self.run_id, step=len(self.steps) + 1, at=datetime.now().astimezone(), agent=agent,
                            kind=kind, **fields)
            self.steps.append(row)
        if self.progress:
            self.progress(f"{fields.get('symbol') or 'day'} · {agent}")
        return row

    def write(self, con) -> None:
        for s in self.steps:
            insert(con, "trace_steps", {
                "desk_id": self.desk_id, "trade_date": str(self.trade_date), "step": s.step, "at": s.at.isoformat(),
                "agent": s.agent, "symbol": s.symbol, "kind": s.kind, "prompt_version": s.prompt_version,
                "provider": s.provider, "input_summary": s.input_summary, "output": s.output,
                "input_tokens": s.input_tokens, "output_tokens": s.output_tokens, "cost_usd": s.cost_usd,
                "latency_ms": s.latency_ms, "cached": int(s.cached)})


# ---------------------------------------------------------------- desks

def get_desk(store: Store, desk_id: int) -> dict:
    desk = store.row("SELECT * FROM desks WHERE id = ?", (desk_id,))
    if desk is None:
        raise DeskError(f"No desk {desk_id}.")
    return desk


def create_desk(store: Store, data: DeskData, cfg: Config, name: str, start, capital: float | None = None,
                provider: str = "offline", end=None) -> int:
    """A new desk at the first trading day on or after `start`. Its first morning runs separately (a job)."""
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
    lo, hi = data.window
    start = pd.Timestamp(start).date()
    end = min(pd.Timestamp(end).date(), hi) if end else hi
    if not lo <= start <= hi:
        raise DeskError(f"Pick a start date between {lo} and {hi} (the walk-forward forecast periods; "
                        "the test period stays frozen).")
    first = data.first_day_from(start)
    if first is None or first > end:
        raise DeskError("No trading day with data between the start and end dates.")
    capital = float(capital or desk_settings(cfg)["capital"])
    if capital < 10_000:
        raise DeskError("Start with at least Rs 10,000.")
    portfolio = {**PORTFOLIO_DEFAULTS, **((cfg.get("backtest") or {}).get("portfolio") or {}), "capital": capital}
    settings = {"portfolio": portfolio, "costs": cfg.get("costs") or {}, "agents": cfg.get("agents") or {},
                "setups": cfg.get("setups") or {}}
    stamp = now()
    with store.tx() as con:
        return insert(con, "desks", {
            "name": name, "provider": provider, "model": status["model"], "start_date": str(first), "end_date": str(end),
            "capital": capital, "settings": settings, "current_date": str(first), "phase": "new",
            "book": Book(capital, portfolio).state(), "created_at": stamp, "updated_at": stamp})


def delete_desk(store: Store, desk_id: int) -> None:
    get_desk(store, desk_id)
    if store.scalar("SELECT 1 FROM jobs WHERE desk_id = ? AND status IN ('queued','running','cancelling')", (desk_id,)):
        raise DeskError("Wait for the running job to finish (or cancel it) before deleting the desk.")
    with store.tx() as con:
        con.execute("DELETE FROM desks WHERE id = ?", (desk_id,))


def book_of(desk: dict) -> Book:
    s = desk["settings"]
    return Book.from_state(desk["book"], s["portfolio"], s.get("costs"))


# ---------------------------------------------------------------- morning: the agents

def run_morning(store: Store, data: DeskData, cfg: Config, desk_id: int, progress: Progress | None = None) -> dict:
    desk = get_desk(store, desk_id)
    if desk["phase"] not in ("new", "closed"):
        raise DeskError(f"The morning for {desk['current_date']} has already run.")
    d = date.fromisoformat(desk["current_date"])
    if desk["phase"] == "closed":                      # after a session: move to the next trading day first
        nxt = data.next_day(d)
        if nxt is None or nxt > date.fromisoformat(desk["end_date"]):
            with store.tx() as con:
                update(con, "desks", {"phase": "finished", "updated_at": now()}, "id = ?", (desk_id,))
            return {"finished": True}
        d = nxt
    view = data.run.day(d)
    dcfg = agent_config(cfg, desk)
    tracer = DbTracer(desk_id, d, progress)
    llm = LLMClient(dcfg, tracer)
    ctx = AgentContext(dcfg, f"desk-{desk_id}", llm, tracer, view)
    book = book_of(desk)
    positions = [{"symbol": s, "entry_date": str(p.entry_date.date()), "entry": p.entry_price, "stop": p.stop,
                  "target": p.signal.target, "max_hold_days": p.signal.max_hold_days} for s, p in book.positions.items()]
    out = day_graph(ctx).invoke({"positions": positions})
    planner = out["planner"]
    results = out.get("results", [])
    stamp = now()
    with store.tx() as con:
        insert(con, "days", {"desk_id": desk_id, "trade_date": str(d), "as_of": view.as_of.isoformat(),
                             "planner": planner.model_dump(mode="json"), **llm_usage(tracer, desk["provider"]),
                             "planned_at": stamp}, replace=True)
        reasons = {t.symbol: t.reasons for t in planner.selected}
        for t in planner.selected:
            insert(con, "watchlist", {"desk_id": desk_id, "trade_date": str(d), "symbol": t.symbol, "origin": "planner",
                                      "reasons": t.reasons, "added_at": stamp}, replace=True)
        accepted = 0
        for r in results:
            accepted += save_result(con, desk_id, d, r, stamp)
        for dec in out.get("exits", []):
            insert(con, "exit_decisions", {"desk_id": desk_id, "trade_date": str(d), "symbol": dec.symbol,
                                           "action": dec.action, "new_stop": dec.new_stop, "reason": dec.reason},
                   replace=True)
            if dec.action != "hold":
                what = "exit at the open" if dec.action == "exit" else f"stop raised to {dec.new_stop:,.2f}"
                event(con, desk_id, d, "08:30", "agent", dec.symbol, f"Exit Monitor: {what}. {dec.reason}")
        skipped = [s for s in reasons if s not in {r["plan"].symbol for r in results}]
        note = f" ({len(skipped)} had no usable levels)" if skipped else ""
        event(con, desk_id, d, "08:30", "system", None,
              f"Plans ready: the Planner picked {len(planner.selected)} of {len(view.symbols)} stocks{note}; "
              f"{accepted} long {'plan' if accepted == 1 else 'plans'} accepted by the Critic.")
        tracer.write(con)
        update(con, "desks", {"current_date": str(d), "phase": "morning", "updated_at": stamp}, "id = ?", (desk_id,))
    return {"trade_date": str(d), "analysed": len(results), "accepted": accepted}


def agent_config(cfg: Config, desk: dict) -> Config:
    """The desk's LLM provider, with the agent and setup settings frozen when the desk was made."""
    c = desk_config(cfg, desk["provider"])
    s = desk["settings"]
    c.raw["agents"] = s.get("agents") or c.raw.get("agents")
    c.raw["setups"] = s.get("setups") or c.raw.get("setups")
    return c


def llm_usage(tracer: DbTracer, provider: str) -> dict:
    calls = [s for s in tracer.steps if s.kind == "llm_call"]
    fallbacks = sum(1 for s in tracer.steps if s.kind == "offline") if provider != "offline" else 0
    return {"llm_calls": len(calls), "llm_cost": round(sum(s.cost_usd for s in calls), 6), "fallbacks": fallbacks}


def save_result(con, desk_id: int, d: date, r: dict, stamp: str) -> int:
    """One analysed stock: its plan, debate, verdict and evidence. Returns 1 when the plan is an accepted long."""
    plan, verdict = r["plan"], r["verdict"]
    accepted = int(verdict.verdict == "accept" and plan.direction == "long")
    con.execute("DELETE FROM evidence WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (desk_id, str(d), plan.symbol))
    insert(con, "plans", {
        "desk_id": desk_id, "trade_date": str(d), "symbol": plan.symbol, "setup_type": plan.setup_type,
        "direction": plan.direction, "entry": plan.entry, "stop": plan.stop, "target": plan.target,
        "max_hold_days": plan.max_hold_days, "confidence": plan.confidence, "vol_forecast": plan.vol_forecast,
        "base_p_up": plan.base_forecast_p_up, "verdict": verdict.verdict, "accepted": accepted,
        "rounds": r.get("rounds", 1), "failed_checks": [c.code for c in verdict.checks if not c.passed],
        "critic_reasons": verdict.reasons, "rationale": plan.rationale, "evidence_ids": plan.evidence_ids,
        "bull": r["bull"].model_dump(mode="json"), "bear": r["bear"].model_dump(mode="json"),
        "revisions": [{"plan": x["plan"].model_dump(mode="json"), "verdict": x["verdict"].model_dump(mode="json")}
                      for x in r.get("revisions", [])],
        "catalysts": [k.model_dump(mode="json") for k in r.get("catalysts", [])], "created_at": stamp}, replace=True)
    for e in r.get("evidence", []):
        insert(con, "evidence", {"desk_id": desk_id, "trade_date": str(d), "symbol": plan.symbol,
                                 "evidence_id": e.evidence_id, "kind": e.kind, "title": e.title, "text": e.text or "",
                                 "url": e.url, "published_at": e.published_at.isoformat(), "important": int(e.important)},
               replace=True)
    if accepted:
        text = (f"Long plan accepted: stop {plan.stop:,.2f}, target {plan.target:,.2f}, hold up to {plan.max_hold_days} "
                f"days, confidence {plan.confidence:.0%}. Buys at the open.")
    elif plan.direction == "no_trade" and verdict.verdict == "accept":
        text = f"No trade: {first_sentence(plan.rationale)}"
    else:
        text = f"Critic {verdict.verdict}ed the plan: {first_sentence('; '.join(verdict.reasons) or plan.rationale)}"
    event(con, desk_id, d, "08:30", "plan", plan.symbol, text)
    if r.get("rounds", 1) > 1:
        event(con, desk_id, d, "08:30", "agent", plan.symbol,
              f"Critic sent the plan back {r['rounds'] - 1} time(s) before the final verdict.")
    return accepted


def first_sentence(text: str, limit: int = 160) -> str:
    t = " ".join((text or "").split())
    cut = t.find(". ")
    t = t[:cut + 1] if 0 < cut < limit else t
    return t if len(t) <= limit else t[:limit - 1] + "…"


def event(con, desk_id: int, d, at: str, kind: str, symbol: str | None, text: str) -> None:
    insert(con, "events", {"desk_id": desk_id, "trade_date": str(d), "at": at, "kind": kind, "symbol": symbol,
                           "text": text, "created_at": now()})


# ---------------------------------------------------------------- session: the book

AT_LABEL = {"open": "09:15", "intraday": "intraday", "close": "15:30"}
REASON_TEXT = {"stop": "Stop hit", "gap_stop": "Gapped through the stop", "target": "Target hit",
               "time": "Holding period over", "exit_monitor": "Exit Monitor exit", "manual": "Your exit"}


def run_session(store: Store, data: DeskData, cfg: Config, desk_id: int) -> dict:
    desk = get_desk(store, desk_id)
    if desk["phase"] != "morning":
        raise DeskError("The session runs after the morning's plans are made.")
    d = date.fromisoformat(desk["current_date"])
    today = data.bars_on(d)
    book = book_of(desk)
    plans = {p["id"]: p for p in store.rows("SELECT * FROM plans WHERE desk_id = ? AND trade_date = ?", (desk_id, str(d)))}
    removed = {r["symbol"] for r in store.rows(
        "SELECT symbol FROM watchlist WHERE desk_id = ? AND trade_date = ? AND removed = 1", (desk_id, str(d)))}
    pending = store.rows("SELECT * FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND status = 'pending'",
                         (desk_id, str(d)))
    manual_buys = {o["symbol"]: o for o in pending if o["action"] == "buy"}
    manual_exits = {o["symbol"] for o in pending if o["action"] == "exit"}

    signals = [Signal(symbol=p["symbol"], signal_date=d, stop=p["stop"], target=p["target"],
                      max_hold_days=p["max_hold_days"], source="agents", confidence=p["confidence"], ref=str(pid))
               for pid, p in plans.items() if p["accepted"] and p["symbol"] not in removed
               and p["symbol"] not in manual_buys]
    signals += [Signal(symbol=o["symbol"], signal_date=d, stop=o["stop"], target=o["target"],
                       max_hold_days=o["max_hold_days"], source="manual", confidence=1.0,
                       ref=str(o["plan_id"]) if o["plan_id"] else None) for o in manual_buys.values()]
    exits = {r["symbol"]: SimpleNamespace(action=r["action"], new_stop=r["new_stop"]) for r in store.rows(
        "SELECT * FROM exit_decisions WHERE desk_id = ? AND trade_date = ?", (desk_id, str(d)))}
    for s in manual_exits:
        exits[s] = SimpleNamespace(action="exit", new_stop=None)

    stamp = now()
    with store.tx() as con:
        if today is None:                      # no prices that day (should not happen inside the window)
            event(con, desk_id, d, "15:30", "system", None, "No prices for this day; nothing traded.")
            res = None
        else:
            res = book.session(d, today, signals, exits)
            plan_setup = {str(pid): p["setup_type"] for pid, p in plans.items()}
            for o in res.orders:
                plan_id = int(o.ref) if o.ref else None
                insert(con, "orders", {"desk_id": desk_id, "trade_date": str(d), "at": o.at, "symbol": o.symbol,
                                       "side": o.side, "kind": o.kind, "quantity": o.quantity, "price": o.price,
                                       "costs": o.costs, "source": o.source, "plan_id": plan_id,
                                       "status": "no_fill" if o.kind == "no_fill" else "filled"})
                event(con, desk_id, d, AT_LABEL[o.at], "trade", o.symbol, order_text(o, manual_exits))
            for t in res.trades:
                if t.exit_reason == "no_fill":
                    continue
                pos_ref = next((o.ref for o in res.orders if o.symbol == t.symbol and o.side == "sell"), None)
                reason = t.exit_reason if t.exit_reason != "exit" else (
                    "manual" if t.symbol in manual_exits else "exit_monitor")
                insert(con, "trades", {
                    "desk_id": desk_id, "symbol": t.symbol, "source": t.source,
                    "plan_id": int(pos_ref) if pos_ref else None,
                    "setup_type": plan_setup.get(pos_ref) or setup_of(store, pos_ref), "signal_date": str(t.signal_date),
                    "entry_date": str(t.entry_date), "entry_price": t.entry_price, "quantity": t.quantity,
                    "exit_date": str(t.exit_date), "exit_price": t.exit_price, "exit_reason": reason,
                    "gross_pnl": t.gross_pnl, "costs": t.costs, "net_pnl": t.net_pnl, "return_pct": t.return_pct,
                    "held_days": data.held_days(t.entry_date, t.exit_date)})
            e = res.equity
            insert(con, "equity", {"desk_id": desk_id, "trade_date": str(d), "cash": e["cash"], "positions": e["positions"],
                                   "equity": e["equity"], "open_positions": e["open_positions"]}, replace=True)
        con.execute("UPDATE pending_orders SET status = 'done' WHERE desk_id = ? AND trade_date = ? AND status = 'pending'",
                    (desk_id, str(d)))
        con.execute("UPDATE days SET closed_at = ? WHERE desk_id = ? AND trade_date = ?", (stamp, desk_id, str(d)))
        update(con, "desks", {"book": book.state(), "phase": "closed", "updated_at": stamp}, "id = ?", (desk_id,))
    return {"trade_date": str(d), "orders": len(res.orders) if res else 0,
            "equity": res.equity["equity"] if res else None}


def setup_of(store: Store, plan_id: str | None) -> str:
    if not plan_id:
        return "none"
    return store.scalar("SELECT setup_type FROM plans WHERE id = ?", (int(plan_id),)) or "none"


def order_text(o, manual_exits: set[str]) -> str:
    who = "your order" if o.source == "manual" else "agents' plan"
    if o.kind == "entry":
        return f"Bought {o.quantity:,} at {o.price:,.2f} on the open ({who})."
    if o.kind == "no_fill":
        return f"Not bought: opened at {o.price:,.2f}, outside the stop-target range or locked at its band ({who})."
    reason = o.kind if o.kind != "exit" else ("manual" if o.symbol in manual_exits else "exit_monitor")
    return f"{REASON_TEXT[reason]}: sold {o.quantity:,} at {o.price:,.2f}."


# ---------------------------------------------------------------- moving the desk forward

def advance(store: Store, data: DeskData, cfg: Config, desk_id: int, days: int = 1,
            progress: Callable[[str, int, int], None] | None = None,
            cancelled: Callable[[], bool] | None = None) -> int:
    """Run `days` sessions, each followed by the next morning. Returns the number of days completed."""
    done = 0
    for _ in range(max(1, days)):
        if cancelled and cancelled():
            break
        desk = get_desk(store, desk_id)
        if desk["phase"] == "finished":
            break
        say = (lambda msg, k=done: progress(msg, k, days)) if progress else None
        if desk["phase"] == "new":
            say and say(f"Plans for {fmt_day(desk['current_date'])}")
            run_morning(store, data, cfg, desk_id, say)
            continue
        if desk["phase"] == "morning":
            say and say(f"Session {fmt_day(desk['current_date'])}")
            run_session(store, data, cfg, desk_id)
        if cancelled and cancelled():
            done += 1
            break
        nxt = data.next_day(desk["current_date"])
        say and say(f"Plans for {fmt_day(nxt) if nxt else 'the next day'}")
        result = run_morning(store, data, cfg, desk_id, say)
        done += 1
        if result.get("finished"):
            break
    return done


def fmt_day(d) -> str:
    return pd.Timestamp(d).strftime("%a %d %b %Y") if d else ""


# ---------------------------------------------------------------- what a person can do in the morning

def morning_desk(store: Store, desk_id: int) -> tuple[dict, date]:
    desk = get_desk(store, desk_id)
    if desk["phase"] != "morning":
        raise DeskError("Orders and changes to the list are taken in the morning, before the session runs.")
    return desk, date.fromisoformat(desk["current_date"])


def add_to_watchlist(store: Store, desk_id: int, symbol: str) -> str:
    """'restored' when the stock was on today's list and removed; 'analyse' when the agents must look at it."""
    _, d = morning_desk(store, desk_id)
    symbol = symbol.strip().upper()
    row = store.row("SELECT * FROM watchlist WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (desk_id, str(d), symbol))
    if row and not row["removed"]:
        raise DeskError(f"{symbol} is already on today's list.")
    if row:
        with store.tx() as con:
            con.execute("UPDATE watchlist SET removed = 0 WHERE desk_id = ? AND trade_date = ? AND symbol = ?",
                        (desk_id, str(d), symbol))
        return "restored"
    return "analyse"


def analyse_stock(store: Store, data: DeskData, cfg: Config, desk_id: int, symbol: str,
                  progress: Progress | None = None) -> dict:
    desk, d = morning_desk(store, desk_id)
    symbol = symbol.strip().upper()
    view = data.run.day(d)
    if symbol not in view.rows.index:
        raise DeskError(f"No data for {symbol} on {d}. The desk covers the Nifty 50 stocks.")
    if view.setup(symbol) is None:
        raise DeskError(f"{symbol} has no price history yet for trade levels on {d}.")
    dcfg = agent_config(cfg, desk)
    tracer = DbTracer(desk_id, d, progress)
    ctx = AgentContext(dcfg, f"desk-{desk_id}", LLMClient(dcfg, tracer), tracer, view)
    result = stock_graph(ctx).invoke({"symbol": symbol, "reasons": ["added by you"]})
    stamp = now()
    with store.tx() as con:
        insert(con, "watchlist", {"desk_id": desk_id, "trade_date": str(d), "symbol": symbol, "origin": "manual",
                                  "reasons": ["added by you"], "added_at": stamp}, replace=True)
        accepted = save_result(con, desk_id, d, result, stamp)
        tracer.write(con)
        usage = llm_usage(tracer, desk["provider"])
        con.execute("UPDATE days SET llm_calls = llm_calls + ?, llm_cost = llm_cost + ?, fallbacks = fallbacks + ? "
                    "WHERE desk_id = ? AND trade_date = ?",
                    (usage["llm_calls"], usage["llm_cost"], usage["fallbacks"], desk_id, str(d)))
    return {"symbol": symbol, "accepted": bool(accepted)}


def remove_from_watchlist(store: Store, desk_id: int, symbol: str) -> None:
    _, d = morning_desk(store, desk_id)
    with store.tx() as con:
        n = con.execute("UPDATE watchlist SET removed = 1 WHERE desk_id = ? AND trade_date = ? AND symbol = ?",
                        (desk_id, str(d), symbol)).rowcount
        if not n:
            raise DeskError(f"{symbol} is not on today's list.")
        con.execute("DELETE FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND symbol = ? AND action = 'buy' "
                    "AND status = 'pending'", (desk_id, str(d), symbol))


def place_order(store: Store, desk_id: int, symbol: str, action: str) -> int:
    """A person's order for today's session: `buy` an analysed stock at the open with its plan's levels, or
    `exit` an open position at the open."""
    desk, d = morning_desk(store, desk_id)
    symbol = symbol.strip().upper()
    held = {p["signal"]["symbol"] for p in desk["book"]["positions"]}
    values = {"desk_id": desk_id, "trade_date": str(d), "symbol": symbol, "action": action, "created_at": now()}
    if action == "exit":
        if symbol not in held:
            raise DeskError(f"There is no open position in {symbol}.")
    elif action == "buy":
        if symbol in held:
            raise DeskError(f"{symbol} is already held.")
        plan = store.row("SELECT * FROM plans WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (desk_id, str(d), symbol))
        if plan is None:
            raise DeskError(f"The agents haven't analysed {symbol} today. Add it to the list first.")
        if plan["accepted"] and not store.scalar(
                "SELECT removed FROM watchlist WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (desk_id, str(d), symbol)):
            raise DeskError(f"The agents' accepted plan for {symbol} already buys at the open.")
        values.update(plan_id=plan["id"], stop=plan["stop"], target=plan["target"], max_hold_days=plan["max_hold_days"])
    else:
        raise DeskError("An order is a buy or an exit.")
    if store.scalar("SELECT 1 FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND symbol = ? AND action = ?",
                    (desk_id, str(d), symbol, action)):
        raise DeskError(f"There is already a pending {action} order for {symbol}.")
    with store.tx() as con:
        return insert(con, "pending_orders", values)


def cancel_order(store: Store, desk_id: int, order_id: int) -> None:
    morning_desk(store, desk_id)
    with store.tx() as con:
        n = con.execute("DELETE FROM pending_orders WHERE id = ? AND desk_id = ? AND status = 'pending'",
                        (order_id, desk_id)).rowcount
    if not n:
        raise DeskError("No such pending order.")
