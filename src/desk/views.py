"""Read models for the app's screens. Everything shown on day D's morning uses data up to D-1's close only."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.agents.base import AGENT_DEFAULTS
from src.desk.data import DeskData, forecast_band
from src.desk.store import Store
from src.features.setups import settings as setup_settings

ACTIVE_JOB = "('queued','running','cancelling')"


def visible_through(data: DeskData, desk: dict) -> date | None:
    """The last day whose close the desk has seen: D-1 before D's session has run, D after it."""
    d = date.fromisoformat(desk["current_date"])
    return data.prev_close_day(d) if desk["phase"] in ("new", "morning") else d


def num(x, digits: int = 6):
    return None if x is None or pd.isna(x) else round(float(x), digits)


# ---------------------------------------------------------------- desk summary

def desk_summary(store: Store, data: DeskData, desk: dict) -> dict:
    did = desk["id"]
    cur = desk["current_date"]
    seen = visible_through(data, desk)
    eq = store.rows("SELECT * FROM equity WHERE desk_id = ? ORDER BY trade_date DESC LIMIT 2", (did,))
    equity = eq[0]["equity"] if eq else desk["capital"]
    prev = eq[1]["equity"] if len(eq) > 1 else desk["capital"]
    last_session = eq[0]["trade_date"] if eq else None
    positions = desk["book"]["positions"]
    exposure = sum(p["quantity"] * (data.raw_close(p["signal"]["symbol"], seen) or p["entry_price"]) for p in positions) \
        if seen else 0.0
    plans = store.row("SELECT COUNT(*) AS analysed, COALESCE(SUM(accepted), 0) AS accepted FROM plans "
                      "WHERE desk_id = ? AND trade_date = ?", (did, cur)) or {"analysed": 0, "accepted": 0}
    removed = store.scalar("SELECT COUNT(*) FROM plans p JOIN watchlist w USING (desk_id, trade_date, symbol) "
                           "WHERE p.desk_id = ? AND p.trade_date = ? AND p.accepted = 1 AND w.removed = 1", (did, cur)) or 0
    buys = store.scalar("SELECT COUNT(*) FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND action = 'buy' "
                        "AND status = 'pending'", (did, cur)) or 0
    month = recent_days(data, seen, 21)
    recent = store.rows("SELECT net_pnl FROM trades WHERE desk_id = ? AND exit_date >= ?", (did, str(month))) \
        if month else []
    wins = [t["net_pnl"] for t in recent if t["net_pnl"] > 0]
    losses = [-t["net_pnl"] for t in recent if t["net_pnl"] <= 0]
    day = store.row("SELECT llm_calls, llm_cost, fallbacks FROM days WHERE desk_id = ? AND trade_date = ?", (did, cur))
    totals = store.row("SELECT COALESCE(SUM(llm_calls),0) AS calls, COALESCE(SUM(llm_cost),0) AS cost, "
                       "COALESCE(SUM(fallbacks),0) AS fallbacks, COUNT(*) AS days FROM days WHERE desk_id = ?", (did,))
    job = store.row(f"SELECT * FROM jobs WHERE desk_id = ? AND status IN {ACTIVE_JOB} ORDER BY id DESC LIMIT 1", (did,)) \
        or store.row("SELECT * FROM jobs WHERE desk_id = ? ORDER BY id DESC LIMIT 1", (did,))
    out = {
        "id": did, "name": desk["name"], "provider": desk["provider"], "model": desk["model"],
        "start_date": desk["start_date"], "end_date": desk["end_date"], "current_date": cur, "phase": desk["phase"],
        "visible_through": str(seen) if seen else None, "next_day": str(data.next_day(cur) or "") or None,
        "capital": desk["capital"], "cash": desk["book"]["cash"], "equity": equity,
        "total_return": equity / desk["capital"] - 1, "last_session": last_session,
        "day_pnl": equity - prev if eq else 0.0, "open_positions": len(positions), "exposure": exposure,
        "signals": {"analysed": plans["analysed"], "accepted": plans["accepted"] - removed, "manual_buys": buys},
        "win_rate_month": {"trades": len(recent), "win_rate": len(wins) / len(recent) if recent else None,
                           "profit_factor": sum(wins) / sum(losses) if losses and sum(losses) > 0 else None},
        "llm_today": day or {"llm_calls": 0, "llm_cost": 0, "fallbacks": 0}, "llm_total": totals,
        "job": job,
        "days_done": store.scalar("SELECT COUNT(*) FROM equity WHERE desk_id = ?", (did,)) or 0,
        "days_total": int(((data.days >= pd.Timestamp(desk["start_date"])) &
                           (data.days <= pd.Timestamp(desk["end_date"]))).sum()),
        "mode": desk.get("mode", "replay"),
    }
    return live_summary(store, data, desk, out) if desk.get("mode") == "live" else out


def live_summary(store: Store, data: DeskData, desk: dict, out: dict) -> dict:
    """A live desk's figures come from the holdings you reported, marked at the last close."""
    from src.desk.live import holding_rows
    from src.live import update as live_update

    hs = holding_rows(store, data, desk)
    seen = visible_through(data, desk)
    prior = data.prev_close_day(seen) if seen else None
    day = 0.0
    for h in hs:
        a, b = data.raw_close(h["symbol"], seen) if seen else None, data.raw_close(h["symbol"], prior) if prior else None
        if a is not None and b is not None and h["entry_date"] < str(seen):
            day += (a - b) * h["quantity"]
    info = live_update.built(data.cfg)
    out.update(open_positions=len(hs), exposure=sum((h["last_close"] or h["entry_price"]) * h["quantity"] for h in hs),
               open_pnl=sum(h["pnl"] or 0 for h in hs), day_pnl=day, last_session=str(seen) if seen else None,
               urgent=sum(1 for h in hs if h["advice"]["urgent"]), equity=None, total_return=None,
               days_total=None, live={"built_at": info.get("built_at"), "last_price_day": info.get("last_price_day"),
                                      "day": info.get("day"),
                                      "planned_at": store.scalar("SELECT planned_at FROM days WHERE desk_id = ? AND "
                                                                 "trade_date = ?", (desk["id"], desk["current_date"]))})
    return out


def held_symbols(store: Store, desk: dict) -> set[str]:
    if desk.get("mode") == "live":
        return {r["symbol"] for r in store.rows("SELECT symbol FROM holdings WHERE desk_id = ? AND status = 'open'",
                                                (desk["id"],))}
    return {p["signal"]["symbol"] for p in desk["book"]["positions"]}


def recent_days(data: DeskData, through, n: int) -> date | None:
    """The first of the last n trading days up to `through`."""
    if through is None:
        return None
    d = data.run.days
    i = d.searchsorted(pd.Timestamp(through), side="right")
    return d[max(0, i - n)].date() if i else None


# ---------------------------------------------------------------- watchlist

def watchlist(store: Store, data: DeskData, desk: dict, d: str | None = None) -> dict:
    did, d = desk["id"], d or desk["current_date"]
    view = data.run.day(d)
    seen = data.prev_close_day(d)
    prior = data.prev_close_day(seen) if seen else None
    rows = store.rows("SELECT * FROM watchlist WHERE desk_id = ? AND trade_date = ? ORDER BY added_at, symbol", (did, d))
    plans = {p["symbol"]: p for p in store.rows("SELECT * FROM plans WHERE desk_id = ? AND trade_date = ?", (did, d))}
    pending = store.rows("SELECT * FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND status = 'pending'", (did, d))
    held = held_symbols(store, desk)
    out = []
    for w in rows:
        s = w["symbol"]
        r = view.rows.loc[s] if s in view.rows.index else None
        close, before = data.raw_close(s, seen), data.raw_close(s, prior) if prior else None
        spark = data.chart_frame(s, seen, 20)["close"].round(2).tolist() if seen else []
        p = plans.get(s)
        out.append({
            "symbol": s, "name": data.names.get(s, s), "origin": w["origin"], "removed": bool(w["removed"]),
            "reasons": w["reasons"], "last_close": close, "change_pct": (close / before - 1) if close and before else None,
            "spark": spark, "rsi14": num(r["rsi14"] * 100, 2) if r is not None else None,
            "ema20_gap": num(r["ema20_gap"]) if r is not None else None,
            "setup_type": str(r["setup_type"]) if r is not None else "none",
            "plan": None if p is None else {k: p[k] for k in ("id", "direction", "verdict", "accepted", "confidence",
                                                               "stop", "target", "max_hold_days", "rounds")},
            "held": s in held, "pending": [o["action"] for o in pending if o["symbol"] == s],
        })
    return {"trade_date": d, "visible_through": str(seen) if seen else None, "rows": out,
            "planner": store.scalar("SELECT planner FROM days WHERE desk_id = ? AND trade_date = ?", (did, d))}


# ---------------------------------------------------------------- one stock

VALUE_ROWS = [
    ("ema20_gap", "vs EMA20", "pct", "close ÷ EMA20 − 1"),
    ("ema200_gap", "vs EMA200", "pct", "close ÷ EMA200 − 1"),
    ("rsi14", "RSI 14", "rsi", "100 − 100 ÷ (1 + avg gain ÷ avg loss)"),
    ("atr14_pct", "ATR 14", "pct", "average true range ÷ close"),
    ("rvol_20", "Volatility σ, daily", "pct", "RMS of daily log returns, 20 days"),
    ("ret_20", "Return, 20 days", "pct", "close ÷ close 20 days ago − 1"),
    ("dist_52w_high", "From 52-week high", "pct", "close ÷ 52-week high − 1"),
    ("rs_nifty_20", "Strength vs Nifty, 20d", "pct", "20-day return − Nifty 50's"),
    ("beta_nifty", "Beta to Nifty", "num", "cov(stock, Nifty) ÷ var(Nifty), daily"),
    ("vix", "India VIX", "pct", "the market's implied volatility"),
]


def setup_checks(row: pd.Series, cfg_setups: dict | None) -> dict:
    s = setup_settings(cfg_setups)
    g = lambda k: None if k not in row.index or pd.isna(row[k]) else float(row[k])
    pct = lambda v: "—" if v is None else f"{v:+.2%}"
    p, b, o = s["pullback"], s["breakout"], s["oversold"]
    ok = lambda v, test: v is not None and test(v)
    rsi = g("rsi14")
    rsi_txt = "—" if rsi is None else f"RSI {rsi * 100:.1f}"
    return {
        "pullback_uptrend": [
            {"label": "Close above the 200-day EMA", "pass": ok(g("ema200_gap"), lambda v: v > 0), "detail": pct(g("ema200_gap"))},
            {"label": "20-day EMA above the 50-day EMA", "pass": ok(g("ema20_50"), lambda v: v > 0), "detail": pct(g("ema20_50"))},
            {"label": f"Within {p['ema20_gap_min']:+.0%} to {p['ema20_gap_max']:+.0%} of the 20-day EMA",
             "pass": ok(g("ema20_gap"), lambda v: p["ema20_gap_min"] <= v <= p["ema20_gap_max"]), "detail": pct(g("ema20_gap"))},
            {"label": f"RSI 14 between {p['rsi_min'] * 100:.0f} and {p['rsi_max'] * 100:.0f}",
             "pass": ok(rsi, lambda v: p["rsi_min"] <= v <= p["rsi_max"]), "detail": rsi_txt},
        ],
        "breakout": [
            {"label": f"Within {abs(b['dist_52w_high_min']):.0%} of the 52-week high",
             "pass": ok(g("dist_52w_high"), lambda v: v >= b["dist_52w_high_min"]), "detail": pct(g("dist_52w_high"))},
            {"label": f"Volume above normal (z ≥ {b['logvol_z_min']})", "pass": ok(g("logvol_z20"), lambda v: v >= b["logvol_z_min"]),
             "detail": "—" if g("logvol_z20") is None else f"z {g('logvol_z20'):+.2f}"},
            {"label": "Up over the last 5 days", "pass": ok(g("ret_5"), lambda v: v > 0), "detail": pct(g("ret_5"))},
        ],
        "oversold_bounce": [
            {"label": "Close above the 200-day EMA", "pass": ok(g("ema200_gap"), lambda v: v > 0), "detail": pct(g("ema200_gap"))},
            {"label": f"RSI 14 at or below {o['rsi_max'] * 100:.0f}", "pass": ok(rsi, lambda v: v <= o["rsi_max"]),
             "detail": rsi_txt},
        ],
    }


def stock_detail(store: Store, data: DeskData, desk: dict, symbol: str, d: str | None = None) -> dict:
    did, d = desk["id"], d or desk["current_date"]
    view = data.run.day(d)
    if symbol not in view.rows.index:
        raise KeyError(symbol)
    row = view.rows.loc[symbol]
    is_today = d == desk["current_date"]
    seen = visible_through(data, desk) if is_today else data.prev_close_day(d)
    morning_seen = data.prev_close_day(d)
    plan = store.row("SELECT * FROM plans WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (did, d, symbol))
    setup = view.setup(symbol)
    s = desk["settings"]
    holds = {**AGENT_DEFAULTS["hold_days"], **((s.get("agents") or {}).get("hold_days") or {})}
    hold = plan["max_hold_days"] if plan else holds.get(str(row["setup_type"]), 5)
    base = view.base_forecast(symbol)
    vol = data.vol_forecast(symbol, d)
    ref = data.raw_close(symbol, morning_seen) if morning_seen else None
    band = forecast_band(ref, int(hold), vol, base.exp_return if base else None,
                         num(row.get("rvol_20"))) if ref else []
    chart = data.chart_frame(symbol, seen, 120) if seen else pd.DataFrame()
    future = [str(x.date()) for x in data.run.calendar_days_after(pd.Timestamp(morning_seen) + pd.Timedelta(days=1),
                                                                    int(hold))] if morning_seen else []
    orders = store.rows("SELECT trade_date, side, kind, price, quantity, at FROM orders WHERE desk_id = ? AND symbol = ? "
                        "AND status = 'filled' AND trade_date <= ? ORDER BY id", (did, symbol, str(seen) if seen else d))
    position = next((p for p in desk["book"]["positions"] if p["signal"]["symbol"] == symbol), None)
    watch = store.row("SELECT * FROM watchlist WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (did, d, symbol))
    pending = store.rows("SELECT * FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND symbol = ? "
                         "AND status = 'pending'", (did, d, symbol))
    exit_dec = store.row("SELECT * FROM exit_decisions WHERE desk_id = ? AND trade_date = ? AND symbol = ?", (did, d, symbol))
    close, prev = data.raw_close(symbol, seen) if seen else None, None
    if seen:
        p = data.prev_close_day(seen)
        prev = data.raw_close(symbol, p) if p else None
    morning = is_today and desk["phase"] == "morning"
    removed = bool(watch and watch["removed"])
    out = {
        "symbol": symbol, "name": data.names.get(symbol, symbol), "trade_date": d, "is_today": is_today,
        "visible_through": str(seen) if seen else None,
        "price": {"last_close": close, "prev_close": prev,
                  "change": close - prev if close and prev else None,
                  "change_pct": close / prev - 1 if close and prev else None},
        "setup": setup.model_dump(mode="json") if setup else None, "setup_type": str(row["setup_type"]),
        "values": [{"key": k, "label": label, "kind": kind, "formula": f,
                    "value": num(row[k] * 100 if kind == "rsi" else row[k]) if k in row.index else None}
                   for k, label, kind, f in VALUE_ROWS],
        "checks": setup_checks(row, s.get("setups")),
        "plan": plan,
        "evidence": store.rows("SELECT evidence_id, kind, title, text, url, published_at, important FROM evidence "
                               "WHERE desk_id = ? AND trade_date = ? AND symbol = ? ORDER BY important DESC, "
                               "published_at DESC", (did, d, symbol)),
        "forecast": {"horizon": int(hold), "reference": ref, "points": band, "days": future,
                     "vol_model": "lstm" if vol else ("realised (20d)" if band else None),
                     "base": None if base is None else {"model": base.model, "p_up": base.p_up,
                                                        "exp_return": base.exp_return}},
        "chart": {"bars": [{"d": str(r.trade_date.date()), "o": round(r.open, 2), "h": round(r.high, 2),
                            "l": round(r.low, 2), "c": round(r.close, 2), "v": num(r.volume, 0),
                            "ema20": round(r.ema20, 2), "ema50": round(r.ema50, 2)} for r in chart.itertuples()],
                  "markers": orders},
        "position": None if position is None else {
            "quantity": position["quantity"], "entry_price": position["entry_price"], "entry_date": position["entry_date"],
            "stop": position["stop"], "target": position["signal"]["target"], "days": position["days"],
            "max_hold_days": position["signal"]["max_hold_days"], "source": position["signal"]["source"]},
        "exit_decision": exit_dec,
        "watch": watch, "pending": pending,
        "actions": {
            "buy": morning and plan is not None and position is None and setup is not None
            and not any(o["action"] == "buy" for o in pending) and (not plan["accepted"] or removed),
            "exit": morning and position is not None and not any(o["action"] == "exit" for o in pending),
            "remove": morning and watch is not None and not removed,
            "restore": morning and removed,
        },
        "trace_steps": store.scalar("SELECT COUNT(*) FROM trace_steps WHERE desk_id = ? AND trade_date = ? AND symbol = ?",
                                    (did, d, symbol)) or 0,
        "mode": desk.get("mode", "replay"),
    }
    return live_detail(store, data, desk, out) if desk.get("mode") == "live" else out


def live_detail(store: Store, data: DeskData, desk: dict, out: dict) -> dict:
    from src.desk.live import holding_rows, suggested_quantity

    sym, morning = out["symbol"], out["is_today"] and desk["phase"] == "morning"
    h = next((x for x in holding_rows(store, data, desk) if x["symbol"] == sym), None)
    plan = out["plan"]
    out["holding"] = h
    out["position"] = None if h is None else {
        "quantity": h["quantity"], "entry_price": h["entry_price"], "entry_date": h["entry_date"], "stop": h["stop"],
        "target": h["target"], "days": h["checks"]["days_held"], "max_hold_days": h["max_hold_days"],
        "source": h["source"]}
    marks = []
    for r in store.rows("SELECT * FROM holdings WHERE desk_id = ? AND symbol = ?", (desk["id"], sym)):
        marks.append({"trade_date": r["entry_date"], "side": "buy", "price": r["entry_price"], "quantity": r["quantity"]})
        if r["status"] == "closed":
            marks.append({"trade_date": r["exit_date"], "side": "sell", "price": r["exit_price"], "quantity": r["quantity"]})
    out["chart"]["markers"] = marks
    ref = out["forecast"]["reference"]
    out["suggested_quantity"] = suggested_quantity(desk, ref, plan["stop"]) if plan and ref else None
    removed = bool(out["watch"] and out["watch"]["removed"])
    out["actions"] = {"buy": False, "exit": False, "bought": morning and h is None, "sold": h is not None,
                      "remove": morning and out["watch"] is not None and not removed, "restore": morning and removed}
    return out


def trace(store: Store, desk_id: int, d: str, symbol: str | None) -> list[dict]:
    if symbol:
        return store.rows("SELECT * FROM trace_steps WHERE desk_id = ? AND trade_date = ? AND symbol = ? ORDER BY step",
                          (desk_id, d, symbol))
    return store.rows("SELECT * FROM trace_steps WHERE desk_id = ? AND trade_date = ? AND symbol IS NULL ORDER BY step",
                      (desk_id, d))


# ---------------------------------------------------------------- trade book

def book(store: Store, data: DeskData, desk: dict) -> dict:
    did, cur = desk["id"], desk["current_date"]
    seen = visible_through(data, desk)
    decisions = {r["symbol"]: r for r in store.rows("SELECT * FROM exit_decisions WHERE desk_id = ? AND trade_date = ?",
                                                    (did, cur))} if desk["phase"] == "morning" else {}
    pending = store.rows("SELECT * FROM pending_orders WHERE desk_id = ? AND trade_date = ? AND status = 'pending' "
                         "ORDER BY id", (did, cur))
    exits = {o["symbol"] for o in pending if o["action"] == "exit"}
    positions = []
    for p in desk["book"]["positions"]:
        s = p["signal"]["symbol"]
        close = data.raw_close(s, seen) if seen else None
        close = close if close is not None else p["entry_price"]
        pnl = (close - p["entry_price"]) * p["quantity"] - p["entry_cost"]
        positions.append({
            "symbol": s, "name": data.names.get(s, s), "quantity": p["quantity"], "entry_price": p["entry_price"],
            "entry_date": p["entry_date"], "stop": p["stop"], "target": p["signal"]["target"], "days": p["days"],
            "max_hold_days": p["signal"]["max_hold_days"], "source": p["signal"]["source"],
            "last_close": close, "pnl": pnl, "return_pct": pnl / (p["entry_price"] * p["quantity"]),
            "pending_exit": s in exits, "exit_decision": decisions.get(s)})
    last = store.scalar("SELECT MAX(trade_date) FROM orders WHERE desk_id = ?", (did,))
    orders = store.rows("SELECT * FROM orders WHERE desk_id = ? AND trade_date = ? ORDER BY id DESC", (did, last)) \
        if last else []
    session = store.scalar("SELECT MAX(trade_date) FROM equity WHERE desk_id = ?", (did,))
    booked = store.rows("SELECT net_pnl FROM trades WHERE desk_id = ? AND exit_date = ?", (did, session)) if session else []
    eq = store.rows("SELECT equity FROM equity WHERE desk_id = ? ORDER BY trade_date DESC LIMIT 2", (did,))
    day_pnl = (eq[0]["equity"] - (eq[1]["equity"] if len(eq) > 1 else desk["capital"])) if eq else 0.0
    filled = [o for o in orders if o["status"] == "filled"]
    return {"visible_through": str(seen) if seen else None, "session": session, "positions": positions,
            "pending": pending, "orders": orders, "orders_date": last,
            "kpis": {"day_pnl": day_pnl, "booked": sum(t["net_pnl"] for t in booked), "closed_today": len(booked),
                     "open_pnl": sum(p["pnl"] for p in positions), "orders": len(filled),
                     "turnover": sum((o["price"] or 0) * o["quantity"] for o in filled)}}


def trades(store: Store, data: DeskData, desk_id: int) -> list[dict]:
    rows = store.rows("SELECT * FROM trades WHERE desk_id = ? ORDER BY exit_date DESC, id DESC", (desk_id,))
    for r in rows:
        r["name"] = data.names.get(r["symbol"], r["symbol"])
    return rows


def days(store: Store, data: DeskData, desk_id: int) -> list[dict]:
    """Every trading day the desk has reached, oldest first (for period filters and the day picker)."""
    return store.rows("SELECT d.trade_date, d.llm_calls, d.fallbacks, d.closed_at IS NOT NULL AS closed, "
                      "(SELECT COUNT(*) FROM plans p WHERE p.desk_id = d.desk_id AND p.trade_date = d.trade_date) AS plans, "
                      "(SELECT COALESCE(SUM(accepted),0) FROM plans p WHERE p.desk_id = d.desk_id "
                      "AND p.trade_date = d.trade_date) AS accepted FROM days d WHERE d.desk_id = ? ORDER BY d.trade_date",
                      (desk_id,))
