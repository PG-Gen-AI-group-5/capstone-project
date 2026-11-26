"""The desk's HTTP API (FastAPI) and the web frontend in frontend/.

    python -m src.api                  # http://127.0.0.1:8000

Reads come straight from the desk database and the lake; slow work (agent mornings, advancing days, analysing an
added stock) runs as background jobs that the page polls.
"""
from __future__ import annotations

import math
import threading
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import Config
from src.desk import jobs, service, views
from src.desk.data import DeskData, desk_settings, desk_window
from src.desk.data import shared as shared_data
from src.desk.store import Store


class CreateDesk(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    mode: str = "replay"                                   # replay | live
    start_date: date | None = None                         # replay desks only
    end_date: date | None = None
    capital: float | None = Field(default=None, gt=0)
    provider: str = "offline"


class Advance(BaseModel):
    days: int = Field(default=1, ge=1, le=260)


class AddStock(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)


class PlaceOrder(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    action: str


class Bought(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    entry_date: date
    entry_price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    stop: float | None = Field(default=None, gt=0)
    target: float | None = Field(default=None, gt=0)
    max_hold_days: int | None = Field(default=None, ge=1, le=30)


class Sold(BaseModel):
    exit_date: date
    exit_price: float = Field(gt=0)
    reason: str = "manual"


class EditHolding(BaseModel):
    stop: float | None = Field(default=None, gt=0)
    target: float | None = Field(default=None, gt=0)


def clean(value):
    """JSON without NaN or infinity (the browser's JSON.parse rejects them)."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    return value


def create_app(cfg: Config | None = None, store: Store | None = None, data: DeskData | None = None,
               run_worker: bool = True, live_data: DeskData | None = None) -> FastAPI:
    from src.desk import live

    cfg = cfg or Config.load()
    store = store or Store(cfg.data_dir / desk_settings(cfg)["db"])
    get_data = (lambda: data) if data is not None else (lambda: shared_data(cfg))
    get_live = (lambda: live_data) if live_data is not None else (lambda: live.live_data(cfg))
    worker = jobs.Worker(store, cfg, get_data, get_live)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        jobs.recover(store)
        if run_worker:
            worker.start()
            threading.Thread(target=get_data, name="desk-data-preload", daemon=True).start()
        yield
        worker.stop()

    app = FastAPI(title="Trading with AI", lifespan=lifespan)
    app.state.store, app.state.cfg, app.state.worker = store, cfg, worker

    @app.exception_handler(service.DeskError)
    async def desk_error(request: Request, exc: service.DeskError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def ok(value, status: int = 200):
        return JSONResponse(clean(jsonable_encoder(value)), status_code=status)

    def desk_or_404(desk_id: int) -> dict:
        desk = store.row("SELECT * FROM desks WHERE id = ?", (desk_id,))
        if desk is None:
            raise HTTPException(404, f"No desk {desk_id}.")
        return desk

    def wake() -> None:
        worker.wake.set()

    def data_for(desk: dict) -> DeskData:
        return get_live() if desk.get("mode") == "live" else get_data()

    def summary(desk: dict) -> dict:
        try:
            return views.desk_summary(store, data_for(desk), desk)
        except service.DeskError:                 # a live desk before its first update
            job = store.row("SELECT * FROM jobs WHERE desk_id = ? ORDER BY id DESC LIMIT 1", (desk["id"],))
            return {"id": desk["id"], "name": desk["name"], "mode": desk.get("mode"), "provider": desk["provider"],
                    "model": desk["model"], "phase": desk["phase"], "current_date": desk["current_date"],
                    "capital": desk["capital"], "job": job, "live": {}, "open_positions": 0, "days_done": 0}

    # ---- the app's settings and status
    @app.get("/api/meta")
    def meta():
        d = get_data()
        lo, hi = desk_window(cfg)
        return ok({"providers": service.provider_status(cfg), "window": {"start": lo, "end": hi},
                   "capital": desk_settings(cfg)["capital"],
                   "symbols": [{"symbol": s, "name": n} for s, n in sorted(d.names.items())],
                   "base_model": d.run.base_model, "vol_model": desk_settings(cfg)["vol_model"],
                   "portfolio": (cfg.get("backtest") or {}).get("portfolio"), "llm": {
                       k: (cfg.get("llm") or {}).get(k) for k in ("daily_call_limit", "daily_budget_usd", "training_cutoff")}})

    @app.get("/api/pipeline")
    def pipeline():
        from src.pipeline.runner import status
        return ok({"stages": status(cfg).to_dict(orient="records")})

    # ---- desks
    @app.get("/api/desks")
    def list_desks():
        return ok([summary(x) for x in store.rows("SELECT * FROM desks ORDER BY id")])

    @app.post("/api/desks", status_code=201)
    def create(body: CreateDesk):
        if body.mode == "live":
            desk_id = live.create_live_desk(store, cfg, body.name, body.capital, body.provider)
            job = jobs.submit(store, desk_id, "live", {"update": True})
        else:
            if body.start_date is None:
                raise service.DeskError("Pick the first day for the replay.")
            desk_id = service.create_desk(store, get_data(), cfg, body.name, body.start_date, body.capital,
                                          body.provider, body.end_date)
            job = jobs.submit(store, desk_id, "create")
        wake()
        return ok({"desk": summary(desk_or_404(desk_id)), "job_id": job}, 201)

    @app.get("/api/desks/{desk_id}")
    def desk(desk_id: int):
        return ok(summary(desk_or_404(desk_id)))

    # ---- live desks
    @app.post("/api/desks/{desk_id}/refresh")
    def refresh(desk_id: int):
        live.live_desk(store, desk_id)
        job = jobs.submit(store, desk_id, "live", {"update": True})
        wake()
        return ok({"job_id": job})

    @app.get("/api/live")
    def live_status():
        from src.live import models as live_models
        from src.live import update as live_update
        from src.live.env import live_config, live_settings
        from src.live.schedule import installed

        lcfg = live_config(cfg)
        return ok({"lake": live_update.lake_status(cfg), "built": live_update.built(cfg),
                   "models": {m: live_models.fit_info(lcfg, m) for m in live_settings(cfg)["models"]},
                   "schedule": installed(), "settings": live_settings(cfg)})

    @app.get("/api/desks/{desk_id}/holdings")
    def holdings(desk_id: int, status: str = "open"):
        desk = live.live_desk(store, desk_id)
        try:
            d = get_live()
        except service.DeskError:
            d = None
        return ok(live.holding_rows(store, d, desk, "closed" if status == "closed" else "open"))

    @app.post("/api/desks/{desk_id}/holdings", status_code=201)
    def bought(desk_id: int, body: Bought):
        hid = live.add_holding(store, get_live(), desk_id, body.symbol, body.entry_date, body.entry_price,
                               body.quantity, body.stop, body.target, body.max_hold_days)
        return ok({"holding_id": hid}, 201)

    @app.patch("/api/desks/{desk_id}/holdings/{holding_id}")
    def edit_holding(desk_id: int, holding_id: int, body: EditHolding):
        live.update_holding(store, desk_id, holding_id, body.stop, body.target)
        return ok({"holding_id": holding_id})

    @app.post("/api/desks/{desk_id}/holdings/{holding_id}/close")
    def sold(desk_id: int, holding_id: int, body: Sold):
        tid = live.close_holding(store, get_live(), desk_id, holding_id, body.exit_date, body.exit_price, body.reason)
        return ok({"trade_id": tid})

    @app.delete("/api/desks/{desk_id}/holdings/{holding_id}")
    def delete_holding(desk_id: int, holding_id: int):
        live.delete_holding(store, desk_id, holding_id)
        return ok({"deleted": holding_id})

    @app.delete("/api/desks/{desk_id}")
    def delete(desk_id: int):
        service.delete_desk(store, desk_id)
        return ok({"deleted": desk_id})

    @app.post("/api/desks/{desk_id}/advance")
    def advance(desk_id: int, body: Advance):
        desk = desk_or_404(desk_id)
        service.replay_only(desk)
        if desk["phase"] == "finished":
            raise service.DeskError("This desk has reached its last day.")
        job = jobs.submit(store, desk_id, "advance", {"days": body.days}, total=body.days)
        wake()
        return ok({"job_id": job})

    # ---- jobs
    @app.get("/api/jobs/{job_id}")
    def job(job_id: int):
        j = store.row("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if j is None:
            raise HTTPException(404, "No such job.")
        return ok(j)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        jobs.cancel(store, job_id)
        return ok({"cancelled": job_id})

    # ---- the morning: watchlist, stocks, orders
    @app.get("/api/desks/{desk_id}/watchlist")
    def watchlist(desk_id: int, date: str | None = None):
        desk = desk_or_404(desk_id)
        return ok(views.watchlist(store, data_for(desk), desk, date))

    @app.post("/api/desks/{desk_id}/watchlist")
    def add_stock(desk_id: int, body: AddStock):
        desk_or_404(desk_id)
        symbol = body.symbol.strip().upper().removesuffix(".NS")
        if symbol not in get_data().names:
            raise service.DeskError(f"{symbol} isn't in the desk's universe (the Nifty 50 stocks).")
        outcome = service.add_to_watchlist(store, desk_id, symbol)
        if outcome == "restored":
            return ok({"symbol": symbol, "restored": True})
        job = jobs.submit(store, desk_id, "analyse", {"symbol": symbol})
        wake()
        return ok({"symbol": symbol, "job_id": job})

    @app.delete("/api/desks/{desk_id}/watchlist/{symbol}")
    def remove_stock(desk_id: int, symbol: str):
        desk_or_404(desk_id)
        service.remove_from_watchlist(store, desk_id, symbol.upper())
        return ok({"symbol": symbol.upper(), "removed": True})

    @app.get("/api/desks/{desk_id}/stocks/{symbol}")
    def stock(desk_id: int, symbol: str, date: str | None = None):
        try:
            desk = desk_or_404(desk_id)
            return ok(views.stock_detail(store, data_for(desk), desk, symbol.upper(), date))
        except KeyError:
            raise HTTPException(404, f"No data for {symbol.upper()} on that day.") from None

    @app.get("/api/desks/{desk_id}/trace")
    def trace(desk_id: int, date: str, symbol: str | None = None):
        desk_or_404(desk_id)
        return ok(views.trace(store, desk_id, date, symbol.upper() if symbol else None))

    @app.post("/api/desks/{desk_id}/orders", status_code=201)
    def place_order(desk_id: int, body: PlaceOrder):
        desk_or_404(desk_id)
        return ok({"order_id": service.place_order(store, desk_id, body.symbol, body.action)}, 201)

    @app.delete("/api/desks/{desk_id}/orders/{order_id}")
    def cancel_order(desk_id: int, order_id: int):
        service.cancel_order(store, desk_id, order_id)
        return ok({"cancelled": order_id})

    # ---- book, history, analysis, feed
    @app.get("/api/desks/{desk_id}/book")
    def book(desk_id: int):
        desk = desk_or_404(desk_id)
        return ok(views.book(store, data_for(desk), desk))

    @app.get("/api/desks/{desk_id}/trades")
    def trades(desk_id: int):
        return ok(views.trades(store, data_for(desk_or_404(desk_id)), desk_id))

    @app.get("/api/desks/{desk_id}/equity")
    def equity(desk_id: int):
        desk_or_404(desk_id)
        return ok(store.rows("SELECT * FROM equity WHERE desk_id = ? ORDER BY trade_date", (desk_id,)))

    @app.get("/api/desks/{desk_id}/days")
    def days(desk_id: int):
        return ok(views.days(store, data_for(desk_or_404(desk_id)), desk_id))

    @app.get("/api/desks/{desk_id}/events")
    def events(desk_id: int, after: int = 0, limit: int = 120):
        desk_or_404(desk_id)
        return ok(store.rows("SELECT * FROM events WHERE desk_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
                             (desk_id, after, min(max(limit, 1), 500))))

    frontend = cfg.root / "frontend"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
