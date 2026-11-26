"""Background jobs: one worker thread runs the desk's slow work (agent mornings can take minutes with an LLM).

Jobs live in the `jobs` table, so the app can show progress and a person can cancel between days. One job at a time
per desk; the worker takes jobs in order. A job still marked running when the server starts was interrupted: it is
marked failed, and the desk itself is intact because every morning and session is its own transaction.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable

from src.config import Config
from src.desk import service
from src.desk.data import DeskData
from src.desk.store import Store, insert, now, update

ACTIVE = ("queued", "running", "cancelling")


def submit(store: Store, desk_id: int, kind: str, params: dict | None = None, total: int = 1) -> int:
    if store.scalar(f"SELECT 1 FROM jobs WHERE desk_id = ? AND status IN {ACTIVE}", (desk_id,)):
        raise service.DeskError("This desk is busy with another job. Wait for it or cancel it.")
    with store.tx() as con:
        return insert(con, "jobs", {"desk_id": desk_id, "kind": kind, "params": params or {}, "status": "queued",
                                    "message": "Waiting to start", "total": total, "created_at": now()})


def cancel(store: Store, job_id: int) -> None:
    with store.tx() as con:
        n = con.execute("UPDATE jobs SET status = CASE status WHEN 'queued' THEN 'cancelled' ELSE 'cancelling' END, "
                        "finished_at = CASE status WHEN 'queued' THEN ? ELSE finished_at END "
                        "WHERE id = ? AND status IN ('queued', 'running')", (now(), job_id)).rowcount
    if not n:
        raise service.DeskError("That job is not running.")


def recover(store: Store) -> None:
    with store.tx() as con:
        con.execute("UPDATE jobs SET status = 'failed', error = 'Interrupted: the server stopped while it ran.', "
                    "finished_at = ? WHERE status IN ('running', 'cancelling')", (now(),))


class Worker(threading.Thread):
    def __init__(self, store: Store, cfg: Config, data: Callable[[], DeskData],
                 live_data: Callable[[], DeskData] | None = None):
        super().__init__(name="desk-worker", daemon=True)
        self.store, self.cfg, self.data = store, cfg, data
        self.live_data = live_data
        self.wake = threading.Event()
        self.stopping = False

    def run(self) -> None:
        while not self.stopping:
            job = self.store.row("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
            if job is None:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            self.execute(job)

    def stop(self) -> None:
        self.stopping = True
        self.wake.set()

    def finish(self, jid: int, status: str, message: str, done: int) -> None:
        with self.store.tx() as con:
            update(con, "jobs", {"status": status, "message": message, "done": done, "finished_at": now()},
                   "id = ?", (jid,))

    def execute(self, job: dict) -> None:
        jid = job["id"]
        with self.store.tx() as con:
            update(con, "jobs", {"status": "running", "started_at": now(), "message": "Starting"}, "id = ?", (jid,))
        last = [0.0]

        def progress(msg: str, done: int | None = None, total: int | None = None) -> None:
            t = time.monotonic()
            if t - last[0] < 0.4 and done is None:
                return
            last[0] = t
            values = {"message": msg[:200]}
            if done is not None:
                values["done"] = done
            if total is not None:
                values["total"] = total
            with self.store.tx() as con:
                update(con, "jobs", values, "id = ?", (jid,))

        def cancelled() -> bool:
            return self.store.scalar("SELECT status FROM jobs WHERE id = ?", (jid,)) == "cancelling"

        try:
            desk = self.store.row("SELECT mode FROM desks WHERE id = ?", (job["desk_id"],)) or {}
            p = job["params"]
            done = job["total"]
            if job["kind"] == "live":
                from src.desk import live

                out = live.daily(self.cfg, self.store, [job["desk_id"]], update_data=bool(p.get("update", True)),
                                 progress=progress)
                d = out["desks"][job["desk_id"]]
                result = (f"Suggestions for {d['trade_date']} are ready: {d['accepted']} accepted, "
                          f"{d['holdings']} holding(s) checked.")
                self.finish(jid, "done", result, done)
                return
            data = (self.live_data or self.data)() if desk.get("mode") == "live" else self.data()
            if job["kind"] == "create":
                service.run_morning(self.store, data, self.cfg, job["desk_id"], progress)
                result = "First morning's plans are ready."
            elif job["kind"] == "advance":
                done = service.advance(self.store, data, self.cfg, job["desk_id"], int(p.get("days", 1)), progress,
                                       cancelled)
                result = f"Advanced {done} trading day{'s' if done != 1 else ''}."
            elif job["kind"] == "analyse":
                out = service.analyse_stock(self.store, data, self.cfg, job["desk_id"], p["symbol"], progress)
                result = f"{out['symbol']} analysed" + (": long plan accepted." if out["accepted"] else ".")
            else:
                raise service.DeskError(f"Unknown job kind {job['kind']!r}.")
            self.finish(jid, "cancelled" if cancelled() else "done", result, done)
        except Exception as exc:  # noqa: BLE001 - any error fails the job; the desk stays at its last whole step
            msg = str(exc) if isinstance(exc, service.DeskError) else f"{type(exc).__name__}: {exc}"
            with self.store.tx() as con:
                update(con, "jobs", {"status": "failed", "error": msg[:2000], "message": "Failed",
                                     "finished_at": now()}, "id = ?", (jid,))
            if not isinstance(exc, service.DeskError):
                traceback.print_exc()
