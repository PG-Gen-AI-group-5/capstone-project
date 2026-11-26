"""Windows Task Scheduler entries for the live suggester (scripts/live.cmd runs python -m src.live).

    Live daily   weekdays at live.daily_time (08:00 IST): download, rebuild, forecast, suggest
    Live refit   weekly on live.refit_day at live.refit_time: refit the live TFT and LSTM (about an hour)

The tasks run while you are logged in and the PC is on. Remove them with: python -m src.live schedule remove
"""
from __future__ import annotations

import csv
import io
import os
import subprocess

from src.config import Config
from src.live.env import live_settings

TASKS = {"daily": r"TradingWithAI\Live daily", "refit": r"TradingWithAI\Live refit"}


def script(cfg: Config) -> str:
    return str(cfg.root / "scripts" / "live.cmd")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True, timeout=30, check=False)


def install(cfg: Config) -> list[str]:
    if os.name != "nt":
        raise RuntimeError("Scheduling uses the Windows Task Scheduler; on Linux or macOS add a cron entry for "
                           "scripts/live.cmd's command instead.")
    s = live_settings(cfg)
    runs = [(TASKS["daily"], ["/SC", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI", "/ST", s["daily_time"]], "daily"),
            (TASKS["refit"], ["/SC", "WEEKLY", "/D", s["refit_day"], "/ST", s["refit_time"]], "refit")]
    out = []
    for name, when, command in runs:
        r = _run(["/Create", "/F", "/TN", name, *when, "/TR", f'"{script(cfg)}" {command}'])
        if r.returncode != 0:
            raise RuntimeError(f"schtasks failed for {name}: {r.stderr.strip() or r.stdout.strip()}")
        out.append(f"{name}: {' '.join(when[1:])}")
    return out


def remove() -> list[str]:
    out = []
    for name in TASKS.values():
        r = _run(["/Delete", "/F", "/TN", name])
        out.append(f"{name}: {'removed' if r.returncode == 0 else 'not found'}")
    return out


def installed() -> dict:
    """{task: {next_run, status}} for the tasks that exist (empty off Windows)."""
    if os.name != "nt":
        return {}
    out = {}
    for key, name in TASKS.items():
        try:
            r = _run(["/Query", "/TN", name, "/FO", "CSV", "/NH"])
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip():
            row = next(csv.reader(io.StringIO(r.stdout.strip().splitlines()[-1])))
            out[key] = {"task": name, "next_run": row[1] if len(row) > 1 else "", "status": row[2] if len(row) > 2 else ""}
    return out
