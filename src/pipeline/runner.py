"""Run pipeline stages in order, skip the ones that are up to date, and report each stage's state.

After a stage runs, its manifest gets a `pipeline` record: the hash of its config keys, the time of each
input it read, and how long it took. `status` compares that record with the inputs and config now.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import Config
from src.pipeline.stages import STAGES, TARGETS, RunOptions, Stage
from src.store.db import config_hash


def plan(targets: list[str], stages: list[Stage] = STAGES, targets_map: dict = TARGETS) -> tuple[list[Stage], set]:
    """Stages to consider, in run order, and the names asked for explicitly (those always run)."""
    by_name = {s.name: s for s in stages}
    wanted, explicit = set(), set()
    for t in targets:
        if t in targets_map:
            wanted.update(targets_map[t])
        elif t in by_name:
            wanted.add(t)
            explicit.add(t)
        else:
            raise ValueError(f"unknown stage or target {t!r}; choose from {sorted(targets_map) + list(by_name)}")
    return [s for s in stages if s.name in wanted], explicit


def manifest_path(cfg: Config, stage: Stage) -> Path:
    return cfg.derived_dir / stage.manifest / "_manifest.json"


def read_manifest(cfg: Config, stage: Stage) -> dict:
    path = manifest_path(cfg, stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def config_value(cfg: Config, dotted: str):
    value = cfg.raw
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def stage_config_hash(cfg: Config, stage: Stage) -> str:
    return config_hash({k: config_value(cfg, k) for k in stage.config_keys})


def input_time(cfg: Config, ref: str, stages: list[Stage]) -> datetime | None:
    """When an input last changed (UTC): file modification time, or the built_at of a stage's manifest."""
    kind, name = ref.split(":", 1)
    if kind == "stage":
        stage = next(s for s in stages if s.name == name)
        built = read_manifest(cfg, stage).get("built_at")
        return datetime.fromisoformat(built) if built else None
    paths = list((cfg.lake_dir / name).glob("*.parquet")) if kind == "lake" else [cfg.manual_dir / name]
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return datetime.fromtimestamp(max(times), timezone.utc) if times else None


def staleness(cfg: Config, stage: Stage, opts: RunOptions, stages: list[Stage] = STAGES) -> str | None:
    """Why the stage needs to run, or None when it is up to date."""
    manifest = read_manifest(cfg, stage)
    if not manifest.get("built_at"):
        return "never built"
    record = manifest.get("pipeline")
    if record is None:
        return "no pipeline record (built by another command)"
    if record.get("config_hash") != stage_config_hash(cfg, stage):
        return f"settings changed ({', '.join(stage.config_keys)})"
    built = datetime.fromisoformat(manifest["built_at"])
    for ref in stage.reads:
        t = input_time(cfg, ref, stages)
        if t is None:
            return f"input {ref} is missing"
        if t > built:
            return f"{ref} changed after the last build"
    return stage.fresh(cfg, manifest, opts) if stage.fresh else None


def record_run(cfg: Config, stage: Stage, stages: list[Stage], runtime: float) -> None:
    """Add the pipeline record to the manifest the stage wrote."""
    path = manifest_path(cfg, stage)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "built_at": datetime.now(timezone.utc).isoformat(), "stage": stage.name}
    inputs = {ref: (t.isoformat() if (t := input_time(cfg, ref, stages)) else None) for ref in stage.reads}
    manifest["pipeline"] = {"config_hash": stage_config_hash(cfg, stage), "inputs": inputs,
                            "runtime_sec": round(runtime, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def run(cfg: Config, targets: list[str], opts: RunOptions | None = None, *, skip: tuple[str, ...] = (),
        stages: list[Stage] = STAGES, targets_map: dict = TARGETS, log=print) -> list[dict]:
    """Run the stages for `targets` in order. A stage named explicitly always runs; stages from a group
    target (all, data, models) run only when out of date, unless opts.force."""
    opts = opts or RunOptions()
    todo, explicit = plan(targets, stages, targets_map)
    results = []
    for stage in todo:
        if stage.name in skip:
            log(f"{stage.name}: skipped")
            results.append({"stage": stage.name, "action": "skipped"})
            continue
        reason = staleness(cfg, stage, opts, stages)
        if reason is None and stage.name not in explicit and not opts.force:
            log(f"{stage.name}: up to date")
            results.append({"stage": stage.name, "action": "up to date"})
            continue
        why = reason or ("asked for" if stage.name in explicit else "forced")
        log(f"{stage.name}: running ({why})")
        t0 = time.time()
        summary = stage.run(cfg, opts)
        record_run(cfg, stage, stages, time.time() - t0)
        log(f"{stage.name}: done in {time.time() - t0:.0f}s {summary or ''}")
        results.append({"stage": stage.name, "action": "ran", "reason": why, "summary": summary})
    return results


def status(cfg: Config, opts: RunOptions | None = None, stages: list[Stage] = STAGES) -> pd.DataFrame:
    """One row per stage: tasks, last build (IST), rows, and whether it is up to date (and why not)."""
    opts = opts or RunOptions()
    rows = []
    for stage in stages:
        manifest = read_manifest(cfg, stage)
        built = manifest.get("built_at")
        built_ist = (pd.Timestamp(built).tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M") if built else "")
        reason = staleness(cfg, stage, opts, stages)
        rows.append({"stage": stage.name, "tasks": stage.tasks, "last_built_ist": built_ist,
                     "rows": manifest.get("rows"), "state": "up to date" if reason is None else reason})
    return pd.DataFrame(rows)
