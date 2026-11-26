"""Agent runs (tasks 13.4-13.6): one day, or a date range with resume after a crash.

    python -m src.run --date 2023-06-01
    python -m src.run --from 2023-01-01 --to 2023-03-31 --run-id agents-2023q1
    python -m src.run --date 2023-06-01 --symbols RELIANCE,INFY,TCS

Writes data/runs/<run_id>/: plans.jsonl (every plan with the Critic's verdict), exits.jsonl (Exit Monitor
decisions), positions.json (open positions carried day to day), days.txt (finished days, for resume),
trace.jsonl and manifest.json. `plans.parquet` collects the plans at the end. LLM responses are cached, so
re-running a finished range costs nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.base import AgentContext
from src.agents.context import RunData, trading_dates
from src.config import Config
from src.contracts import Signal
from src.graph.build import day_graph
from src.llm.client import LLMClient
from src.tracing.writer import TraceWriter, write_run_manifest


def run_dir(cfg: Config, run_id: str) -> Path:
    return cfg.data_dir / "runs" / run_id


def run_days(cfg: Config, start, end=None, *, run_id: str | None = None, symbols: list[str] | None = None,
             data: RunData | None = None, log=print) -> Path:
    end = end or start
    run_id = run_id or f"agents-{pd.Timestamp(start):%Y%m%d}-{pd.Timestamp(end):%Y%m%d}"
    folder = run_dir(cfg, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    data = data or RunData.load(cfg, start, end, symbols)
    tracer = TraceWriter(folder, run_id)
    llm = LLMClient(cfg, tracer)
    done_file, pos_file = folder / "days.txt", folder / "positions.json"
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.exists() else set()
    positions: list[dict] = json.loads(pos_file.read_text(encoding="utf-8")) if pos_file.exists() else []
    days = trading_dates(data, start, end)
    log(f"{run_id}: {len(days)} trading day(s), {len(done)} already done; LLM provider {llm.provider}")

    for d in days:
        if str(d) in done:
            continue
        ctx = AgentContext(cfg, run_id, llm, tracer, data.day(d))
        out = day_graph(ctx).invoke({"positions": positions})
        positions, exit_rows = apply_exits(positions, out.get("exits", []), d)
        plan_rows = [plan_row(r, d) for r in out.get("results", [])]
        held = {p["symbol"] for p in positions}
        for row in plan_rows:
            if row["accepted"] and row["symbol"] not in held:
                positions.append({"symbol": row["symbol"], "entry_date": str(d), "entry": row["entry"],
                                  "stop": row["stop"], "target": row["target"], "max_hold_days": row["max_hold_days"]})
        append_jsonl(folder / "plans.jsonl", plan_rows)
        append_jsonl(folder / "exits.jsonl", exit_rows)
        pos_file.write_text(json.dumps(positions, indent=1), encoding="utf-8")
        with done_file.open("a", encoding="utf-8") as f:
            f.write(f"{d}\n")
        accepted = sum(r["accepted"] for r in plan_rows)
        log(f"  {d}: {len(out['planner'].selected)} analysed, {accepted} accepted trade(s), "
            f"{len(exit_rows)} exit decision(s), {len(positions)} open")

    plans = read_jsonl(folder / "plans.jsonl")
    if plans:
        pd.DataFrame(plans).to_parquet(folder / "plans.parquet", index=False)
    write_run_manifest(cfg, folder, run_id, start=str(start), end=str(end), symbols=symbols,
                       days=len(days), plans=len(plans), accepted=sum(p["accepted"] for p in plans))
    return folder


def apply_exits(positions: list[dict], decisions, d: date) -> tuple[list[dict], list[dict]]:
    rows, keep = [], []
    by_symbol = {x.symbol: x for x in decisions}
    for p in positions:
        dec = by_symbol.get(p["symbol"])
        if dec is None or dec.action == "hold":
            keep.append(p)
        elif dec.action == "tighten_stop":
            keep.append({**p, "stop": dec.new_stop})
        if dec is not None:
            rows.append({"trade_date": str(d), "symbol": p["symbol"], "action": dec.action, "new_stop": dec.new_stop,
                         "reason": dec.reason})
    return keep, rows


def plan_row(result: dict, d: date) -> dict:
    plan, verdict = result["plan"], result["verdict"]
    return {"trade_date": str(d), "symbol": plan.symbol, "direction": plan.direction, "entry": plan.entry,
            "stop": plan.stop, "target": plan.target, "max_hold_days": plan.max_hold_days,
            "confidence": plan.confidence, "vol_forecast": plan.vol_forecast, "setup_type": plan.setup_type,
            "base_p_up": plan.base_forecast_p_up, "verdict": verdict.verdict, "rounds": result.get("rounds", 1),
            "accepted": verdict.verdict == "accept" and plan.direction == "long",
            "failed_checks": [c.code for c in verdict.checks if not c.passed], "reasons": result.get("reasons", []),
            "rationale": plan.rationale, "evidence_ids": plan.evidence_ids,
            "bull_confidence": result["bull"].confidence, "bear_confidence": result["bear"].confidence}


def signals_from_run(folder: Path) -> list[Signal]:
    """Accepted long plans as backtester signals (one per plan; the run ID is the source)."""
    run_id = Path(folder).name
    return [Signal(symbol=p["symbol"], signal_date=date.fromisoformat(p["trade_date"]), stop=p["stop"],
                   target=p["target"], max_hold_days=p["max_hold_days"], source=run_id, confidence=p["confidence"])
            for p in read_jsonl(Path(folder) / "plans.jsonl") if p["accepted"]]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m src.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=date.fromisoformat)
    ap.add_argument("--from", dest="start", type=date.fromisoformat)
    ap.add_argument("--to", dest="end", type=date.fromisoformat)
    ap.add_argument("--symbols", help="comma-separated; default: every Nifty 50 stock")
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    start = args.date or args.start
    if start is None:
        ap.error("give --date or --from/--to")
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    folder = run_days(Config.load(), start, args.end or args.date or start, run_id=args.run_id, symbols=symbols)
    print(f"Run written to {folder}")


if __name__ == "__main__":
    main()
