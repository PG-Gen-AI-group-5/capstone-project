"""Command line for the pipeline: run stages in order and see what is out of date.

    python -m src.pipeline status                          every stage, its last build and whether it is current
    python -m src.pipeline run all                         ingest -> prices_adj -> features -> labels -> forecasts -> evaluate
    python -m src.pipeline run all --no-ingest             the same without downloading (offline)
    python -m src.pipeline run data                        ingest, prices_adj, features, labels (the evening job)
    python -m src.pipeline run models --period validation --period inner
    python -m src.pipeline run features labels             named stages always run
    python -m src.pipeline run forecasts --models lstm,tft --force      retrain these models

Group targets (all, data, models) skip stages that are up to date; --force runs them anyway. Forecasts
reuse saved models whose settings and inputs are unchanged, so only changed models retrain. The test
period needs --period test --freeze and is run once, by task 16.9.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import Config
from src.pipeline.runner import run, status
from src.pipeline.stages import STAGES, TARGETS, RunOptions

PERIODS = ["validation", "inner", "test"]


def options(args) -> RunOptions:
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    return RunOptions(periods=args.period or ["validation"], models=models, force=args.force, freeze=args.freeze)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m src.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("targets", nargs="+", help=f"{', '.join(TARGETS)} or stages: "
                                                      f"{', '.join(s.name for s in STAGES)}")
            p.add_argument("--no-ingest", action="store_true", help="skip downloading new data")
            p.add_argument("--force", action="store_true", help="run stages even when up to date; retrain models")
            p.add_argument("--freeze", action="store_true", help="allow the test period (task 16.9 only)")
        p.add_argument("--period", action="append", choices=PERIODS, help="model period (repeatable)")
        p.add_argument("--models", help="comma-separated model names (default: all)")
    args = ap.parse_args(argv)
    cfg = Config.load()
    if args.cmd == "status":
        args.force = args.freeze = False
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(status(cfg, options(args)).to_string(index=False))
    else:
        run(cfg, args.targets, options(args), skip=("ingest",) if args.no_ingest else ())


if __name__ == "__main__":
    main()
