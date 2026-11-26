"""Command line for model building.

    python -m src.models prepare                         adjusted prices -> features -> labels
    python -m src.models train --model arima             walk-forward on the validation period
    python -m src.models train --model all
    python -m src.models train --model all --period inner      walk-forward folds inside 2016-2022
    python -m src.models evaluate                        metrics table for every model with forecasts
    python -m src.models evaluate --by vix_regime        the same, split by market regime
    python -m src.models train --model tft --period test --freeze      task 16.9 only
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import Config
from src.models.registry import MODELS

PERIODS = ["validation", "inner", "test"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.models", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="build adjusted prices, features and labels")
    t = sub.add_parser("train", help="walk-forward train and predict")
    t.add_argument("--model", required=True, help=f"one of {', '.join(MODELS)} or 'all'")
    t.add_argument("--period", default="validation", choices=PERIODS)
    t.add_argument("--freeze", action="store_true", help="allow the test period (task 16.9 only)")
    e = sub.add_parser("evaluate", help="metrics for every model with saved forecasts")
    e.add_argument("--period", default="validation", choices=PERIODS)
    e.add_argument("--by", choices=["trend_regime", "vix_regime"], help="also split by market regime")
    args = ap.parse_args(argv)
    cfg = Config.load()

    if args.cmd == "prepare":
        from src.models.data import prepare
        prepare(cfg)
    elif args.cmd == "train":
        from src.models.data import load_panel
        from src.models.walkforward import run
        panel = load_panel(cfg)
        names = list(MODELS) if args.model == "all" else [args.model]
        for name in names:
            run(name, cfg, args.period, panel=panel, allow_test=args.freeze)
    elif args.cmd == "evaluate":
        from src.evaluation.metrics import evaluate, majority_guess, with_regimes
        from src.models.data import label_frame, load_panel, regimes
        from src.models.walkforward import load_forecasts
        panel = load_panel(cfg)
        fc = load_forecasts(cfg, args.period)
        labels = with_regimes(label_frame(panel), regimes(cfg))
        by = ["model", "horizon_days"] + ([args.by] if args.by else [])
        table = evaluate(fc, labels, by=by)
        ref = majority_guess(labels[labels["trade_date"].isin(fc["trade_date"].unique())], by[1:])
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(table.round(4).to_string(index=False))
            print()
            print("Constant majority guess for the same days (hindsight reference):")
            print(ref.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
