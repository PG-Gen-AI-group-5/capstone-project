"""Pipeline stages: what each reads, where its manifest lives, which settings it depends on, and how it runs.

Inputs are named by kind:
- `lake:<table>`   Parquet files under data/lake/<table>/
- `manual:<file>`  a hand-made file under data/manual/
- `stage:<name>`   the output of an earlier stage (its _manifest.json)

A stage is out of date when it has never been built, when one of its inputs changed after it was built,
when one of its `config_keys` changed, or when its own `fresh` check says so.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.config import Config
from src.store.db import write_manifest, write_table


@dataclass
class RunOptions:
    periods: list[str] = field(default_factory=lambda: ["validation"])
    models: list[str] | None = None          # None = every registered model
    force: bool = False                      # rebuild even when up to date; retrain models
    freeze: bool = False                     # allow the test period (task 16.9 only)


@dataclass(frozen=True)
class Stage:
    name: str
    tasks: str
    reads: tuple[str, ...]
    manifest: str                            # folder under data/derived/ holding the stage's _manifest.json
    config_keys: tuple[str, ...]             # dotted keys of config.yaml, e.g. "features.skip_sessions"
    run: Callable[[Config, RunOptions], dict]
    fresh: Callable[[Config, dict, RunOptions], str | None] | None = None   # extra staleness check
    network: bool = False


# ---------------------------------------------------------------- stage functions

def run_ingest(cfg: Config, opts: RunOptions) -> dict:
    from src.ingest.__main__ import main as ingest_main

    ingest_main(["daily"])
    ingest_main(["views"])
    info = {"job": "daily"}
    write_manifest(cfg.derived_dir / "ingest", stage="ingest", **info)
    return info


def ingest_fresh(cfg: Config, manifest: dict, opts: RunOptions) -> str | None:
    from zoneinfo import ZoneInfo

    from src.timeutil import today_ist

    last = datetime.fromisoformat(manifest["built_at"]).astimezone(ZoneInfo("Asia/Kolkata")).date()
    return None if last >= today_ist() else f"last run on {last}"


def run_prices_adj(cfg: Config, opts: RunOptions) -> dict:
    from src.store import adjust
    return adjust.build(cfg)


def run_features(cfg: Config, opts: RunOptions) -> dict:
    from src.features import build
    return build.build(cfg)


def run_labels(cfg: Config, opts: RunOptions) -> dict:
    from src.labels import labels
    return labels.build(cfg)


def model_names(opts: RunOptions) -> list[str]:
    from src.models.registry import MODELS
    return list(opts.models or MODELS)


def run_forecasts(cfg: Config, opts: RunOptions) -> dict:
    """Walk-forward forecasts per model and period. Saved forecasts are reused while their model version
    (settings + built inputs) is current, so only changed models retrain; `force` retrains all."""
    if "test" in opts.periods and not opts.freeze:
        raise PermissionError("Test-period forecasts are made once, by the freeze task 16.9 (use --freeze).")
    from src.models.data import load_panel
    from src.models.walkforward import model_version, run, run_or_load

    panel = load_panel(cfg)
    versions = {}
    for period in opts.periods:
        for name in model_names(opts):
            if opts.force or period == "test":
                run(name, cfg, period, panel=panel, allow_test=opts.freeze)
            else:
                run_or_load(name, cfg, period, panel=panel)
            versions[f"{name}/{period}"] = model_version(cfg, name, None)
    previous = read_manifest(cfg, "forecasts").get("versions", {})
    info = {"versions": {**previous, **versions}}
    write_manifest(cfg.derived_dir / "forecasts", stage="forecasts", **info)
    return {"models": len(versions)}


def forecasts_fresh(cfg: Config, manifest: dict, opts: RunOptions) -> str | None:
    from src.models.walkforward import model_version

    have = manifest.get("versions", {})
    for period in opts.periods:
        for name in model_names(opts):
            if have.get(f"{name}/{period}") != model_version(cfg, name, None):
                return f"{name} {period} forecasts missing or made with other settings"
    return None


def run_evaluate(cfg: Config, opts: RunOptions) -> dict:
    """Metrics per model and horizon, the constant majority guess and a split by VIX regime, per period."""
    from src.evaluation.metrics import evaluate, majority_guess, with_regimes
    from src.models.data import label_frame, load_panel, regimes
    from src.models.walkforward import load_forecasts

    labels = with_regimes(label_frame(load_panel(cfg)), regimes(cfg))
    done = {}
    for period in opts.periods:
        fc = load_forecasts(cfg, period)
        days = labels[labels["trade_date"].isin(fc["trade_date"].unique())]
        write_table(cfg, "metrics", evaluate(fc, labels), name=period)
        write_table(cfg, "metrics", majority_guess(days), name=f"{period}_majority")
        write_table(cfg, "metrics", evaluate(fc, labels, by=("model", "horizon_days", "vix_regime")),
                    name=f"{period}_by_vix")
        done[period] = sorted(fc["model"].unique())
    previous = read_manifest(cfg, "metrics").get("periods", {})
    write_manifest(cfg.derived_dir / "metrics", stage="evaluate", periods={**previous, **done})
    return {"periods": list(done)}


def evaluate_fresh(cfg: Config, manifest: dict, opts: RunOptions) -> str | None:
    missing = [p for p in opts.periods if p not in manifest.get("periods", {})]
    return f"no metrics for {', '.join(missing)}" if missing else None


def agent_run_id(cfg: Config, period: str) -> str:
    """One run folder per period and LLM provider, so rule-based and LLM runs never mix (finished days are
    skipped on a re-run of the same folder)."""
    return f"agents-{period}-{(cfg.get('llm') or {}).get('provider', 'offline')}"


def run_agents(cfg: Config, opts: RunOptions) -> dict:
    """The agent graph over each period (one run per period; finished days are skipped on a re-run)."""
    if "test" in opts.periods and not opts.freeze:
        raise PermissionError("Agent results on the test period are produced once, by task 18.4 (use --freeze).")
    from src.run import read_jsonl, run_days

    runs = {}
    for period in opts.periods:
        start, end = cfg.period(period)
        folder = run_days(cfg, start, end, run_id=agent_run_id(cfg, period))
        plans = read_jsonl(folder / "plans.jsonl")
        runs[period] = {"run_id": folder.name, "plans": len(plans), "accepted": sum(p["accepted"] for p in plans)}
    previous = read_manifest(cfg, "agent_runs").get("runs", {})
    write_manifest(cfg.derived_dir / "agent_runs", stage="agent_run", runs={**previous, **runs})
    return runs


def agents_fresh(cfg: Config, manifest: dict, opts: RunOptions) -> str | None:
    missing = [p for p in opts.periods if p not in manifest.get("runs", {})]
    return f"no agent run for {', '.join(missing)}" if missing else None


def run_backtest(cfg: Config, opts: RunOptions) -> dict:
    """Models, buy-and-hold, the index and agent runs on the same engine, per period."""
    from src.backtest.run import run_backtests

    extra = list((cfg.get("backtest") or {}).get("agent_runs") or [])
    done = {}
    for period in opts.periods:
        run_id = agent_run_id(cfg, period)
        runs = extra + ([run_id] if (cfg.data_dir / "runs" / run_id).exists() else [])
        done[period] = sorted(run_backtests(cfg, period, runs))
    previous = read_manifest(cfg, "backtest").get("periods", {})
    write_manifest(cfg.derived_dir / "backtest", stage="backtest", periods={**previous, **done})
    return {"strategies": sum(len(v) for v in done.values())}


def backtest_fresh(cfg: Config, manifest: dict, opts: RunOptions) -> str | None:
    missing = [p for p in opts.periods if p not in manifest.get("periods", {})]
    return f"no backtest for {', '.join(missing)}" if missing else None


def read_manifest(cfg: Config, folder: str) -> dict:
    import json

    path = cfg.derived_dir / folder / "_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------- registry (in run order)

STAGES: list[Stage] = [
    Stage("ingest", "6.9", (), "ingest", (), run_ingest, ingest_fresh, network=True),
    Stage("prices_adj", "2.4", ("lake:prices_daily", "lake:corporate_actions", "lake:index_daily",
                                "manual:ca_adjustments.csv"),
          "prices_adj", ("adjustment",), run_prices_adj),
    Stage("features", "10.1-10.3", ("stage:prices_adj", "lake:index_daily", "lake:macro", "lake:announcements",
                                     "lake:financial_results", "manual:calendar_overrides.csv"),
          "features_daily", ("features", "forecast_time_ist"), run_features),
    Stage("labels", "16.1", ("stage:prices_adj", "manual:calendar_overrides.csv"),
          "labels", ("labels", "features.skip_sessions"), run_labels),
    Stage("forecasts", "16.3-16.7b", ("stage:features", "stage:labels", "manual:universe.csv"),
          "forecasts", ("models", "periods"), run_forecasts, forecasts_fresh),
    Stage("evaluate", "18.1", ("stage:forecasts", "stage:labels", "lake:index_daily"),
          "metrics", ("regime", "periods"), run_evaluate, evaluate_fresh),
    Stage("agent_run", "12-13", ("stage:features", "stage:forecasts", "lake:announcements", "lake:board_meetings",
                                 "manual:calendar_overrides.csv"),
          "agent_runs", ("agents", "llm", "setups", "periods"), run_agents, agents_fresh),
    Stage("backtest", "16.8, 17", ("stage:prices_adj", "stage:forecasts", "stage:agent_run"),
          "backtest", ("backtest", "costs", "setups", "periods"), run_backtest, backtest_fresh),
]

TARGETS = {
    "all": [s.name for s in STAGES],
    "data": ["ingest", "prices_adj", "features", "labels"],
    "models": ["forecasts", "evaluate"],
    "agents": ["agent_run", "backtest"],
}
