"""Backtest every strategy on the same engine and write derived/backtest/<strategy>_<period>/.

Strategies: each model in `backtest.models` (its forecasts -> signals, task 16.8), equal-weight buy-and-hold
of the Nifty 50 stocks and the Nifty 50 index (16.3), and agent runs (their accepted plans and Exit Monitor
decisions). Each folder holds journal.parquet, equity.parquet and summary.json.
"""
from __future__ import annotations

import json

import pandas as pd

from src.backtest.engine import PORTFOLIO_DEFAULTS, buy_and_hold, simulate, summarise
from src.backtest.signals import forecast_signals
from src.config import Config
from src.store.db import read_table, universe, write_manifest

DEFAULTS = {"models": ["ma_crossover", "arima", "lstm", "tft"], "horizon": 5, "default_threshold": 0.55,
            "thresholds": {}}


def raw_bars(cfg: Config) -> pd.DataFrame:
    """Raw (as-traded) OHLC from the adjusted table: adjusted price / cumulative factor."""
    p = read_table(cfg, "prices_adj", layer="derived",
                   columns=["symbol", "trade_date", "high", "low", "raw_open", "raw_close", "adj_factor"])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    return pd.DataFrame({"symbol": p["symbol"], "trade_date": p["trade_date"], "open": p["raw_open"],
                         "high": p["high"] / p["adj_factor"], "low": p["low"] / p["adj_factor"], "close": p["raw_close"]})


def signal_features(cfg: Config, bars: pd.DataFrame) -> pd.DataFrame:
    f = read_table(cfg, "features_daily", layer="derived", columns=["symbol", "trade_date", "atr14_pct", "ready"])
    f["trade_date"] = pd.to_datetime(f["trade_date"])
    b = bars.sort_values(["symbol", "trade_date"])
    b = b.assign(ref_close=b.groupby("symbol")["close"].shift(1))
    return f[f["ready"]].merge(b[["symbol", "trade_date", "ref_close"]], on=["symbol", "trade_date"], how="left")


def write_result(cfg: Config, name: str, journal: pd.DataFrame | None, equity: pd.DataFrame, summary: dict) -> None:
    folder = cfg.derived_dir / "backtest" / name
    folder.mkdir(parents=True, exist_ok=True)
    equity.to_parquet(folder / "equity.parquet", index=False)
    if journal is not None:
        journal.to_parquet(folder / "journal.parquet", index=False)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def run_backtests(cfg: Config, period: str = "validation", agent_runs: list[str] | None = None, log=print) -> dict:
    s = {**DEFAULTS, **(cfg.get("backtest") or {})}
    port = {**PORTFOLIO_DEFAULTS, **(s.get("portfolio") or {})}
    costs = cfg.get("costs")
    start, end = cfg.period(period)
    bars = raw_bars(cfg)
    end = pd.Timestamp(end) if end else bars["trade_date"].max()
    feats = signal_features(cfg, bars)
    window = bars[(bars["trade_date"] >= pd.Timestamp(start)) & (bars["trade_date"] <= end)]
    results = {}

    for model in s["models"]:
        path = cfg.derived_dir / "forecasts" / model / f"{period}.parquet"
        if not path.exists():
            log(f"  {model}: no {period} forecasts, skipped")
            continue
        threshold = float(s["thresholds"].get(model, s["default_threshold"]))
        sig = forecast_signals(pd.read_parquet(path), feats, threshold, int(s["horizon"]), cfg.get("setups"))
        journal, equity = simulate(sig, window, start, end, port, costs)
        results[f"{model}_{period}"] = (journal, equity, {"signals": len(sig), "threshold": threshold})

    symbols = list(universe(cfg)["symbol"])
    bh = buy_and_hold(window, symbols, start, end, port["capital"], costs)
    results[f"buy_and_hold_{period}"] = (None, bh, {})
    idx = read_table(cfg, "index_daily", where="index_name = 'Nifty 50'", columns=["trade_date", "close"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    idx = idx[(idx["trade_date"] >= pd.Timestamp(start)) & (idx["trade_date"] <= end)].sort_values("trade_date")
    nifty = pd.DataFrame({"trade_date": idx["trade_date"], "cash": 0.0, "open_positions": 1,
                          "equity": (port["capital"] * idx["close"] / idx["close"].iloc[0]).round(2)})
    nifty["positions"] = nifty["equity"]
    results[f"nifty50_index_{period}"] = (None, nifty, {"note": "index level, no costs"})

    from src.run import read_jsonl, run_dir, signals_from_run
    for run_id in agent_runs or []:
        folder = run_dir(cfg, run_id)
        if not folder.exists():
            log(f"  {run_id}: no such run, skipped")
            continue
        exits = pd.DataFrame(read_jsonl(folder / "exits.jsonl"))
        sig = signals_from_run(folder)
        first = min((x.signal_date for x in sig), default=start)
        last = max((x.signal_date for x in sig), default=end)
        journal, equity = simulate(sig, bars, first, last, port, costs, exits if len(exits) else None)
        results[run_id] = (journal, equity, {"signals": len(sig)})

    out = {}
    for name, (journal, equity, extra) in results.items():
        summary = {**summarise(journal if journal is not None else pd.DataFrame(columns=["exit_reason"]), equity,
                               port["capital"]), **extra}
        write_result(cfg, name, journal, equity, summary)
        out[name] = summary
        log(f"  {name}: {summary.get('trades', '-')} trades, total return {summary.get('total_return', float('nan')):+.1%}")
    write_manifest(cfg.derived_dir / "backtest", stage="backtest", period=period, strategies=sorted(out))
    return out
