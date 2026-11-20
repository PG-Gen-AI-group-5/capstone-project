"""Walk-forward runner: refit on a schedule, predict each fold, write forecasts and a manifest.

Output: data/derived/forecasts/<model>/<period>.parquet and data/models/<model>/<period>/
(one folder per fold plus _manifest.json). The test period runs only with allow_test=True, which
is reserved for the freeze task 16.9.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.config import Config
from src.labels.splits import make_folds, train_mask
from src.models.base import check_forecasts
from src.models.data import SEQUENCE_INPUTS_VERSION, Panel, load_panel
from src.models.registry import create
from src.store.db import config_hash, write_manifest, write_table


def model_version(cfg: Config, name: str, params: dict | None) -> str:
    m = cfg["models"]
    sequence = (m.get("sequence"), SEQUENCE_INPUTS_VERSION) if name in ("lstm", "tft") else None
    parts = (name, m.get(name), sequence, m.get("refit", {}).get(name), cfg["labels"], cfg["features"], params)
    return f"{name}-{config_hash(*parts)[:8]}"


def run(name: str, cfg: Config, period: str = "validation", *, panel: Panel | None = None,
        params: dict | None = None, allow_test: bool = False, save: bool = True, log=print) -> pd.DataFrame:
    if period == "test" and not allow_test:
        raise PermissionError("Test-period forecasts are made once, by the freeze task 16.9. "
                              "Use allow_test=True (CLI: --freeze) only for that run.")
    panel = panel or load_panel(cfg)
    start, end = cfg.period(period)
    last = panel.dates.max()
    end = min(pd.Timestamp(end), last) if end else last
    freq = (params or {}).get("refit") or cfg["models"]["refit"].get(name, "yearly")
    folds = make_folds(panel.dates, start, end, freq)
    version = model_version(cfg, name, params)
    model = create(name, cfg, params)
    out_dir = cfg.models_dir / name / period
    log(f"{name} ({version}): {period} {pd.Timestamp(start).date()}..{end.date()}, {len(folds)} fold(s), refit {freq}")

    forecasts, fold_info, t_start = [], [], time.time()
    for fold in folds:
        t0 = time.time()
        tmask = train_mask(panel.df, fold.start, panel.max_h)
        pmask = panel.eval_mask(fold.start, fold.end)
        if not tmask.any() or not pmask.any():
            continue
        train_end = pd.to_datetime(panel.df.loc[tmask, f"end_{panel.max_h}"]).max()
        model.fit(panel, tmask)
        fc = model.predict(panel, pmask)
        if fc.empty:
            continue
        fc["train_end"] = train_end
        forecasts.append(fc)
        if save:
            model.save(out_dir / f"fold_{fold.start:%Y%m%d}")
        info = {"start": fold.start.date(), "end": fold.end.date(), "train_end": train_end.date(),
                "train_rows": int(tmask.sum()), "pred_rows": int(pmask.sum()), "sec": round(time.time() - t0, 1)}
        fold_info.append(info)
        log(f"  fold {info['start']}..{info['end']}: trained on {info['train_rows']} rows to {info['train_end']}, "
            f"{info['sec']}s")

    fc = pd.concat(forecasts, ignore_index=True)
    fc["model_version"] = version
    fc = fc[["model", "model_version", "symbol", "trade_date", "as_of", "horizon_days", "p_up", "direction",
             "exp_return", "vol_forecast", "train_end"]]
    check_forecasts(fc)
    if save:
        write_table(cfg, f"forecasts/{name}", fc, name=period)
        write_manifest(out_dir, model=name, model_version=version, period=period, refit=freq,
                       seed=cfg["models"]["seed"], params=params or {}, settings=model.describe(),
                       folds=fold_info, rows=len(fc), runtime_sec=round(time.time() - t_start, 1))
    return fc


def saved_version(cfg: Config, name: str, period: str) -> str | None:
    path = cfg.models_dir / name / period / "_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")).get("model_version") if path.exists() else None


def run_or_load(name: str, cfg: Config, period: str = "validation", *, panel: Panel | None = None,
                params: dict | None = None, retrain: bool = False, log=print) -> pd.DataFrame:
    """Saved forecasts when they were made with the current settings; otherwise train (for notebooks)."""
    path = cfg.derived_dir / "forecasts" / name / f"{period}.parquet"
    if not retrain and path.exists() and saved_version(cfg, name, period) == model_version(cfg, name, params):
        log(f"{name}: loaded saved forecasts ({model_version(cfg, name, params)})")
        return pd.read_parquet(path)
    return run(name, cfg, period, panel=panel, params=params, log=log)


def load_forecasts(cfg: Config, period: str = "validation", models: list[str] | None = None) -> pd.DataFrame:
    base = cfg.derived_dir / "forecasts"
    paths = sorted(base.glob(f"*/{period}.parquet"))
    frames = [pd.read_parquet(p) for p in paths if models is None or p.parent.name in models]
    if not frames:
        raise FileNotFoundError(f"No forecasts for {period} under {base}")
    return pd.concat(frames, ignore_index=True)

