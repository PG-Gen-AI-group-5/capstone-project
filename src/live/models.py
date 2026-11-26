"""Live forecasts: the TFT and LSTM refitted on all data up to the day before, predicting the next trading day.

A fit for day D trains on every row whose labels end before D (the walk-forward gap rule, src/labels/splits.py)
and is saved under data/live/models/<model>/fit_<D>/. A refit is due when there is no fit, the model settings
changed, or the last fit is `live.refit_every_days` trading days old (weekly by default; the scheduled refit
runs on Saturdays so the 08:00 run only predicts). Predictions go to data/live/derived/forecasts/<model>/live.parquet,
one set per day, so the agents read them as their base forecast and the chart draws the LSTM's volatility range.
None of this is written to data/derived/forecasts, where the walk-forward and frozen test forecasts live.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.config import Config
from src.labels.splits import train_mask
from src.live.env import LiveConfig, live_config, live_settings, trading_days
from src.models.base import check_forecasts
from src.models.data import SEQUENCE_INPUTS_VERSION, Panel, load_panel
from src.models.registry import create
from src.store.db import config_hash, write_table

LOADABLE = ("lstm", "tft")              # models that can be saved and reloaded for daily predictions


def settings_key(cfg: Config, name: str) -> str:
    m = cfg["models"]
    return config_hash(name, m.get(name), m.get("sequence"), SEQUENCE_INPUTS_VERSION, cfg["labels"], cfg["features"])[:10]


def fit_info(lcfg: LiveConfig, name: str) -> dict | None:
    p = lcfg.models_dir / name / "_live.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def refit_reason(lcfg: LiveConfig, name: str, day: date) -> str | None:
    """Why `name` should be refitted before predicting `day`, or None when the last fit is fine."""
    info = fit_info(lcfg, name)
    if info is None:
        return "no live fit yet"
    if info["settings"] != settings_key(lcfg, name):
        return "model settings changed"
    every = int(live_settings(lcfg)["refit_every_days"])
    fitted = date.fromisoformat(info["fitted_for"])
    age = len([d for d in trading_days(lcfg, fitted, day) if fitted <= d < day])
    return f"last fit is {age} trading days old" if age >= every else None


def refit(cfg: Config, name: str, day: date, panel: Panel | None = None, log=print) -> dict:
    if name not in LOADABLE:
        raise ValueError(f"Live forecasts support {', '.join(LOADABLE)}, not {name}")
    lcfg = cfg if isinstance(cfg, LiveConfig) else live_config(cfg)
    panel = panel or load_panel(lcfg)
    tmask = train_mask(panel.df, day, panel.max_h)
    train_end = pd.to_datetime(panel.df.loc[tmask, f"end_{panel.max_h}"]).max()
    log(f"  refitting {name} for {day}: {int(tmask.sum()):,} training rows, labels to {train_end.date()}")
    t0 = time.time()
    model = create(name, lcfg)
    model.fit(panel, tmask)
    folder = lcfg.models_dir / name / f"fit_{pd.Timestamp(day):%Y%m%d}"
    if folder.exists():
        shutil.rmtree(folder)
    model.save(folder)
    info = {"model": name, "fitted_for": str(day), "train_end": str(train_end.date()), "train_rows": int(tmask.sum()),
            "settings": settings_key(lcfg, name), "folder": folder.name, "sec": round(time.time() - t0, 1),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (lcfg.models_dir / name / "_live.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    for old in sorted((lcfg.models_dir / name).glob("fit_*"))[:-3]:          # keep the last three fits
        shutil.rmtree(old, ignore_errors=True)
    log(f"  {name} refitted in {info['sec'] / 60:.1f} min")
    return info


def predict(cfg: Config, name: str, day: date, panel: Panel | None = None) -> pd.DataFrame:
    """Forecast rows for `day` from the latest live fit, appended to the live forecast table."""
    lcfg = cfg if isinstance(cfg, LiveConfig) else live_config(cfg)
    info = fit_info(lcfg, name)
    if info is None:
        raise FileNotFoundError(f"No live fit for {name}; run: python -m src.live refit")
    panel = panel or load_panel(lcfg)
    model = type(create(name, lcfg)).load(lcfg, lcfg.models_dir / name / info["folder"], panel)
    d = pd.Timestamp(day)
    mask = (panel.df["trade_date"].eq(d) & panel.df["ready"].astype(bool) & panel.df["evaluate"].astype(bool)).to_numpy()
    if not mask.any():
        raise ValueError(f"No feature rows for {d.date()}; build the live tables first")
    fc = model.predict(panel, mask)
    fc["model_version"] = f"{name}-live-{info['fitted_for']}"
    fc["train_end"] = pd.Timestamp(info["train_end"])
    fc = fc[["model", "model_version", "symbol", "trade_date", "as_of", "horizon_days", "p_up", "direction",
             "exp_return", "vol_forecast", "train_end"]]
    check_forecasts(fc)
    path = lcfg.derived_dir / "forecasts" / name / "live.parquet"
    old = pd.read_parquet(path) if path.exists() else fc.iloc[0:0]
    old = old[pd.to_datetime(old["trade_date"]) != d] if len(old) else old
    write_table(lcfg, f"forecasts/{name}", pd.concat([old, fc], ignore_index=True), name="live")
    return fc


def forecast_all(cfg: Config, day: date, refit_if: str = "missing", log=print) -> dict:
    """Predict `day` with every live model. refit_if: 'missing' (refit only without a usable fit), 'due'
    (also when the weekly refit is due) or 'always'."""
    lcfg = live_config(cfg)
    panel = load_panel(lcfg)
    out = {}
    for name in live_settings(cfg)["models"]:
        reason = refit_reason(lcfg, name, day)
        must = reason in ("no live fit yet", "model settings changed")
        if refit_if == "always" or (reason and (must or refit_if == "due")):
            refit(lcfg, name, day, panel, log)
        elif reason:
            log(f"  {name}: {reason}; the weekly refit will update it (using the current fit)")
        fc = predict(lcfg, name, day, panel)
        out[name] = {"rows": len(fc), "fit": fit_info(lcfg, name)["fitted_for"],
                     "p_up_5d_mean": float(np.nanmean(fc.loc[fc["horizon_days"] == 5, "p_up"]))}
        log(f"  {name}: {len(fc)} forecast rows for {day} (fit for {out[name]['fit']})")
    return out
