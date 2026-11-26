import copy
import json
import os
import time

import pandas as pd
import pytest

from src.config import Config
from src.models.walkforward import data_fingerprint
from src.pipeline.runner import plan, run, staleness, status
from src.pipeline.stages import STAGES, RunOptions, Stage, run_forecasts
from src.store.db import write_manifest


@pytest.fixture()
def cfg(tmp_path):
    raw = copy.deepcopy(Config.load().raw)
    raw["paths"]["data"] = "data"
    raw["toy"] = {"factor": 2}
    c = Config(raw, tmp_path)
    (c.lake_dir / "prices").mkdir(parents=True)
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(c.lake_dir / "prices" / "2024.parquet")
    return c


CALLS: list[str] = []


def toy_stage(name, reads, out):
    def run_fn(cfg, opts):
        CALLS.append(name)
        write_manifest(cfg.derived_dir / out, stage=name, rows=3)
        return {"rows": 3}
    return Stage(name, "-", reads, out, ("toy",), run_fn)


TOY = [toy_stage("clean", ("lake:prices",), "clean"), toy_stage("feats", ("stage:clean",), "feats")]
TOY_TARGETS = {"all": ["clean", "feats"]}


def go(cfg, targets, **kw):
    CALLS.clear()
    return run(cfg, targets, RunOptions(**kw), stages=TOY, targets_map=TOY_TARGETS, log=lambda *_: None)


def test_group_run_builds_in_order_then_skips_up_to_date_stages(cfg):
    go(cfg, ["all"])
    assert CALLS == ["clean", "feats"]
    assert all(staleness(cfg, s, RunOptions(), TOY) is None for s in TOY)
    go(cfg, ["all"])
    assert CALLS == []
    go(cfg, ["all"], force=True)
    assert CALLS == ["clean", "feats"]


def test_a_named_stage_always_runs_and_makes_later_stages_stale(cfg):
    go(cfg, ["all"])
    time.sleep(0.01)
    go(cfg, ["clean"])
    assert CALLS == ["clean"]
    assert "stage:clean changed" in staleness(cfg, TOY[1], RunOptions(), TOY)
    go(cfg, ["all"])
    assert CALLS == ["feats"]


def test_new_input_data_or_changed_settings_make_a_stage_stale(cfg):
    go(cfg, ["all"])
    later = time.time() + 60
    os.utime(cfg.lake_dir / "prices" / "2024.parquet", (later, later))
    assert "lake:prices changed" in staleness(cfg, TOY[0], RunOptions(), TOY)
    go(cfg, ["all"])
    assert CALLS == ["clean", "feats"]
    cfg.raw["toy"]["factor"] = 3
    assert "settings changed" in staleness(cfg, TOY[0], RunOptions(), TOY)


def test_status_lists_every_stage(cfg):
    table = status(cfg, stages=TOY)
    assert table["state"].tolist() == ["never built", "never built"]
    go(cfg, ["clean"])
    assert status(cfg, stages=TOY)["state"].tolist() == ["up to date", "never built"]


def test_unknown_target_is_an_error():
    with pytest.raises(ValueError, match="unknown stage"):
        plan(["everything"], TOY, TOY_TARGETS)


def test_real_registry_runs_data_before_models():
    order = [s.name for s in plan(["all"])[0]]
    assert order == ["ingest", "prices_adj", "features", "labels", "forecasts", "evaluate", "agent_run", "backtest"]
    assert {r.split(":", 1)[1] for s in STAGES for r in s.reads if r.startswith("stage:")} <= set(order)


def test_test_period_forecasts_need_the_freeze_flag(cfg):
    with pytest.raises(PermissionError):
        run_forecasts(cfg, RunOptions(periods=["test"]))


def test_the_pipeline_record_does_not_change_model_versions(cfg):
    write_manifest(cfg.derived_dir / "features_daily", stage="features", rows=10)
    before = data_fingerprint(cfg)
    path = cfg.derived_dir / "features_daily" / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["pipeline"] = {"config_hash": "abc", "runtime_sec": 1.0}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert data_fingerprint(cfg) == before
