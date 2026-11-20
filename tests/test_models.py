import numpy as np
import pandas as pd
import pytest

from src.evaluation.bootstrap import metrics_with_intervals
from src.evaluation.metrics import direction_metrics, evaluate, join_labels
from src.models.base import check_forecasts
from src.models.data import label_frame
from src.models.walkforward import run
from tests import synthetic


@pytest.fixture(scope="module")
def cfg():
    return synthetic.config()


@pytest.fixture(scope="module")
def panel(cfg):
    return synthetic.panel(cfg)


@pytest.mark.parametrize("name", ["base_rate", "ma_crossover", "hist_vol", "arima", "lstm", "tft"])
def test_every_model_writes_valid_forecasts(name, cfg, panel):
    fc = run(name, cfg, "validation", panel=panel, save=False, log=lambda *_: None)
    check_forecasts(fc)
    assert set(fc["horizon_days"]) == {1, 5, 10}
    assert fc["trade_date"].between(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30")).all()
    assert (pd.to_datetime(fc["train_end"]) < fc["trade_date"]).all()


def test_test_period_needs_the_freeze_flag(cfg, panel):
    with pytest.raises(PermissionError):
        run("base_rate", cfg, "test", panel=panel, save=False)


def test_check_forecasts_rejects_training_on_future_labels(cfg, panel):
    fc = run("base_rate", cfg, "validation", panel=panel, save=False, log=lambda *_: None)
    fc.loc[0, "train_end"] = fc.loc[0, "trade_date"]
    with pytest.raises(ValueError, match="trained on labels"):
        check_forecasts(fc)


def test_direction_metrics():
    up = np.array([1, 0, 1, 1.0])
    m = direction_metrics(up, np.array([1, -1, -1, 1.0]), np.array([0.9, 0.2, 0.4, 0.6]))
    assert m["accuracy"] == 0.75
    assert m["f1_up"] == pytest.approx(0.8)
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.36 + 0.16) / 4)


def test_evaluate_and_bootstrap(cfg, panel):
    fc = run("ma_crossover", cfg, "validation", panel=panel, save=False, log=lambda *_: None)
    labels = label_frame(panel)
    table = evaluate(fc, labels)
    assert len(table) == 3 and table["accuracy"].between(0, 1).all()
    ci = metrics_with_intervals(join_labels(fc, labels), metrics=("accuracy",), n=50, block=5)
    assert (ci["accuracy_low"] <= ci["accuracy"]).all() and (ci["accuracy"] <= ci["accuracy_high"]).all()
