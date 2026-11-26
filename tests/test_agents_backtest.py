import copy
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.agents.context import RunData
from src.agents.critic import rule_checks, rule_verdict
from src.backtest.costs import order_cost
from src.backtest.engine import simulate, summarise
from src.catalysts.calendar import CatalystCalendar
from src.config import Config
from src.contracts import Evidence, PlannerOutput, Setup, Signal, StockTask, TradePlan
from src.features.build import build_features
from src.features.setups import detect, levels
from src.llm import client as llm_client
from src.llm.client import LLMClient
from src.llm.promptlib import load_prompt
from src.run import read_jsonl, run_days, signals_from_run
from src.store.pit import AnnouncementIndex
from tests import synthetic

UTC = timezone.utc


# ---------------------------------------------------------------- costs and the trade engine

def test_costs_match_a_hand_calculation():
    # Rs 1 lakh delivery: STT 100, exchange 2.97, SEBI 0.10, GST 18% of 3.07, stamp 15 (buy) or DP 15.93 (sell)
    assert order_cost(100_000, "buy") == pytest.approx(100 + 2.97 + 0.10 + 0.5526 + 15, abs=0.01)
    assert order_cost(100_000, "sell") == pytest.approx(100 + 2.97 + 0.10 + 0.5526 + 15.93, abs=0.01)


def bars(rows, symbol="AAA"):
    days = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame([{"symbol": symbol, "trade_date": d, "open": o, "high": h, "low": lo, "close": c}
                         for d, (o, h, lo, c) in zip(days, rows)])


def signal(stop=95.0, target=110.0, hold=3, day=date(2024, 1, 1)):
    return Signal(symbol="AAA", signal_date=day, stop=stop, target=target, max_hold_days=hold, source="test",
                  confidence=0.6)


def one_trade(rows, **kw):
    journal, equity = simulate([signal(**kw)], bars(rows), cost_rates={"stt_pct": 0, "exchange_pct": 0,
                                                                       "sebi_per_crore": 0, "stamp_buy_pct": 0,
                                                                       "dp_per_sell": 0})
    return journal.iloc[0], equity


def test_stop_counts_first_when_stop_and_target_fall_on_the_same_day():
    t, _ = one_trade([(100, 112, 94, 100)])
    assert t.exit_reason == "stop" and t.exit_price == 95.0


def test_a_gap_through_the_stop_fills_at_the_open():
    t, _ = one_trade([(100, 101, 99, 100), (90, 92, 89, 91)])
    assert t.exit_reason == "gap_stop" and t.exit_price == 90.0


def test_no_fill_when_the_stock_is_locked_at_its_band():
    t, _ = one_trade([(100, 100, 100, 100)])
    assert t.exit_reason == "no_fill" and t.quantity == 0


def test_time_exit_at_the_close_of_the_last_holding_day():
    t, equity = one_trade([(100, 101, 99, 100), (100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)])
    assert t.exit_reason == "time" and t.exit_date == date(2024, 1, 3) and t.exit_price == 102.0
    assert summarise(pd.DataFrame([t]), equity, 1_000_000)["trades"] == 1


# ---------------------------------------------------------------- setups and the critic

def test_levels_come_from_the_reference_close_and_atr():
    entry, stop, target, rr = levels(100.0, 0.02)          # 1.5 ATR stop, 2:1 reward-to-risk
    assert (entry, stop, target, rr) == (100.0, 97.0, 106.0, 2.0)


def test_setup_rules():
    f = pd.DataFrame({"ema200_gap": [0.05, 0.05, -0.1], "ema20_50": [0.01, 0.01, -0.01], "ema20_gap": [0.0, 0.05, 0.0],
                      "rsi14": [0.5, 0.8, 0.2], "dist_52w_high": [-0.2, 0.0, -0.4], "logvol_z20": [0, 1.0, 0],
                      "ret_5": [0.0, 0.03, -0.05]})
    assert detect(f).tolist() == ["pullback_uptrend", "breakout", "none"]


def test_critic_rejects_a_plan_citing_a_document_published_after_as_of():
    as_of = datetime(2024, 6, 3, 3, 0, tzinfo=UTC)               # 08:30 IST
    setup = Setup(symbol="AAA", trade_date=date(2024, 6, 3), setup_type="pullback_uptrend", reference_close=100,
                  entry=100, stop=97, target=106, atr_pct=0.02, risk_reward=2.0)
    future = Evidence(evidence_id="ann:planted", kind="announcement", symbol="AAA", published_at=as_of + timedelta(hours=2),
                      title="results")
    plan = TradePlan(symbol="AAA", as_of=as_of, direction="long", entry=100, stop=97, target=106, max_hold_days=5,
                     confidence=0.6, rationale="x", evidence_ids=["ann:planted"])
    settings = {"min_risk_reward": 1.5, "max_confidence": 0.75}
    verdict = rule_verdict(rule_checks(plan, setup, [future], [], date(2024, 6, 10), settings))
    assert verdict.verdict == "reject" and any("LOOKAHEAD" in r for r in verdict.reasons)


def test_the_point_in_time_reader_hides_records_from_as_of_onwards():
    t = datetime(2024, 6, 3, 3, 0, tzinfo=UTC)
    ann = pd.DataFrame({"doc_id": ["a", "b"], "symbol": ["AAA", "AAA"], "category": ["x", "y"], "summary": ["", ""],
                        "published_at": [t - timedelta(seconds=1), t], "is_important": [True, True],
                        "attachment_url": [None, None]})
    seen = AnnouncementIndex(ann).as_of("AAA", t)
    assert seen["doc_id"].tolist() == ["a"]


# ---------------------------------------------------------------- the LLM client

def test_every_agent_has_a_prompt_with_both_sections():
    for agent in ("planner", "retriever", "bull", "bear", "forecaster", "critic", "exit_monitor"):
        p = load_prompt(agent)
        assert p.version in ("v1", "v2") and p.system and p.user


def test_llm_responses_are_cached_and_bad_output_falls_back_to_the_rules(tmp_path, monkeypatch):
    raw = copy.deepcopy(Config.load().raw)
    raw["paths"]["data"] = "data"
    raw["llm"] = {"provider": "fake"}
    cfg = Config(raw, tmp_path)
    calls = []

    def fake(settings, system, user, schema):
        calls.append(user)
        out = {"selected": [{"symbol": "AAA", "reasons": ["r"], "setup_type": "none"}], "skipped": 0, "notes": ""}
        return json.dumps(out), {"input_tokens": 100, "output_tokens": 50}

    monkeypatch.setitem(llm_client.PROVIDERS, "fake", fake)
    rules = lambda: PlannerOutput(selected=[StockTask(symbol="ZZZ", reasons=[])], skipped=1)
    c = LLMClient(cfg)
    first = c.complete("planner", {"trade_date": "2024-06-03", "max_stocks": 5, "candidates": []}, PlannerOutput, rules)
    second = c.complete("planner", {"trade_date": "2024-06-03", "max_stocks": 5, "candidates": []}, PlannerOutput, rules)
    assert first.output.selected[0].symbol == "AAA" and not first.cached
    assert second.cached and len(calls) == 1 and c.spent_today() > 0
    monkeypatch.setitem(llm_client.PROVIDERS, "fake", lambda *a: ("not json", {}))
    bad = c.complete("planner", {"trade_date": "2024-06-04", "max_stocks": 5, "candidates": []}, PlannerOutput, rules)
    assert bad.provider == "offline" and bad.output.selected[0].symbol == "ZZZ"


# ---------------------------------------------------------------- the whole graph, offline

@pytest.fixture()
def world(tmp_path):
    raw = copy.deepcopy(Config.load().raw)
    raw["paths"]["data"] = "data"
    raw["llm"] = {"provider": "offline"}
    raw["agents"] = {"min_planner_score": 0, "max_stocks_per_day": 3}
    cfg = Config(raw, tmp_path)
    prices, index, macro = synthetic.market(320)
    feats, cal = build_features(prices, index, macro, 60)
    p = prices.sort_values(["symbol", "trade_date"])
    feats = feats.merge(p.assign(ref_close=p.groupby("symbol")["close"].shift(1))[["symbol", "trade_date", "ref_close"]],
                        on=["symbol", "trade_date"])
    feats = feats[feats["ready"]].assign(setup_type=lambda d: detect(d)).reset_index(drop=True)
    day = feats["trade_date"].unique()[-20]
    ann = pd.DataFrame({"doc_id": ["1"], "symbol": ["AAA"], "category": ["Press Release"], "summary": ["new order"],
                        "published_at": [pd.Timestamp(day).tz_localize("UTC") - pd.Timedelta(days=2)],
                        "is_important": [True], "attachment_url": [None]})
    releases = pd.DataFrame({"symbol": pd.Series(dtype=str), "day0": pd.Series(dtype="datetime64[ns]")})
    fc = pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "p_up", "exp_return", "vol_forecast", "model_version"])
    data = RunData(cfg, feats, cal, AnnouncementIndex(ann), CatalystCalendar(cal, releases), fc, "tft")
    return cfg, data, prices, pd.Timestamp(day).date()


def test_the_graph_runs_a_week_end_to_end_and_resumes(world):
    cfg, data, prices, day = world
    end = (pd.Timestamp(day) + pd.offsets.BDay(4)).date()
    folder = run_days(cfg, day, end, run_id="t", data=data, log=lambda *_: None)
    plans = read_jsonl(folder / "plans.jsonl")
    assert len(plans) == 5 * 3                                  # 3 stocks a day for 5 days
    assert {p["verdict"] for p in plans} <= {"accept", "revise", "reject"}
    trace = read_jsonl(folder / "trace.jsonl")
    assert {"planner", "retriever", "bull", "bear", "forecaster", "critic"} <= {t["agent"] for t in trace}
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["llm"]["provider"] == "offline" and manifest["prompt_versions"]["critic"] == "v2" and manifest["prompt_versions"]["planner"] == "v1"
    run_days(cfg, day, end, run_id="t", data=data, log=lambda *_: None)
    assert len(read_jsonl(folder / "plans.jsonl")) == len(plans)          # finished days are skipped

    sig = signals_from_run(folder)
    if sig:
        journal, equity = simulate(sig, prices)
        assert len(journal) >= len(sig) and equity["equity"].gt(0).all()


def test_indicator_and_level_records_are_citable_and_dated_before_as_of(world):
    _, data, _, day = world
    view = data.day(day)
    evidence = view.evidence("AAA")
    ids = {e.evidence_id for e in evidence}
    assert f"ctx:AAA:{day}" in ids and f"setup:AAA:{day}" in ids
    assert all(e.published_at < view.as_of for e in evidence)
