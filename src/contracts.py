"""Shared data formats passed between modules (task 1.6). Pydantic models, so every agent output is
validated whether it came from an LLM or from the offline rules.

Prices in a plan are raw, as traded (CLAUDE.md "Prices"); timestamps are timezone-aware UTC.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- inputs the agents read

class Evidence(Contract):
    """One dated record an agent may cite. `published_at` must be before the run's as_of."""
    evidence_id: str                         # e.g. "ann:12345", "macro:usvix:2024-03-01"
    kind: Literal["announcement", "results", "news", "macro", "price", "catalyst", "forecast"]
    symbol: str | None = None
    published_at: datetime
    title: str
    text: str = ""
    url: str | None = None
    important: bool = False


class Setup(Contract):
    """A swing setup found by code (task 10.3) with code-computed levels (task 10.4), in raw prices."""
    symbol: str
    trade_date: date                          # the forecast day d; levels use closes up to d-1
    setup_type: Literal["pullback_uptrend", "breakout", "oversold_bounce", "none"]
    reference_close: float                    # raw close of d-1
    entry: float
    stop: float
    target: float
    atr_pct: float
    risk_reward: float


class Catalyst(Contract):
    """A dated event known in advance: results, F&O expiry, budget, ex-date (task 7)."""
    symbol: str | None                        # None = market-wide
    event_date: date
    kind: Literal["results", "expiry", "budget", "board_meeting", "ex_date", "holiday", "other"]
    description: str
    known_from: datetime | None = None


class BaseForecast(Contract):
    """A trained model's forecast for one stock and day (TFT by default; task 12.6)."""
    model: str
    model_version: str
    p_up: dict[int, float]                    # horizon (days) -> probability of a rise
    exp_return: dict[int, float]
    vol_forecast: dict[int, float]


# ---------------------------------------------------------------- agent outputs

class StockTask(Contract):
    symbol: str
    reasons: list[str]
    setup_type: str = "none"


class PlannerOutput(Contract):
    selected: list[StockTask]
    skipped: int
    notes: str = ""


class Claim(Contract):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceSummary(Contract):
    symbol: str
    summary: str
    claims: list[Claim]


class AnalystCase(Contract):
    side: Literal["bull", "bear"]
    symbol: str
    thesis: str
    claims: list[Claim]
    confidence: float = Field(ge=0, le=1)
    rebuttal: str = ""


class TradePlan(Contract):
    """The Forecaster's plan. Entry, stop and target come from code (task 10.4); the agent picks the
    direction, holding period and confidence inside limits set by code."""
    symbol: str
    as_of: datetime
    direction: Literal["long", "no_trade"]
    entry: float
    stop: float
    target: float
    max_hold_days: int = Field(ge=1, le=10)
    vol_forecast: float | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    setup_type: str = "none"
    base_forecast_p_up: float | None = None


class CheckResult(Contract):
    code: str                                 # e.g. "LOOKAHEAD", "NO_CITATION", "LOW_RR", "CATALYST_IN_HOLD"
    passed: bool
    detail: str = ""


class CriticVerdict(Contract):
    verdict: Literal["accept", "revise", "reject"]
    checks: list[CheckResult]
    reasons: list[str]


class ExitDecision(Contract):
    symbol: str
    action: Literal["hold", "tighten_stop", "exit"]
    new_stop: float | None = None
    reason: str


# ---------------------------------------------------------------- records written by the system

class TraceStep(Contract):
    run_id: str
    step: int
    at: datetime
    agent: str
    symbol: str | None = None
    kind: Literal["llm_call", "offline", "tool", "debate", "critique", "revision", "decision", "error"]
    prompt_version: str | None = None
    provider: str | None = None
    input_summary: str = ""
    output: dict = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False


class Signal(Contract):
    """What the backtester accepts (docs/ARCHITECTURE.md "Signals")."""
    symbol: str
    signal_date: date                         # the forecast day; entry at its open
    side: Literal["long"] = "long"
    stop: float
    target: float
    max_hold_days: int
    source: str                               # model name or agent run ID
    confidence: float | None = None
    ref: str | None = None                    # caller's link back to what made the signal (the desk's plan ID)


class TradeRecord(Contract):
    symbol: str
    source: str
    signal_date: date
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    exit_reason: Literal["stop", "target", "time", "gap_stop", "exit", "no_fill", "open"]   # exit: sold at the open on request
    quantity: int
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
