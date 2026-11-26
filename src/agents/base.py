"""What every agent receives: the run's settings, the LLM client, the trace writer and the day's view."""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.context import DayView
from src.config import Config
from src.llm.client import LLMClient
from src.tracing.writer import TraceWriter

AGENT_DEFAULTS = {
    "base_model": "tft", "max_stocks_per_day": 5, "min_planner_score": 2, "evidence_lookback_days": 30,
    "min_edge": 0.05, "min_risk_reward": 1.5, "max_confidence": 0.75, "critic_max_rounds": 2,
    "hold_days": {"pullback_uptrend": 7, "breakout": 10, "oversold_bounce": 5, "none": 5},
    "high_vix": 0.2,
}


@dataclass
class AgentContext:
    cfg: Config
    run_id: str
    llm: LLMClient
    tracer: TraceWriter
    day: DayView

    @property
    def settings(self) -> dict:
        a = self.cfg.get("agents") or {}
        out = {**AGENT_DEFAULTS, **a}
        out["hold_days"] = {**AGENT_DEFAULTS["hold_days"], **a.get("hold_days", {})}
        return out
