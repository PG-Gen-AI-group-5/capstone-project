"""The agent graph (tasks 13.1-13.3), built with LangGraph.

Day graph:   START -> exit_monitor -> planner -> (one `analyse` branch per selected stock) -> END
Stock graph: START -> retrieve -> debate -> forecast -> critique -> END
             critique routes: accept -> END; reject -> END (no trade); revise -> forecast again with the
             Critic's feedback, at most `critic_max_rounds` times, then END with the last verdict.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents import analysts, critic, exit_monitor, forecaster, planner, retriever
from src.agents.base import AgentContext


class StockState(TypedDict, total=False):
    symbol: str
    reasons: list[str]
    setup: Any
    evidence: list
    summary: Any
    catalysts: list
    bull: Any
    bear: Any
    plan: Any
    verdict: Any
    rounds: int
    revisions: list


class DayState(TypedDict, total=False):
    positions: list[dict]
    exits: list
    planner: Any
    results: Annotated[list, operator.add]


def stock_graph(ctx: AgentContext):
    def retrieve(state: StockState) -> dict:
        symbol = state["symbol"]
        evidence, summary = retriever.retrieve(ctx, symbol)
        hold = max(ctx.settings["hold_days"].values())
        return {"setup": ctx.day.setup(symbol), "evidence": evidence, "summary": summary,
                "catalysts": ctx.day.catalysts(symbol, hold + 1), "rounds": 0, "revisions": []}

    def debate(state: StockState) -> dict:
        args = (ctx, state["symbol"], state["setup"], state["summary"])
        bull, bear = analysts.opening_cases(*args, state["catalysts"], state["evidence"])
        bull, bear = analysts.rebuttals(*args, bull, bear, state["evidence"])
        return {"bull": bull, "bear": bear}

    def forecast(state: StockState) -> dict:
        feedback = state.get("verdict") if state.get("rounds", 0) > 0 else None
        plan = forecaster.forecast(ctx, state["symbol"], state["setup"], state["bull"], state["bear"],
                                   state["catalysts"], state["evidence"], feedback)
        if feedback is not None:
            ctx.tracer.step("forecaster", "revision", symbol=state["symbol"],
                            output={"round": state["rounds"], "direction": plan.direction,
                                    "max_hold_days": plan.max_hold_days, "confidence": plan.confidence})
        return {"plan": plan}

    def critique(state: StockState) -> dict:
        verdict = critic.review(ctx, state["plan"], state["setup"], state["evidence"], state["catalysts"],
                                state["bull"], state["bear"])
        return {"verdict": verdict, "rounds": state.get("rounds", 0) + 1,
                "revisions": state.get("revisions", []) + [{"plan": state["plan"], "verdict": verdict}]}

    def route(state: StockState) -> str:
        v = state["verdict"].verdict
        if v == "revise" and state["rounds"] <= ctx.settings["critic_max_rounds"]:
            return "forecast"
        return END

    g = StateGraph(StockState)
    for name, fn in (("retrieve", retrieve), ("debate", debate), ("forecast", forecast), ("critique", critique)):
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "debate")
    g.add_edge("debate", "forecast")
    g.add_edge("forecast", "critique")
    g.add_conditional_edges("critique", route, ["forecast", END])
    return g.compile()


def day_graph(ctx: AgentContext):
    per_stock = stock_graph(ctx)

    def monitor(state: DayState) -> dict:
        decisions = [exit_monitor.check(ctx, p) for p in state.get("positions", [])]
        return {"exits": decisions}

    def plan_day(state: DayState) -> dict:
        return {"planner": planner.plan(ctx)}

    def fan_out(state: DayState):
        tasks = state["planner"].selected
        return [Send("analyse", {"symbol": t.symbol, "reasons": t.reasons}) for t in tasks] or END

    def analyse(state: StockState) -> dict:
        if ctx.day.setup(state["symbol"]) is None:
            return {"results": []}
        final = per_stock.invoke({"symbol": state["symbol"], "reasons": state.get("reasons", [])})
        return {"results": [final]}

    g = StateGraph(DayState)
    g.add_node("exit_monitor", monitor)
    g.add_node("planner", plan_day)
    g.add_node("analyse", analyse)
    g.add_edge(START, "exit_monitor")
    g.add_edge("exit_monitor", "planner")
    g.add_conditional_edges("planner", fan_out, ["analyse", END])
    g.add_edge("analyse", END)
    return g.compile()
