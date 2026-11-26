"""Exit Monitor (task 12.9): re-checks open positions each morning, before the new plans are made.

Offline rules, in order: exit when the last close passed the stop or target; exit when the holding period
is used up; exit before a results release due by the
next trading day; move the stop up to the entry price once the last close is one ATR above entry. The
backtester applies an `exit` at that day's open and a tightened stop from that day on.
"""
from __future__ import annotations

import pandas as pd

from src.agents.base import AgentContext
from src.contracts import ExitDecision


def check(ctx: AgentContext, position: dict) -> ExitDecision:
    day = ctx.day
    symbol = position["symbol"]
    held = len(day.data.days[(day.data.days >= pd.Timestamp(position["entry_date"])) & (day.data.days < day.trade_date)])
    catalysts = day.catalysts(symbol, 2)
    row = day.rows.loc[symbol] if symbol in day.rows.index else None
    ref = float(row["ref_close"]) if row is not None and pd.notna(row["ref_close"]) else None
    atr = float(row["atr14_pct"]) if row is not None and pd.notna(row["atr14_pct"]) else 0.0

    def rules() -> ExitDecision:
        if ref is not None and (ref <= position["stop"] or ref >= position["target"]):
            return ExitDecision(symbol=symbol, action="exit", reason=f"last close {ref} passed the stop or target")
        if held >= position["max_hold_days"]:
            return ExitDecision(symbol=symbol, action="exit", reason=f"held {held} days (planned {position['max_hold_days']})")
        results = [k for k in catalysts if k.kind == "results" and k.known_from is not None]
        if results:
            return ExitDecision(symbol=symbol, action="exit", reason=f"results due: {results[0].description}")
        if ref is not None and ref >= position["entry"] * (1 + atr) and position["stop"] < position["entry"]:
            return ExitDecision(symbol=symbol, action="tighten_stop", new_stop=round(position["entry"], 2),
                                reason="one ATR in profit: stop moved to the entry price")
        return ExitDecision(symbol=symbol, action="hold", reason="no exit condition")

    variables = {"position": position, "trade_date": str(day.trade_date.date()),
                 "evidence": [e.model_dump(mode="json") for e in day.evidence(symbol, 10)
                              if e.published_at >= pd.Timestamp(position["entry_date"]).tz_localize("UTC")],
                 "catalysts": [k.model_dump(mode="json") for k in catalysts]}
    decision = ctx.llm.complete("exit_monitor", variables, ExitDecision, rules, symbol=symbol).output
    if decision.action == "tighten_stop" and (decision.new_stop is None or decision.new_stop <= position["stop"]):
        decision = decision.model_copy(update={"action": "hold", "new_stop": None})
    return decision
