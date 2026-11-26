"""Planner (task 12.1): picks the stocks worth a full analysis today and skips the rest.

Offline rules score each stock: a code-detected setup (+2), results day 0 or a release in the last 2 days (+1),
an important announcement today (+1), the base model's 5-day p_up above 0.55 (+1). Stocks reaching
`min_planner_score` are ranked by score, then by p_up, and the top `max_stocks_per_day` are selected.
"""
from __future__ import annotations

from src.agents.base import AgentContext
from src.contracts import PlannerOutput, StockTask


def candidates(ctx: AgentContext) -> list[dict]:
    day = ctx.day
    rows = []
    for symbol in day.symbols:
        r = day.rows.loc[symbol]
        fc = day.base_forecast(symbol)
        p5 = fc.p_up.get(5) if fc else None
        reasons, score = [], 0
        if r["setup_type"] != "none":
            score += 2
            reasons.append(f"setup: {r['setup_type']}")
        if r["results_day0"] == 1 or r["days_since_results"] <= 2:
            score += 1
            reasons.append(f"results released {int(r['days_since_results'])} trading day(s) ago")
        if r["ann_important_day0"] > 0:
            score += 1
            reasons.append("important announcement since the last forecast")
        if p5 is not None and p5 > 0.55:
            score += 1
            reasons.append(f"base model 5-day p_up {p5:.2f}")
        rows.append({"symbol": symbol, "score": score, "p_up_5": p5, "setup_type": r["setup_type"],
                     "reasons": reasons})
    return rows


def plan(ctx: AgentContext) -> PlannerOutput:
    s = ctx.settings
    rows = candidates(ctx)

    def rules() -> PlannerOutput:
        keep = [r for r in rows if r["score"] >= s["min_planner_score"]]
        keep.sort(key=lambda r: (r["score"], r["p_up_5"] or 0), reverse=True)
        keep = keep[: s["max_stocks_per_day"]]
        return PlannerOutput(selected=[StockTask(symbol=r["symbol"], reasons=r["reasons"], setup_type=r["setup_type"])
                                       for r in keep],
                             skipped=len(rows) - len(keep),
                             notes=f"{len(keep)} of {len(rows)} stocks reached score {s['min_planner_score']}")

    shown = [r for r in rows if r["score"] > 0]
    result = ctx.llm.complete("planner", {"trade_date": str(ctx.day.trade_date.date()),
                                          "max_stocks": s["max_stocks_per_day"], "candidates": shown},
                              PlannerOutput, rules)
    out = result.output
    allowed = {r["symbol"] for r in rows}
    out.selected = [t for t in out.selected if t.symbol in allowed][: s["max_stocks_per_day"]]
    return out
